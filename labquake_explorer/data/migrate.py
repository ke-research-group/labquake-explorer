"""Bring an experiment saved by an older explorer up to the current layout.

``upgrade_experiment(data)`` edits the loaded dict in place and returns one
note per change; ``upgrade_file(path)`` does it for a file (keeping a copy of
the original next to it).  Run it from the command line::

    python -m labquake_explorer.data.migrate experiment.h5 [--no-backup] [--lvdt-um-per-v=-5040]

``--lvdt-um-per-v`` converts an ``LP_displacement`` still in LVDT volts to
micrometres in every run and event (``recalibrate_lp_displacement``), for
files made before the LVDT slope was known.

What is brought up to date (see ``docs/extending.md`` for the layouts):

* runs: ``file``, ``file_index``, ``ni_file`` and ``sources`` go (each
  recorder block has ``filename`` and ``format``); the NI reference drops the
  file's own ``labels`` and ``member_offsets``; a top-level ``event_indices``
  and an ``event_window`` record merge into ``event_extraction``; a
  ``displacement`` field next to a ``slip`` channel array goes (it duplicated
  the first slip channel).
* events: the event-level ``pzt_spectrum`` map moves into the records it was
  computed on (``waveform/<recorder>/spectrum``); the recorders' ``raw_data``
  slices go (processed channels only);
  ``high_rate``/``low_rate`` and the older per-recorder blocks with
  ``original`` become ``waveform/<recorder>`` records; every record gets its
  ``trigger_time`` and sensor ``positions``; arrival picks kept among a
  record's data keys move under its ``arrivals``; ``event_analysis`` records
  of version 1 or 2 become version 3; PZT spectrum records point at
  ``waveform/<recorder>``.

The PSU-era ``strain`` layout is left as it is: the views read it directly.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from labquake_explorer.analysis.event_metrics import RESULT_VERSION as ANALYSIS_VERSION
from labquake_explorer.data.channels import channel_array, is_channel_array, positions_for
from labquake_explorer.data.picks import EXTRACTION_KEY, LEGACY_INDICES_KEY
from labquake_explorer.data.sources import (WAVEFORM, ARRIVALS_KEY, is_legacy_strain, is_waveform_block,
                                            read_arrivals, run_sources, waveform_trigger_time, write_arrivals)

RUN_KEYS_DROPPED = ("file", "file_index", "ni_file", "sources")
NI_KEYS_DROPPED = ("labels", "member_offsets")
LEGACY_WINDOW_KEY = "event_window"
LP_FIELD = "LP_displacement"
INTERIM_ARRIVAL_KEYS = ("picked_idx", "rupture_arrival_time", "enabled_channels", "fitting_channels", "locations")


def _recorders(run: Mapping) -> list:
    """Reference keys of the run's recorders, the PSU ``strain`` excluded."""
    return [key for key, ref in run_sources(run).items() if not is_legacy_strain(ref)]


def _positions_all_nan(positions) -> bool:
    try:
        return bool(np.all(np.isnan(np.asarray(positions["x"], dtype=float))))
    except (KeyError, TypeError, ValueError):
        return True


# ------------------------------------------------------------------ runs
def upgrade_run(run: dict, where: str = "run") -> list:
    notes = []
    for key in RUN_KEYS_DROPPED:
        if key in run:
            del run[key]
            notes.append(f"{where}: dropped {key}")
    for key in _recorders(run):
        for sub in NI_KEYS_DROPPED:
            if sub in run[key]:
                del run[key][sub]
                notes.append(f"{where}/{key}: dropped {sub}")
    legacy_picks = run.pop(LEGACY_INDICES_KEY, None)
    legacy_window = run.pop(LEGACY_WINDOW_KEY, None)
    if legacy_picks is not None or legacy_window is not None:
        record = dict(run.get(EXTRACTION_KEY) or {})
        if isinstance(legacy_window, Mapping):
            for k, v in legacy_window.items():
                record.setdefault(k, v)
        if legacy_picks is not None:
            record["event_indices"] = [int(i) for i in np.asarray(legacy_picks).ravel()]
        record["version"] = 1
        run[EXTRACTION_KEY] = record
        notes.append(f"{where}: picks and window moved into {EXTRACTION_KEY}")
    if "displacement" in run and is_channel_array(run.get("slip")):
        del run["displacement"]
        units = run.get("units")
        if isinstance(units, dict):
            units.pop("displacement", None)
        notes.append(f"{where}: dropped displacement (slip channels carry it)")
    events = run.get("events")
    if isinstance(events, (list, tuple)):
        for j, event in enumerate(events):
            if isinstance(event, dict):
                notes += upgrade_event(event, run, f"{where}/events/[{j}]")
    return notes


# ---------------------------------------------------------------- events
def _record_from_v2_block(block: Mapping, run: Mapping, key: str):
    """A waveform record from the 2026-10-04 layout ``{format, filename, fields,
    sample_rate, block, original: {time, raw}}``; None when it has no record."""
    original = block.get("original")
    if not isinstance(original, Mapping) or "time" not in original or "raw" not in original:
        return None
    raw = np.asarray(original["raw"])
    raw = raw[None, :] if raw.ndim == 1 else raw
    fields = block.get("fields")
    channels = [str(f) for f in fields] if fields is not None and len(fields) == raw.shape[0] else [str(i) for i in range(raw.shape[0])]
    ref = run.get(key, {})
    raw_run = ref.get("raw_data") if isinstance(ref, Mapping) else None
    meta = {"time": np.asarray(original["time"], dtype=np.float64), "sample_rate": float(block.get("sample_rate", 0.0)),
            "filename": str(block.get("filename", ref.get("filename", "") if isinstance(ref, Mapping) else ""))}
    if block.get("block") is not None:
        meta["block"] = int(block["block"])
    unit = str(raw_run.get("unit", "V")) if is_channel_array(raw_run) else "V"
    positions = positions_for(raw_run, channels) if is_channel_array(raw_run) else None
    record = channel_array(raw, channels, unit=unit, positions=positions, dtype=np.float32, **meta)
    arrivals = {}
    if "picked_idx" in original:
        arrivals["picked_idx"] = [int(i) for i in np.asarray(original["picked_idx"]).ravel()]
    if "rupture_arrival_time" in original:
        arrivals["arrival_time"] = np.asarray(original["rupture_arrival_time"], dtype=float)
    for flag in ("enabled_channels", "fitting_channels", "locations"):
        if flag in block:
            arrivals[flag] = list(block[flag])
    return record, arrivals


def _upgrade_analysis(result: dict) -> bool:
    """Version 1 or 2 event_analysis -> version 3 in place; True when changed."""
    if not isinstance(result, dict) or int(result.get("version", 1) or 1) >= ANALYSIS_VERSION:
        return False
    try:
        loading = [int(result["loading_indices"][0]), int(result["loading_indices"][1])]
        unloading = [int(result["unloading_indices"][0]), int(result["unloading_indices"][1])]
        delta = [int(result["rupture_start_index"]), int(result["rupture_end_index"])]
    except (KeyError, IndexError, TypeError, ValueError):
        return False

    def number(key):
        try:
            return float(result[key])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    stress_drop, displacement = number("stress_drop"), number("displacement")
    new = {
        "version": ANALYSIS_VERSION,
        "x_field": str(result.get("x_field", "")),
        "y_field": str(result.get("y_field", "")),
        "loading_indices": loading,
        "unloading_indices": unloading,
        "delta_indices": delta,
        "loading_slope": number("loading_stiffness"),
        "unloading_slope": number("unloading_stiffness"),
        "delta_x": displacement,
        "delta_y": -stress_drop if np.isfinite(stress_drop) else float("nan"),
    }
    result.clear()
    result.update(new)
    return True


def upgrade_event(event: dict, run: Mapping, where: str = "event") -> list:
    notes = []
    recorders = _recorders(run)
    waveforms = event.get(WAVEFORM)
    if not isinstance(waveforms, dict):
        waveforms = dict(event.pop(WAVEFORM, None) or {}) if isinstance(event.get(WAVEFORM), Mapping) else {}
    # interim names from 2026-10-06
    if isinstance(event.get("high_rate"), Mapping):
        waveforms.update(event.pop("high_rate"))
        notes.append(f"{where}: high_rate renamed to {WAVEFORM}")
    if "low_rate" in event:
        del event["low_rate"]
        notes.append(f"{where}: dropped low_rate")
    # per-recorder blocks: the raw slice goes, a full-rate 'original' becomes a waveform record
    pending_arrivals = {}
    for key in recorders:
        block = event.get(key)
        if not isinstance(block, Mapping):
            continue
        made = _record_from_v2_block(block, run, key)
        if made is not None:
            waveforms[key], pending_arrivals[key] = made
            notes.append(f"{where}: {key}/original became {WAVEFORM}/{key}")
        else:
            notes.append(f"{where}: dropped {key} (raw_data slice)")
        del event[key]
    if waveforms:
        event[WAVEFORM] = waveforms
    # every record: trigger time, positions, arrivals under their own key
    for key, record in list(waveforms.items()):
        if not is_waveform_block(record):
            continue
        if record.get("trigger_time") is None and record.get("block") is not None:
            t_trig = waveform_trigger_time(record, run, f"{WAVEFORM}/{key}")
            if t_trig is not None:
                record["trigger_time"] = t_trig
                notes.append(f"{where}: {WAVEFORM}/{key} trigger_time from the run's block table")
        raw_run = run.get(key, {}).get("raw_data") if isinstance(run.get(key), Mapping) else None
        if is_channel_array(raw_run) and ("positions" not in record or _positions_all_nan(record["positions"])):
            positions = positions_for(raw_run, [str(c) for c in record["channels"]])
            if positions is not None and not _positions_all_nan(positions):
                record["positions"] = positions
                notes.append(f"{where}: {WAVEFORM}/{key} positions from the run")
        arrivals = dict(pending_arrivals.get(key, {}))
        if ARRIVALS_KEY not in record and any(k in record for k in INTERIM_ARRIVAL_KEYS):
            arrivals.update(read_arrivals(record))
            for k in INTERIM_ARRIVAL_KEYS:
                record.pop(k, None)
        if arrivals.get("picked_idx") is not None:
            if "rupture_speed" in event and "strain" not in event and len(waveforms) == 1:
                arrivals["rupture_speed"] = event.pop("rupture_speed")
            write_arrivals(record, arrivals, event)
            notes.append(f"{where}: {WAVEFORM}/{key} arrival picks moved under {ARRIVALS_KEY}")
        else:
            for k in INTERIM_ARRIVAL_KEYS:
                if record.pop(k, None) is not None:
                    notes.append(f"{where}: {WAVEFORM}/{key} dropped stray {k}")
    if "displacement" in event and is_channel_array(event.get("slip")):
        del event["displacement"]
        notes.append(f"{where}: dropped displacement")
    if _upgrade_analysis(event.get("event_analysis")):
        notes.append(f"{where}: event_analysis upgraded to version {ANALYSIS_VERSION}")
    spectrum = event.get("pzt_spectrum")
    channels = spectrum.get("channels") if isinstance(spectrum, Mapping) else None
    if isinstance(channels, Mapping):
        items = list(channels.items())
    elif isinstance(channels, (list, tuple)):
        items = list(enumerate(channels))
    else:
        items = []
    for _, rec in items:
        if isinstance(rec, dict) and rec.get("record") in recorders:
            rec["record"] = f"{WAVEFORM}/{rec['record']}"
            notes.append(f"{where}: pzt_spectrum record renamed to {rec['record']}")
    # the event-level map moves under the record each channel was computed on
    records = {k: b for k, b in _waveform_entries(event).items()}
    if items and records:
        left = []
        for key, rec in items:
            m = re.fullmatch(r"(?:ch)?(\d+)", str(key), re.I)
            if not isinstance(rec, dict) or m is None:
                left.append((key, rec))
                continue
            target = rec.get("record") if rec.get("record") in records else next(iter(records))
            store = records[target].setdefault("spectrum", {"version": 1, "channels": {}})
            if not isinstance(store.get("channels"), dict):
                store["channels"] = {}
            store["channels"][f"ch{int(m.group(1))}"] = rec
            notes.append(f"{where}: pzt_spectrum channel {key} moved under {target}/spectrum")
        if left:
            spectrum["channels"] = dict(left) if isinstance(channels, Mapping) else [r for _, r in left]
        else:
            del event["pzt_spectrum"]
    return notes


def _waveform_entries(event: Mapping) -> dict:
    """``{key: block}`` of the event's records: ``'strain'`` and ``'waveform/<recorder>'``."""
    out = {}
    if is_waveform_block(event.get("strain")):
        out["strain"] = event["strain"]
    waveforms = event.get(WAVEFORM)
    if isinstance(waveforms, Mapping):
        for recorder, block in waveforms.items():
            if is_waveform_block(block):
                out[f"{WAVEFORM}/{recorder}"] = block
    return out


# ------------------------------------------------------------ experiment
def upgrade_experiment(data: dict) -> list:
    """Upgrade every run of a loaded experiment dict in place; returns the notes."""
    notes = []
    runs = data.get("runs") if isinstance(data, Mapping) else None
    if isinstance(runs, (list, tuple)):
        for i, run in enumerate(runs):
            if isinstance(run, dict):
                notes += upgrade_run(run, f"runs/[{i}]")
    return notes


def recalibrate_lp_displacement(data: dict, um_per_v: float) -> list:
    """Turn an ``LP_displacement`` still in LVDT volts into micrometres, in
    every run (and its events) whose unit says ``V``; the calibration record's
    step and note follow.  Runs already in micrometres are left alone, so the
    conversion can be run twice without harm."""
    notes = []
    factor = float(um_per_v)
    runs = data.get("runs") if isinstance(data, Mapping) else None
    for i, run in enumerate(runs if isinstance(runs, (list, tuple)) else []):
        if not isinstance(run, dict) or LP_FIELD not in run:
            continue
        units = run.get("units") if isinstance(run.get("units"), dict) else {}
        if str(units.get(LP_FIELD, "")) != "V":
            continue
        run[LP_FIELD] = (np.asarray(run[LP_FIELD], dtype=np.float64) * factor).astype(np.float32)
        units[LP_FIELD] = "um"
        run["units"] = units
        calibration = run.get("calibration")
        if isinstance(calibration, dict):
            for step in calibration.get("steps") or []:
                if isinstance(step, dict) and step.get("field") == LP_FIELD and step.get("kind") == "linear":
                    step["factor"] = float(step.get("factor", 1.0)) * factor
                    step["unit"] = "um"
                    step["formula"] = (f"{LP_FIELD} (um) = {step['factor']:.6g} * {step.get('source', 'lvdt')} "
                                       f"+ {float(step.get('offset', 0.0)):.6g}")
            note = calibration.get("note")
            if isinstance(note, str) and "LVDT" in note:
                calibration["note"] = re.sub(r"LVDT:[^;]*", f"LVDT: {factor:g} um/V", note)
        n_events = n_stale = 0
        for event in run.get("events") or []:
            if isinstance(event, dict) and LP_FIELD in event:
                event[LP_FIELD] = (np.asarray(event[LP_FIELD], dtype=np.float64) * factor).astype(np.float32)
                n_events += 1
                analysis = event.get("event_analysis")
                if isinstance(analysis, dict) and LP_FIELD in (analysis.get("x_field"), analysis.get("y_field")):
                    n_stale += 1
        notes.append(f"runs/[{i}]: {LP_FIELD} converted from V to um with {factor:g} um/V ({n_events} events)")
        if n_stale:
            notes.append(f"runs/[{i}]: {n_stale} event_analysis record(s) were computed on {LP_FIELD} in volts; "
                         "reopen them in the Event Analyzer to recompute")
    return notes


def upgrade_file(path, backup: bool = True, lvdt_um_per_v=None) -> list:
    """Load an experiment file, upgrade it (and convert ``LP_displacement`` to
    micrometres when ``lvdt_um_per_v`` is given) and write it back.  The first
    time a file is changed a copy of the original is kept as
    ``<name>.before-upgrade<suffix>``; an existing copy is never overwritten."""
    from labquake_explorer.data.data_manager import DataManager
    path = Path(path)
    dm = DataManager()
    dm.load_file(path)
    notes = upgrade_experiment(dm.data)
    if lvdt_um_per_v is not None:
        notes += recalibrate_lp_displacement(dm.data, float(lvdt_um_per_v))
    if not notes:
        return notes
    copy = path.with_name(f"{path.stem}.before-upgrade{path.suffix}")
    if backup and not copy.exists():
        shutil.copy2(path, copy)
    dm.save_file(path)
    return notes


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    backup = "--no-backup" not in args
    lvdt = None
    for a in args:
        if a.startswith("--lvdt-um-per-v="):
            lvdt = float(a.split("=", 1)[1])
    files = [a for a in args if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    for name in files:
        notes = upgrade_file(name, backup=backup, lvdt_um_per_v=lvdt)
        print(f"{name}: {len(notes)} change(s)" + ("" if notes else " (already up to date)"))
        for note in notes:
            print("  " + note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
