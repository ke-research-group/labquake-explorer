"""Event extraction.

For every picked event the run's time history is sliced around the event
time, and each raw-data reference of the run (see
:mod:`labquake_explorer.data.sources`) is asked for its full-rate window.  The
result is one dict per event::

    event_time, time, <every run field aligned with time>,
    <channel array>: {'data': (n_channels, n_window), 'channels', 'unit', ...},   # e.g. 'raw_data', 'slip'
    <reference key>: {'format', 'filename', 'fields', 'sample_rate', 'block',
                      'original': {'time', 'raw'}}      # e.g. 'elsys', 'ni'

The PSU-era ``strain`` layout keeps its historical output (``time``/``raw``
downsampled copies plus ``original``), so older experiment files behave as
before.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from labquake_explorer.data.channels import is_channel_array, slice_channel_array
from labquake_explorer.data.sources import LegacyTpc5Source, Source, open_source, run_sources


class EventProcessor:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else None

    def set_data_path(self, data_path: Path) -> None:
        """The experiment file's path; raw-data references are relative to its folder."""
        self.data_path = Path(data_path) if data_path else None

    @property
    def base_dir(self) -> Optional[Path]:
        return None if self.data_path is None else self.data_path.parent

    # ------------------------------------------------------------ sources
    def open_sources(self, run_data: Dict[str, Any]) -> Dict[str, Source]:
        """Open every raw-data reference of the run that can be resolved."""
        refs = run_sources(run_data)
        if not refs:
            return {}
        if self.base_dir is None:
            print("Warning: experiment file path unknown; raw-data references skipped "
                  f"({', '.join(refs)}). Save the experiment first.")
            return {}
        sources = {}
        for key, ref in refs.items():
            try:
                sources[key] = open_source(ref, self.base_dir, run_data)
            except Exception as exc:
                print(f"Warning: cannot open raw data for '{key}': {exc}")
        return sources

    # ------------------------------------------------------------- events
    def extract_events(self, run_data: Dict[str, Any], event_indices: List[int], window: float,
                       pre: Optional[float] = None, post: Optional[float] = None,
                       dtype=np.float32) -> List[Dict]:
        """Slice the run around each picked index.

        ``window`` is the half-width in seconds; ``pre``/``post`` override it on
        each side.  Sources with their own ``event_window_s`` use that instead.
        """
        pre = float(window) if pre is None else float(pre)
        post = float(window) if post is None else float(post)
        sources = self.open_sources(run_data)
        time = np.asarray(run_data["time"], dtype=np.float64)
        events = []
        for i, idx in enumerate(event_indices):
            event_time = float(time[int(idx)])
            try:
                events.append(self.extract_event(run_data, event_time, pre, post, sources, dtype))
            except Exception as exc:
                print(f"Warning: event {i} at {event_time:.4f} s: {exc}")
                events.append({"event_time": event_time, "time": np.array([event_time]), "notes": [str(exc)]})
        return events

    def extract_event(self, run_data: Dict[str, Any], event_time: float, pre: float, post: float,
                      sources: Optional[Dict[str, Source]] = None, dtype=np.float32) -> Dict:
        time = np.asarray(run_data["time"], dtype=np.float64)
        n = time.size
        beg = int(np.argmin(np.abs(event_time - pre - time)))
        end = int(np.argmin(np.abs(event_time + post - time)))
        event: Dict[str, Any] = {"event_time": event_time, "time": time[beg:end]}
        sources = {} if sources is None else sources
        skip = set(sources) | set(run_sources(run_data))
        for key, value in run_data.items():
            if key in ("time", "events") or key in skip:
                continue
            if is_channel_array(value):
                if value["data"].shape[1] == n:
                    event[key] = slice_channel_array(value, slice(beg, end))
            elif isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == n:
                event[key] = value[beg:end]
        notes = []
        for key, source in sources.items():
            if isinstance(source, LegacyTpc5Source):
                block = self._legacy_strain(run_data, source, event_time, pre, post)
            else:
                win = source.event_window(event_time, pre, post, dtype=dtype)
                block = None if win is None else {
                    "format": source.format,
                    "filename": run_data[key].get("filename", source.path.name),
                    **win.as_dict(dtype),
                }
                if block is not None:
                    block["original"] = {"time": block.pop("time"), "raw": block.pop("raw")}
            if block is None:
                notes.append(f"{key}: no raw record covers {event_time:.4f} s")
            else:
                event[key] = block
        if notes:
            event["notes"] = notes
        return event

    # ------------------------------------------------------------- legacy
    @staticmethod
    def _legacy_strain(run_data, source: LegacyTpc5Source, event_time, pre, post) -> Optional[Dict]:
        """The PSU-era ``strain`` block: downsampled copies plus the baseline-corrected window."""
        ref = run_data["strain"]
        win = source.read_window(event_time - pre, event_time + post, dtype=np.float64)
        if win is None:
            return None
        raw = np.array(win.data, dtype=np.float64)
        head = max(1, raw.shape[1] // 100)
        raw -= raw[:, :head].mean(axis=1, keepdims=True)
        t_ds = np.asarray(ref["time"], dtype=np.float64)
        i0 = int(np.argmin(np.abs(t_ds - (event_time - pre - source.run_t0 - source.time_offset))))
        i1 = int(np.argmin(np.abs(t_ds - (event_time + post - source.run_t0 - source.time_offset))))
        sl = slice(i0, i1 + 1)
        return {
            "filename_downsampled": ref.get("filename_downsampled", ""),
            "filename": ref["filename"],
            "time": source.run_t0 + source.time_offset + t_ds[sl],
            "raw": np.asarray(ref["raw"])[:, sl],
            "original": {"time": win.time, "raw": raw},
        }

    # ------------------------------------------------------------- paths
    def get_data_at_path(self, data: Dict[str, Any], path: str) -> Any:
        """Get data at specified path"""
        current = data
        for key in path.split('/'):
            if key[0] == '[' and key[-1] == ']':
                key = int(key[1:-1])
            current = current[key]
        return current

    def set_data_at_path(self, data: Dict[str, Any], path: str, value: Any, add_key: bool = False) -> None:
        """Set data at specified path"""
        parts = path.split('/')
        current = data
        for i, part in enumerate(parts[:-1]):
            if part[0] == '[' and part[-1] == ']':
                part = int(part[1:-1])
            if part not in current and add_key:
                current[part] = {} if i < len(parts) - 2 else None
            current = current[part]
        last_key = parts[-1]
        if last_key[0] == '[' and last_key[-1] == ']':
            last_key = int(last_key[1:-1])
        current[last_key] = value
