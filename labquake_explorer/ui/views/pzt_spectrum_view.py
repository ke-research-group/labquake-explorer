"""Displacement spectrum and omega-n (Brune) fit of one PZT / strain channel.

The view is a thin Tk shell around :mod:`labquake_explorer.analysis.spectrum`
and :mod:`labquake_explorer.analysis.source`: it reads
``event['strain']['original']['time'/'raw']`` and ``event['event_time']``
(or a picked per-channel arrival), collects and validates the user's
parameters, and calls

    extract_window -> compute_spectrum -> fit_omega_n -> source_parameters

Nothing numerical happens here.  Invalid entries are reported in the status
line, never in a dialog.  Results are stored under ``event['pzt_spectrum']``
as ``{"version": 1, "channels": {"ch<channel>": record}}`` so that several
channels of one event can be kept side by side (see :meth:`build_record`
for the record keys).  The ``ch`` prefix keeps the channel map from looking
like a list to loaders that turn all-digit keys into arrays.  Selecting a
channel with a saved record restores its parameters and shows the saved
numbers; nothing is recomputed until ``Compute`` is pressed.  A saved record
that cannot be displayed never prevents the window from opening: the
controls are left as they are and the status line says why.

Caveats reported by the analysis (``SpectralFit.warnings``: transient not
in the taper's flat region, band-limited corner, parameters at bounds, no
convergence) are shown in the ``flags`` row and in the status line; a fit
with warnings is never labelled "fit OK".  Because the amplitude convention
(``|rfft(w x)| dt`` without coherent-gain division) only holds when the
transient sits in the flat part of the taper, a pre window shorter than the
tukey ramp (``alpha/2`` of the whole window) is warned about at Compute
and ``Compute source`` refuses a plateau the analysis flagged as attenuated.

Typical medium values for the source section (granite-like rock):
rho = 2700 kg/m^3, Vp = 6000 m/s, Vs = 3500 m/s.  They are NOT defaults:
the entries start blank so the values used are always the user's own.
"""
from __future__ import annotations

import copy
import math
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np

from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

RESULT_VERSION = 1

TRIGGER_EVENT_TIME = "event_time"
TRIGGER_PICKED = "picked arrival"
TRIGGERS = (TRIGGER_EVENT_TIME, TRIGGER_PICKED)
CALIBRATION_UNITS = ("V/V", "V/m", "V/(m/s)", "V/(m/s^2)")

#: Tukey taper shape passed to ``compute_spectrum``; the first and last
#: ``alpha/2`` of the window are ramped, so the trigger sample (and the pulse
#: after it) must lie beyond that fraction for the plateau to be unattenuated.
TUKEY_ALPHA = 0.25

#: Prefix of the per-channel keys in ``event['pzt_spectrum']['channels']``.
CHANNEL_KEY_PREFIX = "ch"

#: Start of the analysis warning that flags an attenuated plateau.
FLAT_REGION_WARNING_PREFIX = "transient not in the taper's flat region"

#: Default control values (strings as typed into the entries).
DEFAULTS = {
    "pre_ms": "0.5", "post_ms": "2.0", "taper": "tukey", "baseline": "pre_linear",
    "highpass_hz": "", "remove_step": False,
    "calibration_path": "", "calibration_unit": "V/V", "calibration_db": False,
    "t_star_s": "",
    "fmin_hz": "1000", "fmax_hz": "", "bins_per_decade": "30", "snr_min": "3",
    "n_fixed": True, "loss": "linear",
}
FMAX_FRACTION_OF_FS = 0.4

_NA = "n/a"

#: Exceptions a malformed saved record may raise while being displayed.
_RECORD_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def _parse_float(text, name: str, required: bool = True, lo: Optional[float] = None,
                 strict_lo: bool = False) -> Optional[float]:
    """Parse an entry; ``None`` for a blank optional entry.  Raises ValueError
    with a message suitable for the status line."""
    s = str(text).strip()
    if s == "":
        if required:
            raise ValueError(f"{name}: enter a number")
        return None
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{name}: {s!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if lo is not None:
        if strict_lo and v <= lo:
            raise ValueError(f"{name} must be > {lo:g}")
        if not strict_lo and v < lo:
            raise ValueError(f"{name} must be >= {lo:g}")
    return v


def _parse_int(text, name: str, lo: int = 1) -> int:
    v = _parse_float(text, name, required=True, lo=lo)
    if v != int(v):
        raise ValueError(f"{name} must be an integer")
    return int(v)


def _text(value) -> str:
    """Entry text for a stored number (blank for None)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return ""
    return f"{v:.10g}"


def _fmt(value, fmt: str = ".4g", unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(v):
        return _NA
    s = format(v, fmt)
    return f"{s} {unit}".rstrip()


def _yes(flag) -> str:
    return "yes" if flag else "no"


def _as_list(value) -> list:
    """A plain Python list for a stored sequence (list, tuple or ndarray);
    ``[]`` for None or a scalar.  Never truth-tests an array."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(np.asarray(value, dtype=object).ravel())
    except (TypeError, ValueError):
        return []


def _band_pair(value) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` floats from a stored band, or None when unusable."""
    items = _as_list(value)
    if len(items) != 2:
        return None
    try:
        lo, hi = float(items[0]), float(items[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi


def _warning_texts(value) -> list[str]:
    return [str(w) for w in _as_list(value) if str(w).strip()]


def channel_key(channel: int) -> str:
    """Key of ``channel`` in ``event['pzt_spectrum']['channels']``."""
    return f"{CHANNEL_KEY_PREFIX}{int(channel)}"


def parse_channel_key(key) -> Optional[int]:
    """Channel number of a stored key (``'ch3'``, or a legacy ``'3'``); None otherwise."""
    s = str(key).strip()
    if s.lower().startswith(CHANNEL_KEY_PREFIX):
        s = s[len(CHANNEL_KEY_PREFIX):]
    return int(s) if s.isdigit() else None


@register_view("PZT Spectrum", kinds=[EVENT], order=40)
class PZTSpectrumView(EventView):
    """Window, spectrum, omega-n fit and source parameters for one channel."""

    window_title = "PZT Spectrum"
    result_key = "pzt_spectrum"

    def __init__(self, app, run_idx, event_idx):
        self.strain_time: Optional[np.ndarray] = None
        self.strain_raw: Optional[np.ndarray] = None
        self.window: Optional[sp.WindowResult] = None
        self.spectrum: Optional[sp.SpectrumResult] = None
        self.fit: Optional[sp.SpectralFit] = None
        self.record: Optional[dict] = None       # working record of the current channel
        self._first_load = True
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, padx=5, pady=5, sticky="nsw")
        middle = ttk.Frame(self)
        middle.grid(row=0, column=1, padx=5, pady=5, sticky="nsw")

        self._build_selection(left)
        self._build_window_frame(left)
        self._build_calibration_frame(left)
        self._build_fit_frame(left)
        self._build_buttons(left)
        self._build_results_frame(middle)
        self._build_source_frame(middle)

        figure_frame = ttk.Frame(self)
        figure_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        figure_frame.grid_columnconfigure(0, weight=1)
        figure_frame.grid_rowconfigure(0, weight=1)
        self.make_figure(master=figure_frame, figsize=(8, 8), figure_kwargs={"layout": "constrained"},
                         row=0, column=0, sticky="nsew")
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 1.0])
        self.ax_time = self.figure.add_subplot(gs[0])
        self.ax_spec = self.figure.add_subplot(gs[1])
        self.ax_resid = self.figure.add_subplot(gs[2])   # x limits copied from ax_spec in plot()
        self.axes = (self.ax_time, self.ax_spec, self.ax_resid)
        self._decorate_axes()
        self.canvas.draw()

    def _entry(self, master, label: str, var: tk.StringVar, row: int, width: int = 10, column: int = 0):
        ttk.Label(master, text=label).grid(row=row, column=column, padx=4, pady=2, sticky="e")
        entry = ttk.Entry(master, textvariable=var, width=width)
        entry.grid(row=row, column=column + 1, padx=4, pady=2, sticky="w")
        return entry

    def _build_selection(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Event / channel")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        ttk.Label(frame, text="Channel:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.channel_combobox = ttk.Combobox(frame, width=8, state="readonly")
        self.channel_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self.channel_combobox.bind("<<ComboboxSelected>>", self.on_channel_selected)
        ttk.Label(frame, text="Trigger:").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self.trigger_combobox = ttk.Combobox(frame, width=14, state="readonly", values=list(TRIGGERS))
        self.trigger_combobox.set(TRIGGER_EVENT_TIME)
        self.trigger_combobox.grid(row=2, column=1, padx=4, pady=2, sticky="w")

    def _build_window_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Window")
        frame.pack(fill="x", pady=4)
        self.pre_var = tk.StringVar(master=self, value=DEFAULTS["pre_ms"])
        self.post_var = tk.StringVar(master=self, value=DEFAULTS["post_ms"])
        self.highpass_var = tk.StringVar(master=self, value=DEFAULTS["highpass_hz"])
        self.remove_step_var = tk.BooleanVar(master=self, value=DEFAULTS["remove_step"])
        self._entry(frame, "Pre (ms):", self.pre_var, 0)
        self._entry(frame, "Post (ms):", self.post_var, 1)
        ttk.Label(frame, text="Taper:").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self.taper_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.TAPER_KINDS))
        self.taper_combobox.set(DEFAULTS["taper"])
        self.taper_combobox.grid(row=2, column=1, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="Baseline:").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        self.baseline_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.BASELINE_MODES))
        self.baseline_combobox.set(DEFAULTS["baseline"])
        self.baseline_combobox.grid(row=3, column=1, padx=4, pady=2, sticky="w")
        self._entry(frame, "Highpass (Hz):", self.highpass_var, 4)
        ttk.Checkbutton(frame, text="Remove coseismic step", variable=self.remove_step_var).grid(
            row=5, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text=f"tukey ramp = {TUKEY_ALPHA / 2:.0%} of the window: keep\n"
                              f"pre >= post / {(2.0 / TUKEY_ALPHA) - 1.0:g} or Omega0 is attenuated",
                  foreground="#606060", justify="left").grid(row=6, column=0, columnspan=2, padx=4, pady=2, sticky="w")

    def _build_calibration_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Calibration / attenuation")
        frame.pack(fill="x", pady=4)
        self.calibration_path_var = tk.StringVar(master=self, value=DEFAULTS["calibration_path"])
        self.calibration_db_var = tk.BooleanVar(master=self, value=DEFAULTS["calibration_db"])
        self.t_star_var = tk.StringVar(master=self, value=DEFAULTS["t_star_s"])
        ttk.Label(frame, text="CSV:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.calibration_entry = ttk.Entry(frame, textvariable=self.calibration_path_var, width=18)
        self.calibration_entry.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.browse_button = ttk.Button(frame, text="Browse", width=7, command=self.browse_calibration)
        self.browse_button.grid(row=0, column=2, padx=2, pady=2, sticky="w")
        ttk.Label(frame, text="Unit:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.unit_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(CALIBRATION_UNITS))
        self.unit_combobox.set(DEFAULTS["calibration_unit"])
        self.unit_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        ttk.Checkbutton(frame, text="gain in dB", variable=self.calibration_db_var).grid(
            row=1, column=2, padx=2, pady=2, sticky="w")
        self._entry(frame, "t* (s, blank = off):", self.t_star_var, 2)

    def _build_fit_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Fit band / omega-n fit")
        frame.pack(fill="x", pady=4)
        self.fmin_var = tk.StringVar(master=self, value=DEFAULTS["fmin_hz"])
        self.fmax_var = tk.StringVar(master=self, value=DEFAULTS["fmax_hz"])
        self.bins_var = tk.StringVar(master=self, value=DEFAULTS["bins_per_decade"])
        self.snr_var = tk.StringVar(master=self, value=DEFAULTS["snr_min"])
        self.n_fixed_var = tk.BooleanVar(master=self, value=DEFAULTS["n_fixed"])
        self._entry(frame, "fmin (Hz):", self.fmin_var, 0)
        self._entry(frame, "fmax (Hz):", self.fmax_var, 1)
        self._entry(frame, "bins / decade:", self.bins_var, 2)
        self._entry(frame, "SNR min (blank = off):", self.snr_var, 3)
        ttk.Checkbutton(frame, text="n fixed at 2", variable=self.n_fixed_var).grid(
            row=4, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="Loss:").grid(row=5, column=0, padx=4, pady=2, sticky="e")
        self.loss_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.LOSSES))
        self.loss_combobox.set(DEFAULTS["loss"])
        self.loss_combobox.grid(row=5, column=1, padx=4, pady=2, sticky="w")

    def _build_buttons(self, master) -> None:
        frame = ttk.Frame(master)
        frame.pack(fill="x", pady=4)
        self.compute_button = ttk.Button(frame, text="Compute", command=self.compute)
        self.compute_button.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(master, textvariable=self.status_var, wraplength=260, foreground="#a03030").pack(
            fill="x", pady=2)

    def _build_results_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Spectral fit")
        frame.pack(fill="x", pady=(0, 4))
        self.result_vars: dict[str, tk.StringVar] = {}
        rows = [("omega0", "Omega0:"), ("fc", "fc:"), ("n", "n:"), ("rms", "rms (ln):"),
                ("n_bins", "bins used:"), ("band", "band:"), ("taper", "taper energy:"),
                ("flags", "flags:")]
        for i, (key, label) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=4, pady=2, sticky="ne")
            var = tk.StringVar(master=self, value=_NA)
            ttk.Label(frame, textvariable=var, wraplength=220, justify="left").grid(
                row=i, column=1, padx=4, pady=2, sticky="w")
            self.result_vars[key] = var

    def _build_source_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Source parameters (SI)")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Phase:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.phase_combobox = ttk.Combobox(frame, width=6, state="readonly", values=list(src.PHASES))
        self.phase_combobox.set("P")
        self.phase_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.phase_combobox.bind("<<ComboboxSelected>>", self.on_phase_selected)
        self.rho_var = tk.StringVar(master=self, value="")
        self.c_var = tk.StringVar(master=self, value="")
        self.vs_var = tk.StringVar(master=self, value="")
        self.distance_var = tk.StringVar(master=self, value="")
        self.radiation_var = tk.StringVar(master=self, value=_text(src.PHASE_CONSTANTS["P"]["radiation_rms"]))
        self.radiation_hint_var = tk.StringVar(master=self, value="")
        self.free_surface_var = tk.StringVar(master=self, value="1.0")
        self._entry(frame, "rho (kg/m^3):", self.rho_var, 1)
        self._entry(frame, "Vp = c (m/s):", self.c_var, 2)
        self._entry(frame, "Vs (m/s):", self.vs_var, 3)
        self._entry(frame, "distance (m):", self.distance_var, 4)
        self._entry(frame, "radiation coeff.:", self.radiation_var, 5)
        ttk.Label(frame, textvariable=self.radiation_hint_var, foreground="#606060").grid(
            row=5, column=2, padx=2, pady=2, sticky="w")
        self._entry(frame, "free-surface factor:", self.free_surface_var, 6)
        ttk.Label(frame, text="M0 uses Vp for P, Vs for S; r = k Vs / fc.\n"
                              "e.g. granite: rho 2700, Vp 6000, Vs 3500",
                  foreground="#606060", justify="left").grid(row=7, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        self.source_button = ttk.Button(frame, text="Compute source", command=self.compute_source)
        self.source_button.grid(row=8, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
        self.source_vars: dict[str, tk.StringVar] = {}
        rows = [("m0", "M0:"), ("mw", "Mw:"), ("radius", "radius:"), ("stress_drop", "stress drop:"),
                ("kR", "kR:"), ("warnings", "warnings:")]
        for i, (key, label) in enumerate(rows, start=9):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=4, pady=2, sticky="ne")
            var = tk.StringVar(master=self, value=_NA)
            ttk.Label(frame, textvariable=var, wraplength=220, justify="left").grid(
                row=i, column=1, columnspan=2, padx=4, pady=2, sticky="w")
            self.source_vars[key] = var
        self._update_radiation_hint()

    def _decorate_axes(self) -> None:
        self.ax_time.set_xlabel("time from trigger (ms)")
        self.ax_time.set_ylabel("signal")
        self.ax_spec.set_ylabel("amplitude spectrum")
        self.ax_resid.set_xlabel("frequency (Hz)")
        self.ax_resid.set_ylabel("ln A - ln model")
        self.ax_spec.set_xscale("log")
        self.ax_spec.set_yscale("log")
        self.ax_resid.set_xscale("log")
        for ax in self.axes:
            ax.grid(True, linestyle="--", alpha=0.3)

    # ------------------------------------------------------------- loading
    def on_event_loaded(self) -> None:
        self.strain_time, self.strain_raw = self._strain_arrays()
        n_channels = 0 if self.strain_raw is None else int(self.strain_raw.shape[0])
        self.channel_combobox.config(values=[str(i) for i in range(n_channels)])
        saved = self.saved_channels()
        current = self.current_channel()
        if n_channels == 0:
            self.channel_combobox.set("")
            self.status_var.set("event has no strain/original time and raw arrays")
            self._clear_channel_state()
            return
        if self._first_load and saved and current not in saved:
            current = min(saved)
        if current is None or not (0 <= current < n_channels):
            current = 0
        self.channel_combobox.set(str(current))
        self._first_load = False
        if self.fmax_var.get().strip() == "":
            try:
                self.fmax_var.set(_text(FMAX_FRACTION_OF_FS * sp.sampling_rate(self.strain_time)))
            except ValueError:
                pass
        self.restore_channel()

    def _strain_arrays(self):
        strain = self.event.get("strain") if isinstance(self.event, dict) else None
        original = strain.get("original") if isinstance(strain, dict) else None
        if not isinstance(original, dict) or "time" not in original or "raw" not in original:
            return None, None
        try:
            time = np.asarray(original["time"], dtype=float).ravel()
            raw = np.asarray(original["raw"], dtype=float)
        except (TypeError, ValueError):
            return None, None
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or time.size < 2 or raw.shape[1] != time.size:
            return None, None
        return time, raw

    def current_channel(self) -> Optional[int]:
        text = self.channel_combobox.get().strip()
        return int(text) if text.isdigit() else None

    def set_channel(self, channel: int) -> None:
        """Select a channel (restoring its saved record if there is one)."""
        self.channel_combobox.set(str(int(channel)))
        self.on_channel_selected()

    def on_channel_selected(self, event=None) -> None:
        self.restore_channel()

    # -------------------------------------------------- radiation coefficient
    def radiation_is_default(self) -> bool:
        """True when the radiation entry holds a phase RMS value (any phase),
        i.e. the view put it there and nobody typed over it."""
        try:
            value = float(self.radiation_var.get().strip())
        except ValueError:
            return False
        return any(math.isclose(value, float(c["radiation_rms"]), rel_tol=1e-8)
                   for c in src.PHASE_CONSTANTS.values())

    def _update_radiation_hint(self) -> None:
        phase = self.phase_combobox.get()
        if phase in src.PHASES:
            self.radiation_hint_var.set(f"RMS for {phase}: {src.PHASE_CONSTANTS[phase]['radiation_rms']:.3f}")
        else:
            self.radiation_hint_var.set("")

    def on_phase_selected(self, event=None) -> None:
        """Keep the radiation entry at the phase RMS while it still holds a
        default; a typed (or restored non-default) value is never overwritten.
        The hint label always shows the RMS of the selected phase."""
        phase = self.phase_combobox.get()
        if phase in src.PHASES and self.radiation_is_default():
            self.radiation_var.set(_text(src.PHASE_CONSTANTS[phase]["radiation_rms"]))
        self._update_radiation_hint()

    # ----------------------------------------------------------- records
    def saved_channels(self) -> dict:
        """``{channel: record}`` parsed from ``event['pzt_spectrum']['channels']``
        (keys ``'ch<k>'``; legacy all-digit keys are accepted too).

        A legacy map keyed ``'0'..'n-1'`` comes back from an HDF5 file as a
        LIST (the loader turns contiguous digit keys into one); its entries
        are channels 0..n-1.  Only dict records count.
        """
        saved = self.load_results()
        channels = saved.get("channels") if saved else None
        out = {}
        if isinstance(channels, dict):
            items = list(channels.items())
        elif isinstance(channels, (list, tuple)):
            items = list(enumerate(channels))
        else:
            items = []
        for key, rec in items:
            k = parse_channel_key(key)
            if k is not None and isinstance(rec, dict):
                out[k] = rec
        return out

    def _clear_channel_state(self) -> None:
        self.window = self.spectrum = self.fit = None
        self.record = None
        self.show_fit(None)
        self.show_source(None)
        self.plot()

    def restore_channel(self) -> None:
        """Show the saved record of the current channel (parameters and
        numbers) if there is one; otherwise clear the results.  A record
        that cannot be displayed is reported in the status line and dropped
        (the controls keep whatever was applied), never raised."""
        self._clear_channel_state()
        channel = self.current_channel()
        rec = self.saved_channels().get(channel) if channel is not None else None
        if not isinstance(rec, dict):
            self.status_var.set(f"channel {channel}: no saved record - click Compute")
            return
        try:
            self.record = copy.deepcopy(rec)
            # a record is keyed by its channel; a hand-edited / partial record
            # without the field is filed under the channel it was found at
            self.record.setdefault("channel", int(channel))
            self.apply_record(self.record)
            self.show_record(self.record)
            self.show_source(self.record.get("source"))
            self.plot()
        except _RECORD_ERRORS as e:
            self.record = None
            self.show_fit(None)
            self.show_source(None)
            try:
                self.plot()
            except _RECORD_ERRORS:
                pass
            self.status_var.set(f"channel {channel}: saved record unreadable "
                                f"({type(e).__name__}: {e}) - click Compute")
            return
        self.status_var.set(f"channel {channel}: showing saved record (click Compute to recompute)")

    def apply_record(self, rec: dict) -> None:
        """Put a record's parameters into the controls."""
        g = rec.get
        self.trigger_combobox.set(g("trigger", TRIGGER_EVENT_TIME) if g("trigger") in TRIGGERS else TRIGGER_EVENT_TIME)
        self.pre_var.set(_text(g("pre_ms", DEFAULTS["pre_ms"])))
        self.post_var.set(_text(g("post_ms", DEFAULTS["post_ms"])))
        self.taper_combobox.set(g("taper") if g("taper") in sp.TAPER_KINDS else DEFAULTS["taper"])
        self.baseline_combobox.set(g("baseline") if g("baseline") in sp.BASELINE_MODES else DEFAULTS["baseline"])
        self.highpass_var.set(_text(g("highpass_hz")))
        self.remove_step_var.set(bool(g("remove_step", False)))
        self.calibration_path_var.set(str(g("calibration_path") or ""))
        self.unit_combobox.set(g("calibration_unit") if g("calibration_unit") in CALIBRATION_UNITS
                               else DEFAULTS["calibration_unit"])
        self.calibration_db_var.set(bool(g("calibration_db", False)))
        self.t_star_var.set(_text(g("t_star_s")))
        self.fmin_var.set(_text(g("fmin_hz", DEFAULTS["fmin_hz"])))
        self.fmax_var.set(_text(g("fmax_hz")))
        self.bins_var.set(_text(g("bins_per_decade", DEFAULTS["bins_per_decade"])))
        self.snr_var.set(_text(g("snr_min")))
        self.n_fixed_var.set(bool(g("n_fixed", True)))
        self.loss_combobox.set(g("loss") if g("loss") in sp.LOSSES else DEFAULTS["loss"])
        source = g("source")
        consts = source.get("constants") if isinstance(source, dict) else None
        if isinstance(consts, dict):
            phase = consts.get("phase")
            if phase in src.PHASES:
                self.phase_combobox.set(phase)
            self.rho_var.set(_text(consts.get("rho_kg_m3")))
            self.c_var.set(_text(consts.get("vp_m_s")))
            self.vs_var.set(_text(consts.get("vs_m_s")))
            self.distance_var.set(_text(consts.get("distance_m")))
            self.radiation_var.set(_text(consts.get("radiation_coefficient")))
            self.free_surface_var.set(_text(consts.get("free_surface_factor", 1.0)))
        self._update_radiation_hint()

    # ---------------------------------------------------------- parameters
    def read_parameters(self) -> dict:
        """Validated spectrum/fit parameters from the controls (ValueError on bad input)."""
        if self.strain_time is None or self.strain_raw is None:
            raise ValueError("event has no strain/original time and raw arrays")
        channel = self.current_channel()
        if channel is None or not (0 <= channel < self.strain_raw.shape[0]):
            raise ValueError("select a channel")
        trigger = self.trigger_combobox.get()
        if trigger not in TRIGGERS:
            raise ValueError("select a trigger")
        p = {
            "channel": int(channel),
            "trigger": trigger,
            "pre_ms": _parse_float(self.pre_var.get(), "pre window (ms)", lo=0.0),
            "post_ms": _parse_float(self.post_var.get(), "post window (ms)", lo=0.0),
            "taper": self.taper_combobox.get(),
            "baseline": self.baseline_combobox.get(),
            "highpass_hz": None,
            "remove_step": bool(self.remove_step_var.get()),
            "calibration_path": self.calibration_path_var.get().strip(),
            "calibration_unit": self.unit_combobox.get(),
            "calibration_db": bool(self.calibration_db_var.get()),
            "t_star_s": _parse_float(self.t_star_var.get(), "t* (s)", required=False, lo=0.0),
            "fmin_hz": _parse_float(self.fmin_var.get(), "fmin (Hz)", lo=0.0, strict_lo=True),
            "fmax_hz": _parse_float(self.fmax_var.get(), "fmax (Hz)", lo=0.0, strict_lo=True),
            "bins_per_decade": _parse_int(self.bins_var.get(), "bins per decade"),
            "snr_min": _parse_float(self.snr_var.get(), "SNR min", required=False, lo=0.0),
            "n_fixed": bool(self.n_fixed_var.get()),
            "loss": self.loss_combobox.get(),
        }
        if p["pre_ms"] + p["post_ms"] <= 0:
            raise ValueError("pre + post window must be > 0")
        if p["taper"] not in sp.TAPER_KINDS:
            raise ValueError("select a taper")
        if p["baseline"] not in sp.BASELINE_MODES:
            raise ValueError("select a baseline mode")
        if p["baseline"] == "highpass":
            p["highpass_hz"] = _parse_float(self.highpass_var.get(), "highpass cutoff (Hz)", lo=0.0, strict_lo=True)
        if p["calibration_unit"] not in CALIBRATION_UNITS:
            raise ValueError("select a calibration unit")
        if p["calibration_unit"] != "V/V" and not p["calibration_path"]:
            raise ValueError(f"calibration unit {p['calibration_unit']} needs a CSV file "
                             "(choose V/V for an uncalibrated spectrum)")
        if p["fmax_hz"] <= p["fmin_hz"]:
            raise ValueError("fmax must be larger than fmin")
        if p["loss"] not in sp.LOSSES:
            raise ValueError("select a loss")
        return p

    @staticmethod
    def parameter_warnings(p: dict) -> list[str]:
        """Caveats about a valid parameter set that do not stop the computation.

        With the tukey taper the first ``alpha/2`` of the window is ramped;
        a pre window shorter than that puts the trigger sample (and the
        pulse that follows it) on the ramp, so ``Omega0`` comes out low.
        """
        out = []
        if p.get("taper") == "tukey":
            total = float(p["pre_ms"]) + float(p["post_ms"])
            ramp_ms = 0.5 * TUKEY_ALPHA * total
            if float(p["pre_ms"]) < ramp_ms:
                out.append(f"pre window {float(p['pre_ms']):g} ms is shorter than the tukey ramp "
                           f"({ramp_ms:g} ms = {TUKEY_ALPHA / 2:.0%} of the window): the trigger sample "
                           f"is not in the taper's flat region and Omega0 is attenuated - use pre >= "
                           f"post / {(2.0 / TUKEY_ALPHA) - 1.0:g}")
        return out

    def trigger_time(self, p: dict) -> float:
        """Absolute trigger time for the parameters ``p`` (ValueError if unavailable)."""
        if p["trigger"] == TRIGGER_EVENT_TIME:
            try:
                t = float(self.event["event_time"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("event has no event_time") from None
        else:
            try:
                arrivals = np.asarray(self.event["strain"]["original"]["rupture_arrival_time"], dtype=float).ravel()
                t = float(arrivals[p["channel"]])
            except (KeyError, TypeError, ValueError, IndexError):
                raise ValueError(f"no picked arrival for channel {p['channel']} "
                                 "(strain/original/rupture_arrival_time)") from None
        if not math.isfinite(t):
            raise ValueError("trigger time is not finite")
        if not (self.strain_time[0] <= t <= self.strain_time[-1]):
            raise ValueError(f"trigger time {t:g} s lies outside the strain trace")
        return t

    def load_calibration(self, p: dict) -> Optional[sp.Calibration]:
        if not p["calibration_path"]:
            return None
        try:
            return sp.load_calibration_csv(p["calibration_path"], p["calibration_unit"], p["calibration_db"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"calibration CSV: {e}") from None
        except Exception as e:  # pandas parser errors etc.
            raise ValueError(f"calibration CSV could not be read: {type(e).__name__}: {e}") from None

    # ------------------------------------------------------------- compute
    def compute(self) -> bool:
        """Window + spectrum + fit for the current channel.  Returns True on success."""
        try:
            p = self.read_parameters()
            trigger = self.trigger_time(p)
            calibration = self.load_calibration(p)
            window = sp.extract_window(self.strain_time, self.strain_raw[p["channel"]], trigger,
                                       p["pre_ms"] * 1e-3, p["post_ms"] * 1e-3, baseline=p["baseline"],
                                       highpass_hz=p["highpass_hz"], remove_step=p["remove_step"])
            spectrum = sp.compute_spectrum(window, taper=p["taper"], alpha=TUKEY_ALPHA,
                                           calibration=calibration, t_star_s=p["t_star_s"],
                                           fmin=p["fmin_hz"], fmax=p["fmax_hz"],
                                           bins_per_decade=p["bins_per_decade"])
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        fit = sp.fit_omega_n(spectrum, snr_min=p["snr_min"], n_fixed=p["n_fixed"], loss=p["loss"])
        self.window, self.spectrum, self.fit = window, spectrum, fit
        self.record = self.build_record(p, trigger, spectrum, fit)
        self.show_record(self.record)
        self.show_source(None)
        self.plot()
        if not fit.valid:
            self.status_var.set(f"channel {p['channel']}: fit invalid - {fit.reason}")
            return True
        warnings = self.record_warnings(self.record)
        if warnings:
            self.status_var.set(f"channel {p['channel']}: fit has warnings ({fit.n_bins} bins) - not saved: "
                                + "; ".join(warnings))
        else:
            self.status_var.set(f"channel {p['channel']}: fit OK ({fit.n_bins} bins) - not saved")
        return True

    def build_record(self, p: dict, trigger_time: float, spectrum: sp.SpectrumResult,
                     fit: sp.SpectralFit) -> dict:
        """The JSON-like record stored per channel.

        Keys: version, channel, trigger, trigger_time_s, fs_hz, pre_ms, post_ms,
        taper, baseline, highpass_hz, remove_step, calibration_path,
        calibration_unit, calibration_db, t_star_s, fmin_hz, fmax_hz,
        bins_per_decade, snr_min, n_fixed, loss, parameter_warnings
        (:meth:`parameter_warnings` of the entries), spectrum
        (``SpectrumResult.as_dict()``: binned arrays and the window/taper
        meta incl. ``taper_energy_fraction``), fit (``SpectralFit.as_dict()``
        incl. ``warnings``), source (``SourceParameters.as_dict()`` or None).
        """
        rec = {
            "version": RESULT_VERSION,
            "channel": int(p["channel"]),
            "trigger": p["trigger"],
            "trigger_time_s": float(trigger_time),
            "fs_hz": float(spectrum.meta.get("fs", float("nan"))),
            "pre_ms": float(p["pre_ms"]),
            "post_ms": float(p["post_ms"]),
            "taper": p["taper"],
            "baseline": p["baseline"],
            "highpass_hz": p["highpass_hz"],
            "remove_step": bool(p["remove_step"]),
            "calibration_path": p["calibration_path"],
            "calibration_unit": p["calibration_unit"],
            "calibration_db": bool(p["calibration_db"]),
            "t_star_s": p["t_star_s"],
            "fmin_hz": float(p["fmin_hz"]),
            "fmax_hz": float(p["fmax_hz"]),
            "bins_per_decade": int(p["bins_per_decade"]),
            "snr_min": p["snr_min"],
            "n_fixed": bool(p["n_fixed"]),
            "loss": p["loss"],
            "parameter_warnings": self.parameter_warnings(p),
            "spectrum": spectrum.as_dict(),
            "fit": fit.as_dict(),
            "source": None,
        }
        return rec

    @staticmethod
    def record_warnings(rec) -> list[str]:
        """All caveats of a record: the fit's own warnings followed by the
        parameter warnings (empty for anything that is not a dict)."""
        if not isinstance(rec, dict):
            return []
        fitd = rec.get("fit")
        out = _warning_texts(fitd.get("warnings")) if isinstance(fitd, dict) else []
        out.extend(_warning_texts(rec.get("parameter_warnings")))
        return out

    @staticmethod
    def plateau_attenuated(rec) -> Optional[str]:
        """The reason the record's plateau is attenuated (transient outside the
        taper's flat region, per the analysis meta or the fit warnings), or None."""
        if not isinstance(rec, dict):
            return None
        spec = rec.get("spectrum")
        meta = spec.get("meta") if isinstance(spec, dict) else None
        if isinstance(meta, dict) and meta.get("transient_in_flat_region") is False:
            frac = meta.get("taper_energy_fraction")
            frac_text = _fmt(frac, ".2f")
            return f"{FLAT_REGION_WARNING_PREFIX} (taper_energy_fraction={frac_text})"
        for w in PZTSpectrumView.record_warnings(rec):
            if w.startswith(FLAT_REGION_WARNING_PREFIX):
                return w
        return None

    def read_source_inputs(self) -> dict:
        phase = self.phase_combobox.get()
        if phase not in src.PHASES:
            raise ValueError("select a phase")
        return {
            "phase": phase,
            "rho": _parse_float(self.rho_var.get(), "rho", lo=0.0, strict_lo=True),
            "vp": _parse_float(self.c_var.get(), "Vp (c)", lo=0.0, strict_lo=True),
            "vs": _parse_float(self.vs_var.get(), "Vs", lo=0.0, strict_lo=True),
            "distance_m": _parse_float(self.distance_var.get(), "distance", lo=0.0, strict_lo=True),
            "radiation_coefficient": _parse_float(self.radiation_var.get(), "radiation coefficient"),
            "free_surface_factor": _parse_float(self.free_surface_var.get(), "free-surface factor",
                                                lo=0.0, strict_lo=True),
        }

    def compute_source(self) -> bool:
        """Source parameters from the current record's fit.  Returns True on success.

        Refused (status message, no dialog) for an invalid fit, an
        uncalibrated spectrum (V*s) and a plateau the analysis flagged as
        attenuated (transient outside the taper's flat region): a moment from
        such a plateau would be biased low by the taper, not by physics.
        """
        rec = self.record
        fitd = rec.get("fit") if isinstance(rec, dict) else None
        if not isinstance(fitd, dict):
            self.status_var.set("compute a spectrum first")
            return False
        if not fitd.get("valid"):
            self.status_var.set("the spectral fit is invalid; no source parameters")
            return False
        if fitd.get("amp_units") != "m*s":
            self.status_var.set("spectrum is uncalibrated (V*s): load a calibration CSV with a physical "
                                "unit before computing source parameters")
            return False
        attenuated = self.plateau_attenuated(rec)
        if attenuated:
            self.status_var.set(f"Omega0 is attenuated ({attenuated}): lengthen the pre window so the "
                                "transient sits in the taper's flat region, then recompute before "
                                "computing source parameters")
            return False
        try:
            q = self.read_source_inputs()
            band = _band_pair(fitd.get("band_used"))
            result = src.source_parameters(
                fitd["omega0"], fitd["fc"], q["phase"], q["rho"], q["vp"], q["vs"], q["distance_m"],
                radiation_coefficient=q["radiation_coefficient"],
                free_surface_factor=q["free_surface_factor"],
                f_plateau_hz=None if band is None else band[0])
        except (ValueError, KeyError, TypeError) as e:
            self.status_var.set(f"source parameters: {e}" if not isinstance(e, ValueError) else str(e))
            return False
        rec["source"] = result.as_dict()
        self.show_source(rec["source"])
        if result.valid:
            self.status_var.set(f"source parameters computed ({q['phase']}) - not saved")
        else:
            self.status_var.set(f"source parameters invalid - {result.reason}")
        return bool(result.valid)

    # ---------------------------------------------------------------- save
    def save(self) -> bool:
        """Merge the current channel's record into event['pzt_spectrum']
        (keys ``'ch<k>'``; legacy digit keys of other channels are renamed)."""
        if not isinstance(self.record, dict):
            self.status_var.set("nothing to save - click Compute first")
            return False
        channel = self.record.get("channel")
        if channel is None:
            channel = self.current_channel()
        try:
            channel = int(channel)
        except (TypeError, ValueError):
            channel = None
        if channel is None:
            self.status_var.set("nothing saved - the record has no channel and none is selected")
            return False
        self.record["channel"] = channel
        channels = {channel_key(k): rec for k, rec in sorted(self.saved_channels().items())}
        channels[channel_key(channel)] = copy.deepcopy(self.record)
        self.save_results({"version": RESULT_VERSION, "channels": channels})
        self.status_var.set(f"channel {channel} saved ({len(channels)} channel(s) stored)")
        return True

    # ------------------------------------------------------------- display
    def show_record(self, rec: Optional[dict]) -> None:
        """Fit numbers, taper energy and all warnings of a record."""
        if not isinstance(rec, dict):
            self.show_fit(None)
            return
        spec = rec.get("spectrum")
        meta = spec.get("meta") if isinstance(spec, dict) else None
        self.show_fit(rec.get("fit"), meta=meta if isinstance(meta, dict) else None,
                      extra_warnings=_warning_texts(rec.get("parameter_warnings")))

    def show_fit(self, fitd: Optional[dict], meta: Optional[dict] = None, extra_warnings=()) -> None:
        """Fill the 'Spectral fit' rows.  ``meta`` (the spectrum meta) feeds
        the taper-energy row; ``extra_warnings`` are appended to the flags."""
        for var in self.result_vars.values():
            var.set(_NA)
        if isinstance(meta, dict):
            frac = meta.get("taper_energy_fraction")
            in_flat = meta.get("transient_in_flat_region")
            if frac is not None:
                self.result_vars["taper"].set(
                    f"fraction {_fmt(frac, '.3f')} (flat region: {_yes(in_flat)})")
        if not isinstance(fitd, dict) or not fitd.get("valid"):
            if isinstance(fitd, dict) and fitd.get("reason"):
                self.result_vars["flags"].set(f"invalid: {fitd['reason']}")
            return
        units = str(fitd.get("amp_units") or "")
        self.result_vars["omega0"].set(_fmt(fitd.get("omega0"), ".4g", units))
        self.result_vars["fc"].set(_fmt(fitd.get("fc"), ".4g", "Hz"))
        n_text = _fmt(fitd.get("n"), ".3g")
        self.result_vars["n"].set(n_text + (" (fixed)" if fitd.get("n_fixed") else " (free)"))
        self.result_vars["rms"].set(_fmt(fitd.get("rms"), ".3g"))
        self.result_vars["n_bins"].set(str(fitd.get("n_bins", _NA)))
        band = _band_pair(fitd.get("band_used"))
        if band is None:
            self.result_vars["band"].set(_NA)
        else:
            self.result_vars["band"].set(f"{_fmt(band[0], '.4g')} - {_fmt(band[1], '.4g')} Hz")
        at = fitd.get("at_bounds")
        at = at if isinstance(at, dict) else {}
        bound_names = [str(k) for k, v in at.items() if v] or ["none"]
        warnings = _warning_texts(fitd.get("warnings")) + list(extra_warnings)
        flags = (f"converged: {_yes(fitd.get('converged'))}; at bound: {', '.join(bound_names)}; "
                 f"band-limited: {_yes(fitd.get('band_limited'))}")
        if warnings:
            flags += "; WARNINGS: " + "; ".join(warnings)
        self.result_vars["flags"].set(flags)

    def show_source(self, sd: Optional[dict]) -> None:
        if not isinstance(sd, dict) or not sd.get("valid"):
            for var in self.source_vars.values():
                var.set(_NA)
            if isinstance(sd, dict) and sd.get("reason"):
                self.source_vars["warnings"].set(f"invalid: {sd['reason']}")
            return
        self.source_vars["m0"].set(_fmt(sd.get("seismic_moment_nm"), ".4g", "N m"))
        self.source_vars["mw"].set(_fmt(sd.get("mw"), ".3f"))
        self.source_vars["radius"].set(_fmt(sd.get("source_radius_m"), ".4g", "m"))
        stress = _fmt(sd.get("stress_drop_pa"), ".4g")
        self.source_vars["stress_drop"].set(
            _NA if stress == _NA else f"{float(sd.get('stress_drop_pa')) / 1e6:.4g} MPa")
        kR = sd.get("kR")
        ff = sd.get("far_field_ok")
        self.source_vars["kR"].set(_fmt(kR, ".3g") + ("" if ff is None else f" (far field {'ok' if ff else 'doubtful'})"))
        warnings = _warning_texts(sd.get("warnings"))
        self.source_vars["warnings"].set("; ".join(warnings) if warnings else "none")

    def plot(self) -> None:
        """Redraw the three axes from the live window/spectrum (when present)
        and the current record's binned spectrum and fit."""
        for ax in self.axes:
            ax.clear()
        self._decorate_axes()
        w, s, rec = self.window, self.spectrum, self.record

        if w is not None:
            self.ax_time.plot(w.t_rel * 1e3, w.signal, color="#1f77b4", lw=1.0, label="signal window")
            if w.noise_available:
                t_noise = (np.arange(-w.noise.size, 0) - w.n_pre) / w.fs * 1e3
                self.ax_time.plot(t_noise, w.noise, color="#7f7f7f", lw=0.8, label="noise window")
            self.ax_time.axvline(0.0, color="k", lw=0.8, ls=":")
            self.ax_time.legend(loc="upper right", fontsize=8)
        elif rec is not None:
            self.ax_time.text(0.5, 0.5, "saved record - click Compute to show the window",
                              ha="center", va="center", transform=self.ax_time.transAxes, fontsize=9)

        if s is not None:
            pos = s.f > 0
            with np.errstate(invalid="ignore"):
                self.ax_spec.loglog(s.f[pos], s.amp_signal[pos], color="#1f77b4", lw=0.6, alpha=0.35, label="signal (raw)")
                if np.any(np.isfinite(s.amp_noise)):
                    self.ax_spec.loglog(s.f[pos], s.amp_noise[pos], color="#7f7f7f", lw=0.6, alpha=0.5, label="noise (raw)")

        if isinstance(rec, dict) and isinstance(rec.get("spectrum"), dict):
            spec = rec["spectrum"]
            fitd = rec.get("fit") if isinstance(rec.get("fit"), dict) else {}
            fb = np.asarray(_as_list(spec.get("f_binned")), dtype=float)
            ab = np.asarray(_as_list(spec.get("amp_binned")), dtype=float)
            nb = np.asarray(_as_list(spec.get("noise_binned")), dtype=float)
            snr = np.asarray(_as_list(spec.get("snr")), dtype=float)
            units = str(spec.get("amp_units") or "")
            self.ax_spec.set_ylabel(f"amplitude ({units})" if units else "amplitude")
            fit_valid = bool(fitd.get("valid"))
            band = _band_pair(fitd.get("band_used")) if fit_valid else None
            try:
                model_params = (float(fitd["omega0"]), float(fitd["fc"]), float(fitd["n"])) if band else None
            except (KeyError, TypeError, ValueError):
                model_params = None
            if model_params is not None and not all(math.isfinite(v) for v in model_params):
                model_params = None
            if fb.size and ab.size == fb.size:
                used = np.isfinite(ab) & (ab > 0)
                try:
                    snr_min = None if rec.get("snr_min") is None else float(rec.get("snr_min"))
                except (TypeError, ValueError):
                    snr_min = None
                if snr_min is not None and snr.size == fb.size:
                    used &= np.isfinite(snr) & (snr >= snr_min)
                if band is not None:
                    used &= (fb >= band[0]) & (fb <= band[1])
                else:
                    used = np.zeros(fb.shape, dtype=bool)
                with np.errstate(invalid="ignore"):
                    if nb.size == fb.size and np.any(np.isfinite(nb)):
                        self.ax_spec.loglog(fb, nb, color="#4d4d4d", lw=1.0, ls="--", label="noise (binned)")
                    self.ax_spec.loglog(fb[~used], ab[~used], "o", mfc="none", mec="#1f77b4", ms=4,
                                        label="binned (unused)")
                    self.ax_spec.loglog(fb[used], ab[used], "o", color="#d62728", ms=4,
                                        label=f"binned, SNR >= {snr_min:g}" if snr_min is not None else "binned (used)")
                if model_params is not None:
                    omega0, fc, n = model_params
                    f_model = np.geomspace(band[0], band[1], 200)
                    model = sp.brune_spectrum(f_model, omega0, fc, n)
                    self.ax_spec.loglog(f_model, model, color="k", lw=1.5, label="omega-n model")
                    self.ax_spec.axvline(fc, color="k", lw=0.8, ls=":")
                    self.ax_spec.text(0.02, 0.05,
                                      f"Omega0 = {omega0:.3g} {units}\n"
                                      f"fc = {fc:.4g} Hz, n = {n:.2f}",
                                      transform=self.ax_spec.transAxes, fontsize=9, va="bottom",
                                      bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
                    with np.errstate(invalid="ignore", divide="ignore"):
                        resid = np.log(ab[used]) - np.log(sp.brune_spectrum(fb[used], omega0, fc, n))
                    self.ax_resid.semilogx(fb[used], resid, "o-", color="#d62728", ms=3, lw=0.8)
                self.ax_resid.axhline(0.0, color="k", lw=0.8)
            self.ax_spec.legend(loc="upper right", fontsize=8)
            self.ax_spec.set_title(f"{self.figure_title()} channel {rec.get('channel')}", fontsize=10)
            self.ax_resid.set_xlim(self.ax_spec.get_xlim())
        self.canvas.draw_idle()

    # ------------------------------------------------------------- dialogs
    def browse_calibration(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Calibration CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.calibration_path_var.set(path)
