"""Overlay several run-level signals against time, sample index or another signal.

A read-only preview of one run: every top-level run array with the same
length as ``run['time']`` can be selected in a multi-select listbox and drawn
on one axis, optionally normalised to [0, 1] so that quantities with mixed
units (MPa, um, um/s) share a scale.  Event times are marked with vertical
dotted lines.  Nothing is written back to the data; there is no result key.

Plotting happens only when the user clicks *Plot*, never on every keystroke.
A zoom the user applied in the toolbar survives a re-plot as long as the axis
it applies to still shows the same thing: the x zoom while the x quantity is
unchanged, the y zoom while the same set of signals is shown in the same
normalise state.  The autoscaled view is always recorded as the toolbar's
*Home* entry, so Home (or the *Reset view* button) returns to the full view.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Sequence

import numpy as np

from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.base import RunView

X_TIME = "time"
X_INDEX = "index"
LEGEND_LOC = "upper left"
EVENT_LINE_STYLE = dict(color="0.35", linestyle=":", linewidth=0.8, zorder=0)
DEFAULT_SIGNALS = ("shear_stress",)


# --------------------------------------------------------------------- helpers
def _aligned_array(value, n: int) -> Optional[np.ndarray]:
    """``value`` as a 1-D numeric array of length ``n``, or None if it is not one.

    Arrays containing NaN are accepted: matplotlib draws gaps for them.
    """
    if value is None or isinstance(value, (dict, str, bytes)):
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.shape[0] != n or arr.dtype.kind not in "iuf":
        return None
    return arr


def signal_candidates(run: dict) -> list[str]:
    """Top-level run keys holding 1-D numeric arrays as long as ``run['time']``.

    ``time`` itself is included; the order is the run's insertion order.
    Returns an empty list when the run has no usable ``time`` array.
    """
    if not isinstance(run, dict) or "time" not in run:
        return []
    time = _aligned_array(run["time"], len(run["time"])) if hasattr(run["time"], "__len__") else None
    if time is None or time.shape[0] == 0:
        return []
    n = time.shape[0]
    return [key for key, value in run.items()
            if isinstance(key, str) and _aligned_array(value, n) is not None]


def normalize01(y) -> np.ndarray:
    """Scale ``y`` to [0, 1] over its finite values; non-finite samples stay NaN.

    A constant (or all-NaN) series maps to zeros where finite.
    """
    y = np.asarray(y, dtype=float)
    out = np.full(y.shape, np.nan)
    finite = np.isfinite(y)
    if not finite.any():
        return out
    lo = float(y[finite].min())
    hi = float(y[finite].max())
    span = hi - lo
    if span > 0:
        out[finite] = (y[finite] - lo) / span
    else:
        out[finite] = 0.0
    return out


def _as_time(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def event_times(run: dict) -> np.ndarray:
    """One time per event of the run; NaN where an event has no usable time.

    Each entry of ``run['events']`` contributes its ``event_time``; an event
    without one falls back to ``run['time'][run['event_indices'][i]]`` for the
    same position ``i``.  Without an ``events`` list every ``event_indices``
    entry is used.  The length of the result is the number of events known to
    the run, so callers can report how many could not be placed.
    """
    time = np.asarray(run.get("time", []), dtype=float).ravel() if isinstance(run, dict) else np.array([])
    indices = run.get("event_indices") if isinstance(run, dict) else None
    try:
        idx = np.asarray(indices, dtype=float).ravel() if indices is not None else np.array([])
    except (TypeError, ValueError):
        idx = np.array([])

    def time_at(i: int) -> float:
        if i >= idx.size or not np.isfinite(idx[i]):
            return np.nan
        j = int(idx[i])
        if 0 <= j < time.size:
            return float(time[j])
        return np.nan

    events = run.get("events") if isinstance(run, dict) else None
    if isinstance(events, (list, tuple)) and len(events):
        out = np.full(len(events), np.nan)
        for i, event in enumerate(events):
            t = _as_time(event["event_time"]) if isinstance(event, dict) and "event_time" in event else np.nan
            out[i] = t if np.isfinite(t) else time_at(i)
        return out
    return np.array([time_at(i) for i in range(idx.size)], dtype=float)


def nearest_samples(time: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the finite sample of ``time`` closest to each of ``times``.

    ``time`` need not be sorted (it is sorted once here); non-finite samples
    are never chosen.  Entries with no finite sample available are -1.
    """
    time = np.asarray(time, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    out = np.full(times.shape, -1, dtype=int)
    finite_idx = np.flatnonzero(np.isfinite(time))
    ok = np.isfinite(times)
    if finite_idx.size == 0 or not ok.any():
        return out
    order = finite_idx[np.argsort(time[finite_idx], kind="stable")]
    sorted_time = time[order]
    pos = np.searchsorted(sorted_time, times[ok])
    right = np.clip(pos, 0, sorted_time.size - 1)
    left = np.clip(pos - 1, 0, sorted_time.size - 1)
    choose_left = np.abs(sorted_time[left] - times[ok]) <= np.abs(sorted_time[right] - times[ok])
    out[ok] = order[np.where(choose_left, left, right)]
    return out


def event_positions(run: dict, x: np.ndarray) -> np.ndarray:
    """X-coordinates of the run's markable events on the axis given by ``x``.

    Each event time from :func:`event_times` is mapped to the nearest sample
    of ``run['time']`` (which need not be sorted) and the value of ``x`` at
    that sample is returned, so markers stay correct when ``x`` is not time.
    Events without a usable time, and events whose ``x`` sample is not
    finite, are dropped; compare ``len(result)`` with ``event_times(run).size``
    to count them.
    """
    if not isinstance(run, dict):
        return np.array([], dtype=float)
    time = np.asarray(run.get("time", []), dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if time.size == 0 or x.shape != time.shape:
        return np.array([], dtype=float)
    times = event_times(run)
    if times.size == 0:
        return np.array([], dtype=float)
    nearest = nearest_samples(time, times)
    nearest = nearest[nearest >= 0]
    xs = x[nearest]
    return xs[np.isfinite(xs)]


# ------------------------------------------------------------------------ view
@register_view("Plot Run Signals", kinds=[RUN], order=10)
class RunSignalsView(RunView):
    """Overlay selected run signals on one axis with event markers.

    Pure preview: nothing is saved (``result_key`` is None).  Attributes that
    tests and callers may use: ``candidates`` (listbox entries),
    ``signal_lines`` (Line2D per plotted signal, in listbox order),
    ``event_lines`` (one Line2D per marked event), ``ax``.
    """

    window_title = "Run Signals"
    result_key = None

    def __init__(self, app, run_idx: int):
        self.candidates: list[str] = []
        self.signal_lines: list = []
        self.event_lines: list = []
        self._auto_limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._plotted_x: Optional[str] = None
        self._plotted_normalize: Optional[bool] = None
        self._plotted_keys: Optional[frozenset] = None
        super().__init__(app, run_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="ns")
        controls.grid_rowconfigure(1, weight=1)

        ttk.Label(controls, text="Signals:").grid(row=0, column=0, columnspan=2, sticky="w")
        list_frame = ttk.Frame(controls)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        self.signal_listbox = tk.Listbox(list_frame, selectmode="multiple", exportselection=False,
                                         width=24, height=12)
        self.signal_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.signal_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.signal_listbox.config(yscrollcommand=scrollbar.set)

        ttk.Label(controls, text="X axis:").grid(row=2, column=0, padx=(0, 4), pady=(8, 2), sticky="w")
        self.x_combo = ttk.Combobox(controls, state="readonly", width=18)
        self.x_combo.grid(row=2, column=1, pady=(8, 2), sticky="ew")

        self.normalize_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(controls, text="Normalize to [0, 1]", variable=self.normalize_var,
                        ).grid(row=3, column=0, columnspan=2, pady=2, sticky="w")
        self.mark_events_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(controls, text="Mark events", variable=self.mark_events_var,
                        ).grid(row=4, column=0, columnspan=2, pady=2, sticky="w")

        self.plot_button = ttk.Button(controls, text="Plot", command=self.plot)
        self.plot_button.grid(row=5, column=0, columnspan=2, pady=(8, 2), sticky="ew")
        self.reset_button = ttk.Button(controls, text="Reset view", command=self.reset_view)
        self.reset_button.grid(row=6, column=0, columnspan=2, pady=2, sticky="ew")
        self.status_var = tk.StringVar(master=self, value="Preview only - nothing is saved")
        ttk.Label(controls, textvariable=self.status_var, wraplength=180).grid(
            row=7, column=0, columnspan=2, pady=(4, 0), sticky="w")

        self.make_figure(figsize=(9, 6), row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.ax = self.figure.add_subplot(111)

    def on_run_loaded(self) -> None:
        self.refresh_candidates()
        if not self.selected_signals():
            defaults = [k for k in DEFAULT_SIGNALS if k in self.candidates]
            self.select_signals(defaults or self.candidates[:1])
        self.plot()

    # ----------------------------------------------------------- selection
    def refresh_candidates(self) -> None:
        """Rebuild the listbox and the X-axis choices from the current run."""
        previous = set(self.selected_signals())
        previous_x = self.x_combo.get()
        all_aligned = signal_candidates(self.run)
        self.candidates = [k for k in all_aligned if k != X_TIME]
        self.signal_listbox.delete(0, tk.END)
        for key in self.candidates:
            self.signal_listbox.insert(tk.END, key)
        self.select_signals([k for k in self.candidates if k in previous])
        x_values = [X_TIME, X_INDEX] + self.candidates if X_TIME in all_aligned else [X_INDEX] + self.candidates
        self.x_combo.config(values=x_values)
        if previous_x in x_values:
            self.x_combo.set(previous_x)
        else:
            self.x_combo.set(X_TIME if X_TIME in x_values else X_INDEX)

    def selected_signals(self) -> list[str]:
        return [self.candidates[int(i)] for i in self.signal_listbox.curselection()
                if int(i) < len(self.candidates)]

    def select_signals(self, keys: Sequence[str]) -> None:
        """Select exactly ``keys`` in the listbox (unknown keys are ignored)."""
        self.signal_listbox.selection_clear(0, tk.END)
        for key in keys:
            if key in self.candidates:
                self.signal_listbox.selection_set(self.candidates.index(key))

    # ----------------------------------------------------------------- data
    def x_data(self) -> tuple[np.ndarray, str]:
        """The X array for the current combobox choice and its axis label.

        Falls back to the sample index when the chosen array is missing or no
        longer a 1-D numeric array aligned with ``run['time']``.
        """
        choice = self.x_combo.get() or X_TIME
        run = self.run if isinstance(self.run, dict) else {}
        time = run.get("time")
        n = len(time) if hasattr(time, "__len__") else 0
        if choice != X_INDEX and choice in run:
            arr = _aligned_array(run.get(choice), n)
            if arr is not None:
                return arr.astype(float), choice
        return np.arange(n, dtype=float), "sample index"

    def _zoomed(self) -> tuple[bool, bool]:
        """Whether the user changed the x / y limits away from the autoscaled ones."""
        if self._auto_limits is None:
            return False, False
        auto_x, auto_y = self._auto_limits
        cur_x, cur_y = self.ax.get_xlim(), self.ax.get_ylim()
        return (not np.allclose(cur_x, auto_x, rtol=1e-9, atol=0.0),
                not np.allclose(cur_y, auto_y, rtol=1e-9, atol=0.0))

    def reset_view(self) -> None:
        """Forget any zoom and re-plot at the autoscaled limits."""
        self._auto_limits = None
        self.plot()

    # ----------------------------------------------------------------- plot
    def plot(self) -> None:
        """Redraw the axis from the current selection (called by the Plot button).

        Never raises on a run that changed since the listbox was filled:
        signals that are missing or no longer aligned with the x axis are
        skipped and named in the status line, and the listbox is refreshed.
        """
        keys = self.selected_signals()
        normalize = bool(self.normalize_var.get())
        mark_events = bool(self.mark_events_var.get())
        x, xlabel = self.x_data()
        x_choice = self.x_combo.get() or X_TIME
        n = x.shape[0]

        run = self.run if isinstance(self.run, dict) else {}
        usable: list[tuple[str, np.ndarray]] = []
        skipped: list[str] = []
        for key in keys:
            arr = _aligned_array(run.get(key), n)
            if arr is None:
                skipped.append(key)
            else:
                usable.append((key, arr.astype(float)))
        plotted_keys = frozenset(k for k, _ in usable)

        keep_x, keep_y = self._zoomed()
        prev_xlim, prev_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        keep_x = keep_x and x_choice == self._plotted_x
        keep_y = keep_y and normalize == self._plotted_normalize and plotted_keys == self._plotted_keys

        self.ax.clear()
        self.signal_lines = []
        self.event_lines = []
        for key, y in usable:
            if normalize:
                y = normalize01(y)
            line, = self.ax.plot(x, y, linewidth=1.0, label=key)
            self.signal_lines.append(line)

        n_events_total = 0
        if mark_events:
            n_events_total = int(event_times(run).size) if run.get("time") is not None else 0
            for xe in event_positions(run, x):
                self.event_lines.append(self.ax.axvline(xe, **EVENT_LINE_STYLE))

        self.ax.set_xlabel(xlabel)
        if normalize:
            self.ax.set_ylabel("normalized [0, 1]")
        elif len(usable) == 1:
            self.ax.set_ylabel(usable[0][0])
        else:
            self.ax.set_ylabel("value")
        self.ax.set_title(f"{self.experiment_name()} run{self.run_idx:02d}".strip())
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        if self.signal_lines:
            self.ax.legend(loc=LEGEND_LOC)

        self.ax.relim()
        self.ax.autoscale_view()
        self._auto_limits = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            toolbar.update()          # forget the zoom history of the previous plot
            toolbar.push_current()    # the autoscaled view is the new Home
        if keep_x:
            self.ax.set_xlim(prev_xlim)
        if keep_y:
            self.ax.set_ylim(prev_ylim)
        if toolbar is not None and (keep_x or keep_y):
            toolbar.push_current()    # ... and the preserved zoom the current entry
        self._plotted_x = x_choice
        self._plotted_normalize = normalize
        self._plotted_keys = plotted_keys
        self.canvas.draw_idle()

        n_marked = len(self.event_lines)
        parts: list[str] = []
        if usable:
            parts.append(f"{len(usable)} signal(s), {n_marked} event marker(s).")
        else:
            parts.append("Select one or more signals and click Plot.")
        if mark_events and n_events_total > n_marked:
            parts.append(f"{n_events_total - n_marked} event(s) not markable.")
        if skipped:
            parts.append(f"Skipped {', '.join(skipped)}: missing or not aligned with the x axis.")
        parts.append("Preview only - nothing is saved")
        self.status_var.set(" ".join(parts))
        if skipped:
            self.refresh_candidates()
