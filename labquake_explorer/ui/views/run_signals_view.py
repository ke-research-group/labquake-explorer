"""Overlay several run-level signals against time, sample index or another signal.

A read-only preview of one run: every top-level run array with the same
length as ``run['time']`` can be selected in a multi-select listbox and drawn
on one axis, optionally normalised to [0, 1] so that quantities with mixed
units (MPa, um, um/s) share a scale.  Event times are marked with vertical
dotted lines.  Nothing is written back to the data; there is no result key.

Plotting happens only when the user clicks *Plot*, never on every keystroke,
and a zoom the user applied in the toolbar survives a re-plot as long as the
axis it applies to still shows the same quantity.
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
    if isinstance(value, dict) or isinstance(value, str):
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


def event_positions(run: dict, x: np.ndarray) -> np.ndarray:
    """X-coordinates of the run's events on the axis given by ``x``.

    Event times come from ``run['events'][*]['event_time']`` when present,
    otherwise from ``run['time'][run['event_indices']]``.  Each time is mapped
    to the nearest sample of ``run['time']`` and the value of ``x`` at that
    sample is returned, so markers stay correct when ``x`` is not time.
    """
    time = np.asarray(run.get("time", []), dtype=float)
    if time.size == 0 or x.shape != time.shape:
        return np.array([], dtype=float)
    times: list[float] = []
    events = run.get("events")
    if isinstance(events, list) and events:
        for event in events:
            if isinstance(event, dict) and "event_time" in event:
                try:
                    times.append(float(event["event_time"]))
                except (TypeError, ValueError):
                    continue
    if not times:
        indices = run.get("event_indices")
        if indices is not None:
            try:
                idx = np.asarray(indices, dtype=int).ravel()
            except (TypeError, ValueError):
                idx = np.array([], dtype=int)
            idx = idx[(idx >= 0) & (idx < time.size)]
            times = [float(t) for t in time[idx]]
    if not times:
        return np.array([], dtype=float)
    times_arr = np.asarray(times, dtype=float)
    finite = np.isfinite(times_arr)
    times_arr = times_arr[finite]
    if times_arr.size == 0:
        return np.array([], dtype=float)
    # nearest sample of the (non-decreasing) time axis
    pos = np.searchsorted(time, times_arr)
    pos = np.clip(pos, 1, time.size - 1)
    left = pos - 1
    choose_left = np.abs(time[left] - times_arr) <= np.abs(time[pos] - times_arr)
    nearest = np.where(choose_left, left, pos)
    return np.asarray(x, dtype=float)[nearest]


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
        self.status_var = tk.StringVar(master=self, value="Preview only - nothing is saved")
        ttk.Label(controls, textvariable=self.status_var, wraplength=180).grid(
            row=6, column=0, columnspan=2, pady=(4, 0), sticky="w")

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
        return [self.candidates[int(i)] for i in self.signal_listbox.curselection()]

    def select_signals(self, keys: Sequence[str]) -> None:
        """Select exactly ``keys`` in the listbox (unknown keys are ignored)."""
        self.signal_listbox.selection_clear(0, tk.END)
        for key in keys:
            if key in self.candidates:
                self.signal_listbox.selection_set(self.candidates.index(key))

    # ----------------------------------------------------------------- data
    def x_data(self) -> tuple[np.ndarray, str]:
        """The X array for the current combobox choice and its axis label."""
        choice = self.x_combo.get() or X_TIME
        n = len(self.run["time"]) if "time" in self.run else 0
        if choice == X_INDEX or (choice == X_TIME and X_TIME not in self.run):
            return np.arange(n, dtype=float), "sample index"
        arr = self.run.get(choice)
        if arr is None:
            return np.arange(n, dtype=float), "sample index"
        return np.asarray(arr, dtype=float), choice

    def _zoomed(self) -> tuple[bool, bool]:
        """Whether the user changed the x / y limits away from the autoscaled ones."""
        if self._auto_limits is None:
            return False, False
        auto_x, auto_y = self._auto_limits
        cur_x, cur_y = self.ax.get_xlim(), self.ax.get_ylim()
        return (not np.allclose(cur_x, auto_x, rtol=1e-9, atol=0.0),
                not np.allclose(cur_y, auto_y, rtol=1e-9, atol=0.0))

    # ----------------------------------------------------------------- plot
    def plot(self) -> None:
        """Redraw the axis from the current selection (called by the Plot button)."""
        keys = self.selected_signals()
        normalize = bool(self.normalize_var.get())
        mark_events = bool(self.mark_events_var.get())
        x, xlabel = self.x_data()
        x_choice = self.x_combo.get() or X_TIME

        keep_x, keep_y = self._zoomed()
        prev_xlim, prev_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        keep_x = keep_x and x_choice == self._plotted_x
        keep_y = keep_y and normalize == self._plotted_normalize

        self.ax.clear()
        self.signal_lines = []
        self.event_lines = []
        for key in keys:
            y = np.asarray(self.run[key], dtype=float)
            if normalize:
                y = normalize01(y)
            line, = self.ax.plot(x, y, linewidth=1.0, label=key)
            self.signal_lines.append(line)

        if mark_events:
            for xe in event_positions(self.run, x):
                self.event_lines.append(self.ax.axvline(xe, **EVENT_LINE_STYLE))

        self.ax.set_xlabel(xlabel)
        if normalize:
            self.ax.set_ylabel("normalized [0, 1]")
        elif len(keys) == 1:
            self.ax.set_ylabel(keys[0])
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
        if keep_x:
            self.ax.set_xlim(prev_xlim)
        if keep_y:
            self.ax.set_ylim(prev_ylim)
        self._plotted_x = x_choice
        self._plotted_normalize = normalize

        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            toolbar.update()  # forget the zoom history of the previous plot
        self.canvas.draw_idle()

        n_ev = len(self.event_lines)
        if keys:
            self.status_var.set(f"{len(keys)} signal(s), {n_ev} event marker(s). Preview only - nothing is saved")
        else:
            self.status_var.set("Select one or more signals and click Plot")
