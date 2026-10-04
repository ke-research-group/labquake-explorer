"""Choose the window around the picked events and extract them.

Opened from ``event_indices``.  Plots Y against X for the whole
run with every picked event marked, shades the ``[start, end]`` window of
each event and highlights the selected one, and runs the extraction with the
window typed in.  The window is saved under ``runs/[r]['event_window']`` so
that a later extraction starts from the same settings.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from labquake_explorer.data.channels import aligned_fields, get_field
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT_INDICES
from labquake_explorer.ui.views.base import RunView, event_list

RESULT_VERSION = 1
DEFAULT_HALF_WINDOW_S = 5.0      # start/end default to -/+ config.DEFAULT_WINDOW_SIZE, else this
X_INDEX = "index"
ZOOM_FACTOR = 3.0            # "Zoom to event" shows this many windows around the event


@register_view("Extract Events", kinds=[EVENT_INDICES], order=10)
class ExtractEventsView(RunView):
    window_title = "Extract Events"
    result_key = "event_window"

    def __init__(self, app, run_idx):
        self.candidates: list[str] = []
        self.indices: list[int] = []
        self.time: np.ndarray = np.array([])
        self.result: Optional[dict] = None
        super().__init__(app, run_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        box = ttk.LabelFrame(self, text="Event window")
        box.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(box, text="X Data:").grid(row=0, column=0, padx=4, pady=3, sticky="e")
        self.x_combo = ttk.Combobox(box, state="readonly", width=22)
        self.x_combo.grid(row=0, column=1, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="Y Data:").grid(row=0, column=2, padx=4, pady=3, sticky="e")
        self.y_combo = ttk.Combobox(box, state="readonly", width=22)
        self.y_combo.grid(row=0, column=3, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="Event:").grid(row=0, column=4, padx=4, pady=3, sticky="e")
        self.event_combo = ttk.Combobox(box, state="readonly", width=6)
        self.event_combo.grid(row=0, column=5, padx=4, pady=3, sticky="w")

        half = float(getattr(getattr(self.app, "config", None), "DEFAULT_WINDOW_SIZE", DEFAULT_HALF_WINDOW_S))
        ttk.Label(box, text="Start (s):").grid(row=1, column=0, padx=4, pady=3, sticky="e")
        self.start_var = tk.StringVar(master=self, value=f"{-half:g}")
        self.start_entry = ttk.Entry(box, textvariable=self.start_var, width=8)
        self.start_entry.grid(row=1, column=1, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="End (s):").grid(row=1, column=2, padx=4, pady=3, sticky="e")
        self.end_var = tk.StringVar(master=self, value=f"{half:+g}")
        self.end_entry = ttk.Entry(box, textvariable=self.end_var, width=8)
        self.end_entry.grid(row=1, column=3, padx=4, pady=3, sticky="w")
        ttk.Button(box, text="Zoom to event", command=self.zoom_to_event).grid(row=1, column=4, padx=4, pady=3)
        ttk.Button(box, text="Whole run", command=self.show_whole_run).grid(row=1, column=5, padx=4, pady=3)
        self.extract_button = ttk.Button(box, text="Extract Events", command=self.extract)
        self.extract_button.grid(row=0, column=6, rowspan=2, padx=10, pady=3, sticky="ns")
        ttk.Label(box, text="Window relative to each picked event (seconds before are negative). "
                            "Shaded: the window of every event; dark: the selected event.").grid(
            row=2, column=0, columnspan=7, padx=4, pady=(0, 3), sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(box, textvariable=self.status_var).grid(row=3, column=0, columnspan=7, padx=4, pady=(0, 3), sticky="w")

        self.x_combo.bind("<<ComboboxSelected>>", lambda e: self.plot())
        self.y_combo.bind("<<ComboboxSelected>>", lambda e: self.plot())
        self.event_combo.bind("<<ComboboxSelected>>", lambda e: self.on_event_selected())
        for entry in (self.start_entry, self.end_entry):
            entry.bind("<Return>", lambda e: self.plot())
            entry.bind("<FocusOut>", lambda e: self.plot())

        self.make_figure(figsize=(10, 5), row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.ax = self.figure.add_subplot(111)

    def on_run_loaded(self):
        self.time = np.asarray(self.run.get("time", []), dtype=float).ravel()
        n = self.time.size
        self.candidates = [p for p in aligned_fields(self.run, n, recurse=False)] if n else []
        self.x_combo.config(values=[X_INDEX] + self.candidates)
        self.y_combo.config(values=[c for c in self.candidates if c != "time"])
        saved = self.load_results() or {}
        x = saved.get("x_field") if saved.get("x_field") in self.candidates else ("time" if "time" in self.candidates else X_INDEX)
        y_candidates = [c for c in self.candidates if c != "time"]
        y = saved.get("y_field") if saved.get("y_field") in y_candidates else (
            "shear_stress" if "shear_stress" in y_candidates else (y_candidates[0] if y_candidates else ""))
        self.x_combo.set(x)
        self.y_combo.set(y)
        if "start_s" in saved and "end_s" in saved:
            self.start_var.set(f"{float(saved['start_s']):g}")
            self.end_var.set(f"{float(saved['end_s']):+g}")
        raw = self.run.get("event_indices")
        self.indices = [int(i) for i in (raw if raw is not None else []) if 0 <= int(i) < n] if n else []
        self.event_combo.config(values=[str(i) for i in range(len(self.indices))])
        if self.indices:
            self.event_combo.current(0)
            self.extract_button.state(["!disabled"])
            self.status_var.set(f"{len(self.indices)} picked events; "
                                f"{len(event_list(self.run))} extracted so far")
        else:
            self.event_combo.set("")
            self.extract_button.state(["disabled"])
            self.status_var.set("no event_indices in this run: pick events first")
        self.plot()

    # ----------------------------------------------------------- settings
    def window_s(self) -> tuple[float, float]:
        """``(start, end)`` in seconds relative to the event; ValueError on bad input."""
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            raise ValueError("start and end must be numbers (seconds)") from None
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError("start and end must be finite")
        if start >= end:
            raise ValueError("start must be before end")
        return start, end

    def x_values(self) -> tuple[np.ndarray, str]:
        choice = self.x_combo.get()
        if choice and choice != X_INDEX:
            arr = get_field(self.run, choice)
            if arr is not None and np.asarray(arr).shape == self.time.shape:
                return np.asarray(arr, dtype=float), choice
        return np.arange(self.time.size, dtype=float), "sample index"

    def selected_event(self) -> Optional[int]:
        text = self.event_combo.get().strip()
        return int(text) if text.isdigit() and int(text) < len(self.indices) else None

    def index_range(self, idx: int, start: float, end: float) -> tuple[int, int]:
        """Sample range ``[beg, end]`` of the window around pick ``idx`` (the
        processor's rule: nearest samples to event_time + start / + end)."""
        t0 = self.time[idx]
        beg = int(np.argmin(np.abs(t0 + start - self.time)))
        stop = int(np.argmin(np.abs(t0 + end - self.time)))
        return beg, stop

    # ------------------------------------------------------------- plotting
    def plot(self):
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.3)
        y_field = self.y_combo.get()
        y = get_field(self.run, y_field) if y_field else None
        x, x_label = self.x_values()
        if y is None or np.asarray(y).shape != self.time.shape:
            self.ax.set_title(f"{self.figure_title()}: select a Y array")
            self.canvas.draw_idle()
            return
        y = np.asarray(y, dtype=float)
        self.ax.plot(x, y, lw=0.7, color="C0", zorder=1)
        try:
            start, end = self.window_s()
        except ValueError as e:
            start = end = None
            self.status_var.set(str(e))
        selected = self.selected_event()
        for k, idx in enumerate(self.indices):
            self.ax.axvline(x[idx], color="k", ls=":", lw=0.8, zorder=2)
            if start is not None:
                beg, stop = self.index_range(idx, start, end)
                is_sel = k == selected
                self.ax.axvspan(x[beg], x[stop], color="#33CC66" if is_sel else "0.5",
                                alpha=0.35 if is_sel else 0.12, zorder=0)
        if self.indices:
            self.ax.plot(x[self.indices], y[self.indices], "o", color="#CC3366", ms=4, zorder=3,
                         label="picked events")
            self.ax.legend(loc="upper left", fontsize="small")
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_field)
        self.ax.set_title(self.figure_title())
        self.canvas.draw_idle()

    def figure_title(self) -> str:
        return f"{self.experiment_name()} run{self.run_idx:02d}"

    def on_event_selected(self):
        self.plot()
        self.zoom_to_event()

    def zoom_to_event(self):
        k = self.selected_event()
        if k is None:
            return
        try:
            start, end = self.window_s()
        except ValueError:
            return
        x, _ = self.x_values()
        beg, stop = self.index_range(self.indices[k], ZOOM_FACTOR * start, ZOOM_FACTOR * end)
        lo, hi = sorted((float(x[beg]), float(x[stop])))
        if hi > lo:
            self.ax.set_xlim(lo, hi)
            y_field = self.y_combo.get()
            y = get_field(self.run, y_field) if y_field else None
            if y is not None and np.asarray(y).shape == self.time.shape:
                seg = np.asarray(y, dtype=float)[min(beg, stop):max(beg, stop) + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size and seg.max() > seg.min():
                    pad = 0.05 * (seg.max() - seg.min())
                    self.ax.set_ylim(seg.min() - pad, seg.max() + pad)
            self.canvas.draw_idle()

    def show_whole_run(self):
        self.ax.autoscale(enable=True, axis="both")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- extract
    def extract(self, confirm: bool = True) -> int:
        """Extract every picked event with the current window; returns the count."""
        try:
            start, end = self.window_s()
        except ValueError as e:
            self.status_var.set(str(e))
            return 0
        if not self.indices:
            self.status_var.set("no event_indices in this run: pick events first")
            return 0
        if confirm and event_list(self.run):
            if not messagebox.askokcancel(title="Confirmation",
                                          message=f'This will replace all data in "{self.run_path}/events".',
                                          icon=messagebox.WARNING, parent=self):
                return 0
        events = self.data_manager.event_processor.extract_events(
            self.run, self.indices, window=max(-start, end), pre=-start, post=end)
        self.data_manager.set_data(f"{self.run_path}/events", events, add_key=True)
        self.result = {
            "version": RESULT_VERSION,
            "start_s": float(start),
            "end_s": float(end),
            "x_field": self.x_combo.get(),
            "y_field": self.y_combo.get(),
            "n_events": len(events),
        }
        self.save_results(dict(self.result))      # also refreshes the tree
        self.status_var.set(f"{len(events)} events extracted, window {start:+g} s .. {end:+g} s")
        return len(events)
