"""Read-only raw colored-line preview for explicitly selected run signals.

Rows preserve exact run keys and display settings.  Signals are plotted without
offset, filtering, normalization, smoothing, or analysis; no selection is
inferred and no DataManager write or persistence operation is available.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Mapping
from tkinter import messagebox, ttk
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def find_colored_line_candidates(run_data: Any) -> list[str]:
    """Return aligned finite real top-level signals in insertion order."""
    if not isinstance(run_data, Mapping) or "time" not in run_data:
        return []
    valid_time, time_size = _is_finite_real_array(run_data["time"])
    if not valid_time:
        return []

    candidates = []
    for key, value in run_data.items():
        if not isinstance(key, str) or key == "time":
            continue
        valid_signal, signal_size = _is_finite_real_array(value)
        if valid_signal and signal_size == time_size:
            candidates.append(key)
    return candidates


def parse_line_width(value: Any) -> float:
    """Parse a finite positive Matplotlib line width."""
    try:
        width = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Line width must be a finite positive number") from exc
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Line width must be a finite positive number")
    return width


def _is_finite_real_array(value: Any) -> tuple[bool, int]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False, 0
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        return False, 0
    try:
        converted = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False, 0
    return bool(np.all(np.isfinite(converted))), int(array.size)


class ColoredSlipLinesView(tk.Toplevel):
    """Plot explicitly bound full-run signals and in-memory display settings."""

    def __init__(self, parent, run_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Colored Slip Lines Preview - Run {run_idx}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_manager = self.parent.data_manager
        self.run_idx = run_idx
        self.run_data: Mapping[str, Any] | None = None
        self.signal_candidates: list[str] = []
        self.signal_rows: list[dict[str, Any]] = []
        self._color_index = 0

        self._refresh_run_context(run_idx)
        self._create_controls()
        self._create_figure()
        self._plot_lines()

    def _refresh_run_context(self, run_idx: int | None = None) -> None:
        """Refresh one run reference and clear only bindings that became stale."""
        if run_idx is not None:
            self.run_idx = run_idx
        run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
        self.run_data = run_data if isinstance(run_data, Mapping) else None
        self.signal_candidates = find_colored_line_candidates(self.run_data)

        for row in getattr(self, "signal_rows", []):
            selector = row["signal_selector"]
            selector.configure(values=self.signal_candidates)
            signal_name = row["signal_name"]
            if signal_name not in self.signal_candidates:
                row["signal_name"] = None
                selector.set("")

    def _create_controls(self) -> None:
        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(header, text="Add signal", command=self.add_signal_row).pack(
            side=tk.LEFT
        )
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(header, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        self.rows_frame = ttk.LabelFrame(self, text="Signals")
        self.rows_frame.pack(side=tk.TOP, fill=tk.X, padx=6)

    def _create_figure(self) -> None:
        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _next_default_color(self) -> str:
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        color = colors[self._color_index % len(colors)]
        self._color_index += 1
        return color

    def add_signal_row(self) -> None:
        """Add one unbound row with an independent visual configuration."""
        frame = ttk.Frame(self.rows_frame)
        frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        row: dict[str, Any] = {
            "frame": frame,
            "signal_name": None,
            "color_var": tk.StringVar(value=self._next_default_color()),
            "visible_var": tk.BooleanVar(value=True),
            "linewidth_var": tk.StringVar(value="1.5"),
            "label_var": tk.StringVar(value=""),
        }

        ttk.Label(frame, text="Signal:").pack(side=tk.LEFT)
        selector = ttk.Combobox(
            frame, width=22, state="readonly", values=self.signal_candidates
        )
        selector.pack(side=tk.LEFT, padx=(2, 6))
        selector.set("")
        selector.bind(
            "<<ComboboxSelected>>",
            lambda event, current=row: self.on_signal_changed(current),
        )
        row["signal_selector"] = selector

        for label, key, width in (
            ("Color:", "color_var", 9),
            ("Line width:", "linewidth_var", 7),
            ("Label:", "label_var", 16),
        ):
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=row[key], width=width).pack(
                side=tk.LEFT, padx=(2, 6)
            )
        ttk.Checkbutton(
            frame,
            text="Visible",
            variable=row["visible_var"],
            command=self.recompute_preview,
        ).pack(side=tk.LEFT)
        ttk.Button(
            frame,
            text="Remove",
            command=lambda current=row: self.remove_signal_row(current),
        ).pack(side=tk.RIGHT)

        for variable in (row["color_var"], row["linewidth_var"], row["label_var"]):
            variable.trace_add("write", self._on_style_changed)
        self.signal_rows.append(row)

    def remove_signal_row(self, row: dict[str, Any]) -> None:
        if row not in self.signal_rows:
            return
        self.signal_rows.remove(row)
        row["frame"].destroy()
        self._plot_lines()

    def on_signal_changed(self, row: dict[str, Any]) -> None:
        signal_name = row["signal_selector"].get()
        row["signal_name"] = (
            signal_name if signal_name in self.signal_candidates else None
        )
        self._plot_lines()

    def _on_style_changed(self, *args) -> None:
        self.recompute_preview()

    def _resolve_row_signal(self, row: Mapping[str, Any]) -> Any | None:
        signal_name = row.get("signal_name")
        if (
            signal_name not in self.signal_candidates
            or not isinstance(self.run_data, Mapping)
        ):
            return None
        return self.run_data[signal_name]

    def recompute_preview(self) -> None:
        try:
            self._plot_lines()
            self.status_var.set("Preview only — not saved")
        except ValueError as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Colored slip lines", str(exc))

    def _plot_lines(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Signal")
        self.ax.grid(True, alpha=0.25)

        if not isinstance(self.run_data, Mapping) or "time" not in self.run_data:
            self.canvas.draw_idle()
            return
        time = self.run_data["time"]
        plotted = False
        for row in self.signal_rows:
            signal = self._resolve_row_signal(row)
            if signal is None or not bool(row["visible_var"].get()):
                continue
            label = row["label_var"].get().strip() or row["signal_name"]
            self.ax.plot(
                time,
                signal,
                color=row["color_var"].get().strip(),
                linewidth=parse_line_width(row["linewidth_var"].get()),
                label=label,
            )
            plotted = True
        if plotted:
            self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_close(self) -> None:
        if self in self.parent.child_windows:
            self.parent.child_windows.remove(self)
        self.destroy()
