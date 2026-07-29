"""Preview a single-signal event-drop calculation without persistence."""

from __future__ import annotations

import math
import re
import tkinter as tk
from collections.abc import Mapping
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.analysis.event_drop import (
    calculate_event_drop_metrics,
    compute_half_win,
)


DEFAULT_POINTS = (-1.0, -0.5, 0.5, 1.0)


class _DraggableVerticalLine:
    """Make one Matplotlib vertical line horizontally draggable."""

    def __init__(
        self,
        line,
        on_changed,
        on_released,
        constrain,
        hit_tolerance: float = 7.0,
        on_started=None,
    ):
        self.line = line
        self.axes = line.axes
        self.canvas = line.figure.canvas
        self.on_changed = on_changed
        self.on_released = on_released
        self.constrain = constrain
        self.hit_tolerance = hit_tolerance
        self.on_started = on_started
        self.dragging = False
        self._connection_ids = [
            self.canvas.mpl_connect("button_press_event", self._on_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("button_release_event", self._on_release),
        ]

    @property
    def connected(self) -> bool:
        return bool(self._connection_ids)

    def _is_near_line(self, event) -> bool:
        if event.inaxes is not self.axes or event.xdata is None or event.x is None:
            return False
        line_x = float(self.line.get_xdata()[0])
        line_pixel_x = self.axes.transData.transform((line_x, 0.0))[0]
        return abs(float(event.x) - line_pixel_x) <= self.hit_tolerance

    def _on_press(self, event) -> None:
        if getattr(event, "button", 1) != 1 or not self._is_near_line(event):
            return
        if self.on_started is not None and not self.on_started():
            return
        self.dragging = True

    def _on_motion(self, event) -> None:
        if not self.dragging or event.inaxes is not self.axes or event.xdata is None:
            return
        position = self.constrain(float(event.xdata))
        self.line.set_xdata([position, position])
        self.on_changed(position)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if not self.dragging:
            return
        self.dragging = False
        self.on_released()

    def disconnect(self) -> None:
        for connection_id in self._connection_ids:
            self.canvas.mpl_disconnect(connection_id)
        self._connection_ids.clear()
        self.dragging = False


def find_signal_candidates(event: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return top-level numeric 1-D signals aligned with ``event['time']``.

    Candidate order follows the event mapping's insertion order.  No channel
    identity is inferred from names, and the time axis itself is excluded.
    """
    if "time" not in event:
        return {}
    try:
        time = np.asarray(event["time"])
    except (TypeError, ValueError):
        return {}
    if time.ndim != 1:
        return {}

    candidates: dict[str, np.ndarray] = {}
    for key, value in event.items():
        if key == "time" or not isinstance(value, (np.ndarray, list)):
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if (
            array.ndim == 1
            and array.size == time.size
            and (
                np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.floating)
            )
        ):
            try:
                array.astype(float, copy=False)
            except (TypeError, ValueError, OverflowError):
                continue
            candidates[key] = array
    return candidates


def parse_smooth_window(value: str) -> int | None:
    """Parse a blank or positive-integer smoothing window."""
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"[1-9]\d*", text) is None:
        raise ValueError("smooth_w must be blank or a positive integer")
    return int(text)


def parse_preview_parameters(
    half_win: str,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    smooth_w: str,
) -> dict[str, Any]:
    """Parse preview controls into explicit analysis arguments."""
    try:
        half_window = float(half_win)
        points = (
            float(pre_start),
            float(pre_end),
            float(post_start),
            float(post_end),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("half_win and fitting-window values must be numeric") from exc
    if not math.isfinite(half_window) or half_window < 0:
        raise ValueError("half_win must be a finite non-negative number")
    if not all(math.isfinite(point) for point in points):
        raise ValueError("fitting-window values must be finite")

    pre_start_value, pre_end_value, post_start_value, post_end_value = points
    normalized_pre = (
        min(pre_start_value, pre_end_value),
        max(pre_start_value, pre_end_value),
    )
    normalized_post = (
        min(post_start_value, post_end_value),
        max(post_start_value, post_end_value),
    )
    if normalized_pre[1] > 0:
        raise ValueError("pre fitting window must be at or before event time")
    if normalized_post[0] < 0:
        raise ValueError("post fitting window must be at or after event time")
    if normalized_pre[0] == normalized_pre[1]:
        raise ValueError("pre fitting window must have non-zero width")
    if normalized_post[0] == normalized_post[1]:
        raise ValueError("post fitting window must have non-zero width")
    if any(point < -half_window or point > half_window for point in points):
        raise ValueError(
            "all fitting-window endpoints must be within "
            "[-half_win, +half_win]"
        )
    return {
        "half_win": half_window,
        "points": points,
        "smooth_w": parse_smooth_window(smooth_w),
    }


def format_preview_result(result: Mapping[str, Any]) -> dict[str, str]:
    """Format a helper result for display without changing its signed delta."""
    if not result.get("valid", False):
        return {
            "valid": "False",
            "delta": "—",
            "magnitude": "—",
            "val_pre_0": "—",
            "val_post_0": "—",
            "status": "Invalid fitting windows/data",
        }

    delta = float(result["delta"])
    return {
        "valid": "True",
        "delta": f"{delta:.6g}",
        "magnitude": f"{abs(delta):.6g}",
        "val_pre_0": f"{float(result['val_pre_0']):.6g}",
        "val_post_0": f"{float(result['val_post_0']):.6g}",
        "status": "Preview only — not saved",
    }


class EventDropEditorView(tk.Toplevel):
    """Preview event-drop fitting for one explicitly selected event signal."""

    def __init__(self, parent, run_idx: int, event_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Event Drop Preview - Event {event_idx}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.run_idx = run_idx
        self.event_idx = event_idx
        self.data_manager = self.parent.data_manager
        self.event: Mapping[str, Any] = {}
        self.signal_candidates: dict[str, np.ndarray] = {}
        self.preview_result: dict[str, Any] | None = None
        self.preview_parameters: dict[str, Any] | None = None
        self._endpoint_draggables: dict[str, _DraggableVerticalLine] = {}
        self._active_endpoint: str | None = None

        self._set_event(event_idx)
        self._create_controls()
        self._create_figure()
        self._initialize_event_selector()
        self._refresh_event_widgets()

    def _create_controls(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(controls, text="Event:").grid(row=0, column=0, sticky="e")
        self.event_combobox = ttk.Combobox(controls, width=8, state="readonly")
        self.event_combobox.grid(row=0, column=1, padx=(3, 10), sticky="w")
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_event_changed)

        ttk.Label(controls, text="Signal:").grid(row=0, column=2, sticky="e")
        self.signal_combobox = ttk.Combobox(controls, width=24, state="readonly")
        self.signal_combobox.grid(row=0, column=3, padx=(3, 10), sticky="w")
        self.signal_combobox.bind("<<ComboboxSelected>>", self.on_signal_changed)

        defaults = {
            "half_win": str(compute_half_win()),
            "pre_start": str(DEFAULT_POINTS[0]),
            "pre_end": str(DEFAULT_POINTS[1]),
            "post_start": str(DEFAULT_POINTS[2]),
            "post_end": str(DEFAULT_POINTS[3]),
            "smooth_w": "",
        }
        self.parameter_vars: dict[str, tk.StringVar] = {}
        labels = (
            ("Half window", "half_win"),
            ("Pre start", "pre_start"),
            ("Pre end", "pre_end"),
            ("Post start", "post_start"),
            ("Post end", "post_end"),
            ("Smooth window", "smooth_w"),
        )
        for column, (label, key) in enumerate(labels):
            ttk.Label(controls, text=label).grid(row=1, column=column, padx=3)
            variable = tk.StringVar(value=defaults[key])
            self.parameter_vars[key] = variable
            ttk.Entry(controls, textvariable=variable, width=11).grid(
                row=2, column=column, padx=3
            )

        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=2, column=6, padx=(10, 3))

        results = ttk.LabelFrame(self, text="Preview Result")
        results.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.result_vars = {
            key: tk.StringVar(value="—")
            for key in ("valid", "delta", "magnitude", "val_pre_0", "val_post_0")
        }
        result_labels = (
            ("Valid", "valid"),
            ("Signed delta", "delta"),
            ("Magnitude", "magnitude"),
            ("Pre fit at t=0", "val_pre_0"),
            ("Post fit at t=0", "val_post_0"),
        )
        for column, (label, key) in enumerate(result_labels):
            ttk.Label(results, text=f"{label}:").grid(
                row=0, column=2 * column, padx=(6, 2), pady=3
            )
            ttk.Label(results, textvariable=self.result_vars[key]).grid(
                row=0, column=2 * column + 1, padx=(0, 8), pady=3
            )
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(results, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=10, padx=6, pady=(0, 3), sticky="w"
        )

    def _create_figure(self) -> None:
        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.raw_ax = self.figure.add_subplot(211)
        self.fit_ax = self.figure.add_subplot(212, sharex=self.raw_ax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _initialize_event_selector(self) -> None:
        events = self.data_manager.get_data(f"runs/[{self.run_idx}]/events")
        values = [str(index) for index in range(len(events))]
        self.event_combobox.configure(values=values)
        self.event_combobox.set(str(self.event_idx))

    def _set_event(self, event_idx: int) -> None:
        """Load canonical event state without changing it."""
        self.event_idx = event_idx
        self.event = self.data_manager.get_data(
            f"runs/[{self.run_idx}]/events/[{self.event_idx}]"
        )
        self.signal_candidates = find_signal_candidates(self.event)
        self.preview_result = None
        self.preview_parameters = None

    def _refresh_event_widgets(self) -> None:
        names = list(self.signal_candidates)
        self.signal_combobox.configure(values=names)
        if names:
            self.signal_combobox.set("")
            self.preview_button.configure(state="disabled")
            self.status_var.set("Select a signal to enable preview")
        else:
            self.signal_combobox.set("")
            self.preview_button.configure(state="disabled")
            self.status_var.set("No numeric one-dimensional event signal is available")
        self._clear_result_display()
        self._plot_preview()

    def _clear_result_display(self) -> None:
        for variable in self.result_vars.values():
            variable.set("—")

    def _invalidate_preview(self) -> None:
        self.preview_result = None
        self.preview_parameters = None
        self._clear_result_display()
        self.status_var.set("Fitting windows changed — recompute preview")

    def on_event_changed(self, event=None) -> None:
        try:
            event_idx = int(self.event_combobox.get())
            self._set_event(event_idx)
            self.title(f"Event Drop Preview - Event {event_idx}")
            self._refresh_event_widgets()
        except Exception as exc:
            messagebox.showerror("Event Drop Preview", f"Failed to load event: {exc}")

    def on_signal_changed(self, event=None) -> None:
        self.preview_result = None
        self.preview_parameters = None
        self._clear_result_display()
        if self.signal_combobox.get() in self.signal_candidates:
            self.preview_button.configure(state="normal")
            self.status_var.set("Preview only — not saved")
        else:
            self.preview_button.configure(state="disabled")
            self.status_var.set("Select a signal to enable preview")
        self._plot_preview()

    def _read_parameters(self) -> dict[str, Any]:
        return parse_preview_parameters(
            self.parameter_vars["half_win"].get(),
            self.parameter_vars["pre_start"].get(),
            self.parameter_vars["pre_end"].get(),
            self.parameter_vars["post_start"].get(),
            self.parameter_vars["post_end"].get(),
            self.parameter_vars["smooth_w"].get(),
        )

    def calculate_preview(
        self,
        signal_name: str,
        half_win: float,
        points: tuple[float, float, float, float],
        smooth_w: int | None,
    ) -> dict[str, Any]:
        """Call the in-memory metric orchestrator and keep the selected result."""
        if signal_name not in self.signal_candidates:
            raise ValueError("Select an available signal")
        results = calculate_event_drop_metrics(
            time=self.event["time"],
            event_time=self.event["event_time"],
            signals={signal_name: self.signal_candidates[signal_name]},
            parameters={
                signal_name: {
                    "half_win": half_win,
                    "points": points,
                    "smooth_w": smooth_w,
                }
            },
        )
        result = results[signal_name]
        self.preview_result = result
        self.preview_parameters = {
            "half_win": half_win,
            "points": points,
            "smooth_w": smooth_w,
        }
        return result

    def recompute_preview(self) -> None:
        try:
            parameters = self._read_parameters()
            result = self.calculate_preview(
                self.signal_combobox.get(),
                parameters["half_win"],
                parameters["points"],
                parameters["smooth_w"],
            )
            display = format_preview_result(result)
            for key, variable in self.result_vars.items():
                variable.set(display[key])
            self.status_var.set(display["status"])
            self._plot_preview()
        except ValueError as exc:
            self.preview_result = None
            self.preview_parameters = None
            self._clear_result_display()
            self.status_var.set(str(exc))
            messagebox.showerror("Event Drop Preview", str(exc))

    def _current_points(self) -> tuple[float, float, float, float] | None:
        try:
            points = tuple(
                float(self.parameter_vars[key].get())
                for key in ("pre_start", "pre_end", "post_start", "post_end")
            )
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(point) for point in points):
            return None
        return points

    def _constrain_endpoint(self, endpoint: str, position: float) -> float:
        try:
            half_win = float(self.parameter_vars["half_win"].get())
            if not math.isfinite(half_win) or half_win < 0:
                raise ValueError
            lower_bound, upper_bound = -half_win, half_win
        except (TypeError, ValueError):
            lower_bound, upper_bound = self.raw_ax.get_xlim()

        constrained = min(max(position, lower_bound), upper_bound)
        if endpoint.startswith("pre_"):
            constrained = min(constrained, 0.0)
        else:
            constrained = max(constrained, 0.0)
        return constrained

    def _on_endpoint_changed(self, endpoint: str, position: float) -> None:
        self.parameter_vars[endpoint].set(f"{position:.6g}")
        self._invalidate_preview()

    def _begin_endpoint_drag(self, endpoint: str) -> bool:
        if self._active_endpoint is not None:
            return False
        self._active_endpoint = endpoint
        return True

    def _on_endpoint_released(self, endpoint: str) -> None:
        if self._active_endpoint == endpoint:
            self._active_endpoint = None
        self._plot_preview()

    def _disconnect_endpoint_lines(self) -> None:
        for draggable in getattr(self, "_endpoint_draggables", {}).values():
            draggable.disconnect()
        self._endpoint_draggables = {}
        self._active_endpoint = None

    def _create_endpoint_lines(
        self, points: tuple[float, float, float, float]
    ) -> None:
        endpoint_values = dict(zip(
            ("pre_start", "pre_end", "post_start", "post_end"),
            points,
        ))
        styles = {
            "pre_start": ("C0", "-"),
            "pre_end": ("C0", "--"),
            "post_start": ("C1", "-"),
            "post_end": ("C1", "--"),
        }
        for endpoint, position in endpoint_values.items():
            color, linestyle = styles[endpoint]
            line = self.raw_ax.axvline(
                position,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
            self._endpoint_draggables[endpoint] = _DraggableVerticalLine(
                line=line,
                on_changed=lambda value, key=endpoint: self._on_endpoint_changed(
                    key, value
                ),
                on_released=lambda key=endpoint: self._on_endpoint_released(key),
                constrain=lambda value, key=endpoint: self._constrain_endpoint(
                    key, value
                ),
                on_started=lambda key=endpoint: self._begin_endpoint_drag(key),
            )

    def _plot_preview(self) -> None:
        """Plot raw selected data and the helper's baseline-relative fit lines."""
        self._disconnect_endpoint_lines()
        self.raw_ax.clear()
        self.fit_ax.clear()

        signal_name = self.signal_combobox.get() if hasattr(self, "signal_combobox") else ""
        if signal_name in self.signal_candidates:
            time = np.asarray(self.event["time"], dtype=float)
            relative_time = time - float(self.event["event_time"])
            raw_signal = self.signal_candidates[signal_name]
            self.raw_ax.plot(relative_time, raw_signal, color="black", linewidth=0.9)
            self.raw_ax.set_ylabel(signal_name)
            self.raw_ax.set_title("Raw selected signal (fit below is processed by analysis helper)")

        points = self._current_points() if hasattr(self, "parameter_vars") else None
        if points is not None:
            pre_start, pre_end, post_start, post_end = points
            for axis in (self.raw_ax, self.fit_ax):
                axis.axvspan(
                    min(pre_start, pre_end),
                    max(pre_start, pre_end),
                    color="C0",
                    alpha=0.12,
                    label="Pre window" if axis is self.raw_ax else None,
                )
                axis.axvspan(
                    min(post_start, post_end),
                    max(post_start, post_end),
                    color="C1",
                    alpha=0.12,
                    label="Post window" if axis is self.raw_ax else None,
                )
            self._create_endpoint_lines(points)

        for axis in (self.raw_ax, self.fit_ax):
            axis.axvline(0.0, color="gray", linestyle=":", linewidth=1)
            axis.grid(True, alpha=0.25)

        if self.preview_result and self.preview_result.get("valid") and points is not None:
            pre_x = np.linspace(min(points[0], points[1]), 0.0, 100)
            post_x = np.linspace(0.0, max(points[2], points[3]), 100)
            coeff_pre = self.preview_result["coeff_pre"]
            coeff_post = self.preview_result["coeff_post"]
            self.fit_ax.plot(
                pre_x,
                coeff_pre[0] * pre_x + coeff_pre[1],
                "--",
                color="C0",
                label="Pre fit",
            )
            self.fit_ax.plot(
                post_x,
                coeff_post[0] * post_x + coeff_post[1],
                "--",
                color="C1",
                label="Post fit",
            )
            self.fit_ax.legend(loc="best")

        if points is not None and signal_name in self.signal_candidates:
            self.raw_ax.legend(loc="best")
        self.fit_ax.set_ylabel("Helper fit\n(baseline-relative)")
        self.fit_ax.set_xlabel("Time relative to event (s)")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_close(self) -> None:
        self._disconnect_endpoint_lines()
        self._active_endpoint = None
        if self in self.parent.child_windows:
            self.parent.child_windows.remove(self)
        self.destroy()
