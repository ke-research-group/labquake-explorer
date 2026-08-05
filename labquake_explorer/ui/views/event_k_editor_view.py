"""Read-only loading-stiffness preview for explicitly selected run signals.

The two bindings are generic Tau-like and Slip-like analysis inputs, not
physical-identity inference.  The view resolves canonical context, calls the
Potter-compatible analysis, and displays signed stiffness without persistence
or DataManager writes.
"""

from __future__ import annotations

import math
import re
import tkinter as tk
from collections.abc import Mapping, Sequence
from numbers import Real
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.analysis.k_stiffness import (
    calculate_event_loading_stiffness,
)


class _DraggablePreEndpoint:
    """Make one pre-trigger endpoint line horizontally draggable."""

    def __init__(self, line, on_changed, on_released, constrain):
        self.line = line
        self.axes = line.axes
        self.canvas = line.figure.canvas
        self.on_changed = on_changed
        self.on_released = on_released
        self.constrain = constrain
        self.dragging = False
        self._connection_ids = [
            self.canvas.mpl_connect("button_press_event", self._on_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def _on_press(self, event) -> None:
        if (
            getattr(event, "button", 1) != 1
            or event.inaxes is not self.axes
            or event.x is None
        ):
            return
        line_x = float(self.line.get_xdata()[0])
        line_pixel_x = self.axes.transData.transform((line_x, 0.0))[0]
        if abs(float(event.x) - line_pixel_x) <= 7.0:
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


def parse_k_preview_parameters(
    pre_start: str,
    pre_end: str,
    window_sec: str,
    smooth_w: str,
    highpass_freq: str,
    lowpass_freq: str,
    use_ransac: bool,
) -> dict[str, Any]:
    """Parse explicit K-preview controls without inspecting signal data."""
    values = {}
    for field, text in (
        ("pre_start", pre_start),
        ("pre_end", pre_end),
        ("window_sec", window_sec),
        ("highpass_freq", highpass_freq),
        ("lowpass_freq", lowpass_freq),
    ):
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
        values[field] = value

    if re.fullmatch(r"[1-9]\d*", str(smooth_w).strip()) is None:
        raise ValueError("smooth_w must be a positive integer")
    if values["pre_start"] >= values["pre_end"]:
        raise ValueError("pre_start must be less than pre_end")
    if values["pre_start"] >= 0:
        raise ValueError("pre_start must be less than zero")
    if values["pre_end"] > 0:
        raise ValueError("pre_end must be at or before zero")
    if values["window_sec"] <= 0:
        raise ValueError("window_sec must be greater than zero")
    if values["highpass_freq"] < 0:
        raise ValueError("highpass_freq must be non-negative")
    if values["lowpass_freq"] < 0:
        raise ValueError("lowpass_freq must be non-negative")

    return {
        **values,
        "smooth_w": int(str(smooth_w).strip()),
        "use_ransac": bool(use_ransac),
    }


def find_full_run_signal_candidates(run_data: Any) -> list[str]:
    """Return aligned finite real top-level run arrays in insertion order."""
    if not isinstance(run_data, Mapping) or "time" not in run_data:
        return []
    valid_time, time_size = _is_finite_real_array(run_data["time"])
    if not valid_time:
        return []

    candidates = []
    for key, value in run_data.items():
        if not isinstance(key, str) or key == "time":
            continue
        valid, size = _is_finite_real_array(value)
        if valid and size == time_size:
            candidates.append(key)
    return candidates


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


class EventKEditorView(tk.Toplevel):
    """Preview signed loading stiffness for one canonical event.

    Draggable pre-window endpoints synchronize controls during motion and
    perform the full redraw on release.  All state and results are preview-only.
    """

    def __init__(self, parent, run_idx: int, event_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Loading Stiffness Preview - Event {event_idx}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_manager = self.parent.data_manager
        self.run_idx = run_idx
        self.event_idx = event_idx
        self.run_data: Mapping[str, Any] | None = None
        self.events: Any = None
        self.event: Mapping[str, Any] | None = None
        self.current_event_time: float | None = None
        self.full_run_signal_candidates: list[str] = []
        self.metric_bindings: dict[str, str | None] = {
            "tau": None,
            "slip": None,
        }
        self.preview_result: dict[str, Any] | None = None
        self.preview_parameters: dict[str, Any] | None = None
        self._endpoint_draggables: list[_DraggablePreEndpoint] = []
        self._initializing_parameters = True
        self._updating_endpoint_control = False

        self._set_event(event_idx)
        self._create_controls()
        self._create_figure()
        self._initialize_event_selector()
        self._refresh_event_widgets()
        self._initializing_parameters = False

    @staticmethod
    def _coerce_event_time(value: Any) -> float | None:
        if isinstance(value, (bool, np.bool_, np.ndarray)):
            return None
        if not isinstance(value, (Real, np.integer, np.floating)):
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return converted if math.isfinite(converted) else None

    def _set_event(self, event_idx: int) -> None:
        self.event_idx = event_idx
        self.run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
        self.event = self.data_manager.get_data(
            f"runs/[{self.run_idx}]/events/[{self.event_idx}]"
        )
        self.events = (
            self.run_data.get("events") if isinstance(self.run_data, Mapping) else None
        )
        self.current_event_time = (
            self._coerce_event_time(self.event.get("event_time"))
            if isinstance(self.event, Mapping)
            else None
        )
        self.full_run_signal_candidates = find_full_run_signal_candidates(
            self.run_data
        )
        self.metric_bindings = {"tau": None, "slip": None}
        self.preview_result = None
        self.preview_parameters = None

    def _create_controls(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(controls, text="Event:").grid(row=0, column=0, sticky="e")
        self.event_combobox = ttk.Combobox(controls, width=8, state="readonly")
        self.event_combobox.grid(row=0, column=1, padx=(3, 10), sticky="w")
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_event_changed)

        self.signal_comboboxes = {}
        for column, (role, label) in enumerate(
            (("tau", "Tau signal:"), ("slip", "Slip signal:")), start=2
        ):
            ttk.Label(controls, text=label).grid(row=0, column=2 * column - 2)
            combobox = ttk.Combobox(controls, width=22, state="readonly")
            combobox.grid(row=0, column=2 * column - 1, padx=(3, 10))
            combobox.bind(
                "<<ComboboxSelected>>",
                lambda event, metric=role: self.on_signal_changed(metric),
            )
            self.signal_comboboxes[role] = combobox

        defaults = {
            "pre_start": "-3.0",
            "pre_end": "-0.5",
            "window_sec": "3.5",
            "smooth_w": "100",
            "highpass_freq": "0.0",
            "lowpass_freq": "0.0",
        }
        labels = (
            ("Pre start", "pre_start"),
            ("Pre end", "pre_end"),
            ("Window", "window_sec"),
            ("Smooth window", "smooth_w"),
            ("High-pass", "highpass_freq"),
            ("Low-pass", "lowpass_freq"),
        )
        self.parameter_vars = {}
        for column, (label, key) in enumerate(labels):
            ttk.Label(controls, text=label).grid(row=1, column=column, padx=3)
            variable = tk.StringVar(value=defaults[key])
            variable.trace_add("write", self._on_parameter_changed)
            self.parameter_vars[key] = variable
            ttk.Entry(controls, textvariable=variable, width=12).grid(
                row=2, column=column, padx=3
            )

        self.use_ransac_var = tk.BooleanVar(value=False)
        self.use_ransac_var.trace_add("write", self._on_parameter_changed)
        ttk.Checkbutton(
            controls, text="Use RANSAC", variable=self.use_ransac_var
        ).grid(row=2, column=6, padx=4)
        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=2, column=7, padx=4)

        results = ttk.LabelFrame(self, text="Preview Result")
        results.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.result_vars = {
            "valid": tk.StringVar(value="—"),
            "k": tk.StringVar(value="—"),
            "intercept": tk.StringVar(value="—"),
        }
        for column, (label, key) in enumerate(
            (("Valid", "valid"), ("k", "k"), ("Processed fit intercept", "intercept"))
        ):
            ttk.Label(results, text=f"{label}:").grid(row=0, column=2 * column)
            ttk.Label(results, textvariable=self.result_vars[key]).grid(
                row=0, column=2 * column + 1, padx=(2, 12)
            )
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(results, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=6, sticky="w"
        )

    def _create_figure(self) -> None:
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.slip_ax = self.figure.add_subplot(311)
        self.tau_ax = self.figure.add_subplot(312, sharex=self.slip_ax)
        self.fit_ax = self.figure.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self._plot_preview()

    def _initialize_event_selector(self) -> None:
        event_count = len(self.events) if isinstance(self.events, Sequence) else 0
        self.event_combobox.configure(values=[str(index) for index in range(event_count)])
        self.event_combobox.set(str(self.event_idx))

    def _refresh_event_widgets(self) -> None:
        names = self.full_run_signal_candidates
        for combobox in self.signal_comboboxes.values():
            combobox.configure(values=names)
            combobox.set("")
        self.metric_bindings = {"tau": None, "slip": None}
        self._clear_preview("Select Tau and Slip signals to enable preview")
        if self.current_event_time is None:
            self.status_var.set("Current event has no finite event_time")
        self._update_preview_button()
        self._plot_preview()

    def _update_preview_button(self) -> None:
        ready = (
            self.current_event_time is not None
            and all(self.metric_bindings.values())
        )
        self.preview_button.configure(state="normal" if ready else "disabled")

    def _clear_result_display(self) -> None:
        for variable in self.result_vars.values():
            variable.set("—")

    def _clear_preview(self, status: str) -> None:
        self.preview_result = None
        self.preview_parameters = None
        self._clear_result_display()
        self.status_var.set(status)

    def on_signal_changed(self, role: str) -> None:
        signal_name = self.signal_comboboxes[role].get()
        self.metric_bindings[role] = (
            signal_name if signal_name in self.full_run_signal_candidates else None
        )
        self._clear_preview("Signal binding changed — recompute preview")
        if not all(self.metric_bindings.values()):
            self.status_var.set("Select Tau and Slip signals to enable preview")
        if self.current_event_time is None:
            self.status_var.set("Current event has no finite event_time")
        self._update_preview_button()
        self._plot_preview()

    def _on_parameter_changed(self, *args) -> None:
        if (
            getattr(self, "_initializing_parameters", False)
            or getattr(self, "_updating_endpoint_control", False)
        ):
            return
        self._clear_preview("Parameters changed — recompute preview")
        self._plot_preview()

    def on_event_changed(self, event=None) -> None:
        try:
            event_idx = int(self.event_combobox.get())
            self._set_event(event_idx)
            self.title(f"Loading Stiffness Preview - Event {event_idx}")
            self._refresh_event_widgets()
        except Exception as exc:
            messagebox.showerror("Loading stiffness", f"Failed to load event: {exc}")

    def _read_parameters(self) -> dict[str, Any]:
        return parse_k_preview_parameters(
            self.parameter_vars["pre_start"].get(),
            self.parameter_vars["pre_end"].get(),
            self.parameter_vars["window_sec"].get(),
            self.parameter_vars["smooth_w"].get(),
            self.parameter_vars["highpass_freq"].get(),
            self.parameter_vars["lowpass_freq"].get(),
            self.use_ransac_var.get(),
        )

    def _resolve_bound_signal(self, role: str) -> Any:
        signal_name = self.metric_bindings.get(role)
        if signal_name not in self.full_run_signal_candidates:
            raise ValueError(f"No available signal is bound to {role}")
        return self.run_data[signal_name]

    def calculate_preview(self) -> dict[str, Any]:
        """Call the pure K helper once with explicit full-run bindings."""
        if self.current_event_time is None:
            raise ValueError("Current event has no finite event_time")
        parameters = self._read_parameters()
        if not isinstance(self.run_data, Mapping) or "time" not in self.run_data:
            raise ValueError("Full-run time is unavailable")
        result = calculate_event_loading_stiffness(
            time=self.run_data["time"],
            tau_signal=self._resolve_bound_signal("tau"),
            slip_signal=self._resolve_bound_signal("slip"),
            event_time=self.current_event_time,
            **parameters,
        )
        self.preview_result = result
        self.preview_parameters = parameters
        return result

    def recompute_preview(self) -> None:
        try:
            result = self.calculate_preview()
            if result.get("valid", False):
                self.result_vars["valid"].set("True")
                self.result_vars["k"].set(f"{float(result['k']):.6g}")
                self.result_vars["intercept"].set(
                    f"{float(result['intercept']):.6g}"
                )
            else:
                self.result_vars["valid"].set("False")
                self.result_vars["k"].set("—")
                self.result_vars["intercept"].set("—")
            self.status_var.set("Preview only — not saved")
            self._plot_preview()
        except ValueError as exc:
            self._clear_preview(str(exc))
            self._plot_preview()
            messagebox.showerror("Loading stiffness", str(exc))

    def _disconnect_endpoint_lines(self) -> None:
        for draggable in getattr(self, "_endpoint_draggables", []):
            draggable.disconnect()
        self._endpoint_draggables = []

    def _current_pre_window(self) -> tuple[float, float] | None:
        try:
            values = tuple(
                float(self.parameter_vars[key].get())
                for key in ("pre_start", "pre_end")
            )
        except (TypeError, ValueError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    def _constrain_endpoint(self, position: float) -> float:
        lower, upper = self.slip_ax.get_xlim()
        return min(max(position, lower), min(upper, 0.0))

    def _on_endpoint_changed(self, key: str, position: float) -> None:
        # Keep the trace from rebuilding draggable lines during mouse motion.
        self._updating_endpoint_control = True
        try:
            self.parameter_vars[key].set(f"{position:.6g}")
        finally:
            self._updating_endpoint_control = False
        self._clear_preview("Fitting window changed — recompute preview")

    def _on_endpoint_released(self) -> None:
        self._plot_preview()

    def _create_endpoint_lines(self, pre_window: tuple[float, float]) -> None:
        for key, position, style in (
            ("pre_start", pre_window[0], "-"),
            ("pre_end", pre_window[1], "--"),
        ):
            line = self.slip_ax.axvline(position, color="C0", linestyle=style)
            self._endpoint_draggables.append(
                _DraggablePreEndpoint(
                    line,
                    lambda value, name=key: self._on_endpoint_changed(name, value),
                    self._on_endpoint_released,
                    self._constrain_endpoint,
                )
            )

    def _plot_preview(self) -> None:
        self._disconnect_endpoint_lines()
        for axis in (self.slip_ax, self.tau_ax, self.fit_ax):
            axis.clear()
            axis.grid(True, alpha=0.25)
        self.slip_ax.set_ylabel("Slip")
        self.tau_ax.set_ylabel("Tau")
        self.fit_ax.set_xlabel("Processed slip")
        self.fit_ax.set_ylabel("Processed tau")

        result = self.preview_result
        pre_window = self._current_pre_window()
        if result and result.get("valid", False):
            relative_time = result["relative_time"]
            self.slip_ax.plot(relative_time, result["raw_slip"], label="Raw")
            self.slip_ax.plot(
                relative_time, result["processed_slip"], label="Processed"
            )
            self.tau_ax.plot(relative_time, result["raw_tau"], label="Raw")
            self.tau_ax.plot(
                relative_time, result["processed_tau"], label="Processed"
            )
            fit_mask = result["fit_mask"]
            fit_slip = result["processed_slip"][fit_mask]
            fit_tau = result["processed_tau"][fit_mask]
            self.fit_ax.scatter(fit_slip, fit_tau, s=12, label="Fit samples")
            coefficients = result["coefficients"]
            fit_x = np.linspace(float(np.min(fit_slip)), float(np.max(fit_slip)), 100)
            self.fit_ax.plot(
                fit_x,
                coefficients[0] * fit_x + coefficients[1],
                "--",
                label="Fit",
            )
            for axis in (self.slip_ax, self.tau_ax):
                axis.axvline(0.0, color="gray", linestyle=":")
                if pre_window is not None:
                    axis.axvspan(*pre_window, color="C0", alpha=0.12)
                axis.legend(loc="best")
            self.fit_ax.legend(loc="best")
        else:
            self.slip_ax.text(0.5, 0.5, "Preview to display processed signals", ha="center", transform=self.slip_ax.transAxes)
            self.tau_ax.text(0.5, 0.5, "Preview to display processed signals", ha="center", transform=self.tau_ax.transAxes)
        if pre_window is not None:
            self._create_endpoint_lines(pre_window)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_close(self) -> None:
        self._disconnect_endpoint_lines()
        if self in self.parent.child_windows:
            self.parent.child_windows.remove(self)
        self.destroy()
