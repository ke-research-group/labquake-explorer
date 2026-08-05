"""Preview one explicitly selected event-local signal and inter-event metrics."""

from __future__ import annotations

import math
import re
import tkinter as tk
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.analysis.event_drop import (
    calculate_event_drop_metrics,
    calculate_interevent_displacement_metrics,
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


def parse_interevent_parameters(
    push_speed: str,
    delay_sec: str,
    dmax_smooth_w: str,
) -> dict[str, Any]:
    """Parse the explicit controls used by inter-event displacement preview."""
    try:
        speed = float(push_speed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("push_speed must be a finite number") from exc
    if not math.isfinite(speed):
        raise ValueError("push_speed must be a finite number")

    try:
        delay = float(delay_sec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delay_sec must be a finite number") from exc
    if not math.isfinite(delay):
        raise ValueError("delay_sec must be a finite number")

    if re.fullmatch(r"[1-9]\d*", dmax_smooth_w.strip()) is None:
        raise ValueError("dmax_smooth_w must be a positive integer")
    return {
        "push_speed": speed,
        "delay_sec": delay,
        "dmax_smooth_w": int(dmax_smooth_w),
    }


def format_interevent_result(result: Mapping[str, Any]) -> dict[str, str]:
    """Format one inter-event result without changing its numeric value."""
    if not result.get("valid", False):
        return {"valid": "False", "value": "—"}
    return {
        "valid": "True",
        "value": f"{float(result['value']):.6g}",
    }


class EventDropEditorView(tk.Toplevel):
    """Preview one explicitly selected event-local signal."""

    def __init__(self, parent, run_idx: int, event_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"Event Drop Preview - Event {event_idx}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.run_idx = run_idx
        self.event_idx = event_idx
        self.data_manager = self.parent.data_manager
        self.event: Mapping[str, Any] = {}
        self.events: Any = None
        self.signal_candidates: dict[str, np.ndarray] = {}
        self.run_data: Mapping[str, Any] | None = None
        self.current_event_time: float | None = None
        self.previous_event_time: float | None = None
        self.full_run_signal_candidates: list[str] = []
        self.interevent_bindings: dict[str, str | None] = {
            "dmax": None,
            "reference": None,
        }
        self.interevent_preview_results: dict[str, dict[str, Any]] = {}
        self.interevent_preview_parameters: dict[str, Any] | None = None
        self.selected_signal_name: str | None = None
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
        self.signal_combobox = ttk.Combobox(controls, width=20, state="readonly")
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
        ttk.Label(controls, text="Event drop fitting").grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
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
            ttk.Label(controls, text=label).grid(row=2, column=column, padx=3)
            variable = tk.StringVar(value=defaults[key])
            self.parameter_vars[key] = variable
            ttk.Entry(controls, textvariable=variable, width=11).grid(
                row=3, column=column, padx=3
            )

        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=3, column=6, padx=(10, 3))

        results = ttk.LabelFrame(self, text="Preview Result")
        results.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.result_vars = {
            key: tk.StringVar(value="—")
            for key in (
                "signal_name",
                "valid",
                "delta",
                "magnitude",
                "val_pre_0",
                "val_post_0",
            )
        }
        result_labels = (
            ("Signal name", "signal_name"),
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
            row=1, column=0, columnspan=12, padx=6, pady=(0, 3), sticky="w"
        )

        for variable in self.parameter_vars.values():
            variable.trace_add("write", lambda *args: self._on_parameter_changed())

        self._create_interevent_controls()

    def _create_interevent_controls(self) -> None:
        section = ttk.LabelFrame(self, text="Inter-event displacement")
        section.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))

        defaults = {
            "push_speed": "3.508",
            "delay_sec": "0.05",
            "dmax_smooth_w": "100",
        }
        labels = (
            ("Push speed", "push_speed"),
            ("Delay", "delay_sec"),
            ("D_max smooth window", "dmax_smooth_w"),
        )
        self.interevent_parameter_vars = {}
        for column, (label, key) in enumerate(labels):
            ttk.Label(section, text=label).grid(
                row=0, column=2 * column, padx=(6, 2), pady=3
            )
            variable = tk.StringVar(value=defaults[key])
            self.interevent_parameter_vars[key] = variable
            ttk.Entry(section, textvariable=variable, width=11).grid(
                row=0, column=2 * column + 1, padx=(0, 6), pady=3
            )

        ttk.Label(section, text="D_max signal").grid(
            row=0, column=6, padx=(6, 2), pady=3
        )
        self.dmax_signal_combobox = ttk.Combobox(
            section, width=20, state="readonly"
        )
        self.dmax_signal_combobox.grid(row=0, column=7, padx=(0, 6), pady=3)
        self.dmax_signal_combobox.bind(
            "<<ComboboxSelected>>", self.on_dmax_signal_changed
        )

        ttk.Label(section, text="Reference displacement signal").grid(
            row=0, column=8, padx=(6, 2), pady=3
        )
        self.reference_signal_combobox = ttk.Combobox(
            section, width=20, state="readonly"
        )
        self.reference_signal_combobox.grid(
            row=0, column=9, padx=(0, 6), pady=3
        )
        self.reference_signal_combobox.bind(
            "<<ComboboxSelected>>", self.on_reference_signal_changed
        )
        self.interevent_preview_button = ttk.Button(
            section,
            text="Preview D metrics",
            command=self.recompute_interevent_preview,
        )
        self.interevent_preview_button.grid(
            row=0, column=10, padx=(6, 3), pady=3
        )

        self.interevent_result_vars = {
            metric: {
                "valid": tk.StringVar(value="—"),
                "value": tk.StringVar(value="—"),
            }
            for metric in ("D_Push", "D_max", "D_reference")
        }
        for row, metric in enumerate(
            ("D_Push", "D_max", "D_reference"), start=1
        ):
            ttk.Label(section, text=metric).grid(
                row=row, column=0, padx=(6, 10), pady=2, sticky="w"
            )
            ttk.Label(section, text="Valid:").grid(
                row=row, column=1, padx=(3, 2), pady=2
            )
            ttk.Label(
                section,
                textvariable=self.interevent_result_vars[metric]["valid"],
            ).grid(row=row, column=2, padx=(0, 8), pady=2)
            ttk.Label(section, text="Value:").grid(
                row=row, column=3, padx=(3, 2), pady=2
            )
            ttk.Label(
                section,
                textvariable=self.interevent_result_vars[metric]["value"],
            ).grid(row=row, column=4, padx=(0, 8), pady=2)

        self.interevent_status_var = tk.StringVar(
            value="Preview only — not saved"
        )
        ttk.Label(section, textvariable=self.interevent_status_var).grid(
            row=4, column=0, columnspan=11, padx=6, pady=(2, 3), sticky="w"
        )

        for name, variable in self.interevent_parameter_vars.items():
            variable.trace_add(
                "write",
                lambda *args, parameter=name: (
                    self._on_interevent_parameter_changed(parameter)
                ),
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
        self.selected_signal_name = None
        self.preview_result = None
        self.preview_parameters = None
        self._refresh_interevent_context()
        self.interevent_preview_results = {}
        self.interevent_preview_parameters = None

    @staticmethod
    def _coerce_finite_event_time(value: Any) -> float | None:
        """Return a finite real scalar as a Python float, otherwise ``None``."""
        if isinstance(value, (bool, np.bool_, np.ndarray)):
            return None
        if not isinstance(value, (Real, np.integer, np.floating)):
            return None
        try:
            event_time = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return event_time if math.isfinite(event_time) else None

    def _get_current_event_time(self) -> float | None:
        """Resolve only the current event's top-level ``event_time``."""
        if not isinstance(self.event, Mapping):
            return None
        return self._coerce_finite_event_time(self.event.get("event_time"))

    def _find_previous_event_time(self) -> float | None:
        """Find the first finite event time before the current list position."""
        events = self.events
        is_sequence = isinstance(events, Sequence) and not isinstance(
            events, (str, bytes)
        )
        is_one_dimensional_array = (
            isinstance(events, np.ndarray) and events.ndim == 1
        )
        if not (is_sequence or is_one_dimensional_array):
            return None
        event_idx = self.event_idx
        if isinstance(event_idx, (bool, np.bool_)) or not isinstance(
            event_idx, (Integral, np.integer)
        ):
            return None
        event_idx = int(event_idx)
        if event_idx <= 0 or event_idx >= len(events):
            return None
        for previous_idx in range(event_idx - 1, -1, -1):
            previous_event = events[previous_idx]
            if not isinstance(previous_event, Mapping):
                continue
            previous_time = self._coerce_finite_event_time(
                previous_event.get("event_time")
            )
            if previous_time is not None:
                return previous_time
        return None

    def _get_full_run_time(self) -> Any | None:
        """Return the full-run top-level time object without copying it."""
        if not isinstance(self.run_data, Mapping):
            return None
        return self.run_data.get("time")

    def _resolve_full_run_signal(self, signal_name: str | None) -> Any | None:
        """Resolve an explicit top-level run key without inference or copying."""
        if (
            not isinstance(signal_name, str)
            or not signal_name
            or not isinstance(self.run_data, Mapping)
            or signal_name not in self.run_data
        ):
            return None
        return self.run_data[signal_name]

    @staticmethod
    def _is_finite_real_run_array(value: Any) -> tuple[bool, int]:
        """Check the minimum one-dimensional candidate array contract."""
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError):
            return False, 0
        if (
            raw.ndim != 1
            or raw.size == 0
            or raw.dtype.kind not in "iuf"
        ):
            return False, 0
        try:
            numeric = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return False, 0
        if not np.all(np.isfinite(numeric)):
            return False, 0
        return True, raw.size

    def _find_full_run_signal_candidates(self) -> list[str]:
        """List aligned finite real run arrays in mapping insertion order."""
        if not isinstance(self.run_data, Mapping):
            return []
        full_time = self._get_full_run_time()
        valid_time, time_size = self._is_finite_real_run_array(full_time)
        if not valid_time:
            return []

        candidates = []
        for key, value in self.run_data.items():
            if not isinstance(key, str) or key == "time":
                continue
            valid_signal, signal_size = self._is_finite_real_run_array(value)
            if valid_signal and signal_size == time_size:
                candidates.append(key)
        return candidates

    def _refresh_interevent_context(self) -> None:
        """Refresh read-only run and event context for future D previews."""
        try:
            run_data = self.data_manager.get_data(f"runs/[{self.run_idx}]")
        except (KeyError, IndexError, TypeError, ValueError):
            run_data = None
        self.run_data = run_data if isinstance(run_data, Mapping) else None
        self.events = (
            self.run_data.get("events")
            if isinstance(self.run_data, Mapping)
            else None
        )
        self.current_event_time = self._get_current_event_time()
        self.previous_event_time = self._find_previous_event_time()
        self.full_run_signal_candidates = (
            self._find_full_run_signal_candidates()
        )

    def _clear_interevent_result_display(
        self, metric: str | None = None
    ) -> None:
        result_vars = getattr(self, "interevent_result_vars", {})
        metrics = (
            (metric,)
            if metric is not None
            else ("D_Push", "D_max", "D_reference")
        )
        for metric_name in metrics:
            for variable in result_vars.get(metric_name, {}).values():
                variable.set("—")

    def _invalidate_interevent_preview(
        self, metrics: tuple[str, ...], status: str
    ) -> None:
        results = getattr(self, "interevent_preview_results", {})
        invalidated = False
        for metric in metrics:
            if metric in results:
                invalidated = True
                results.pop(metric, None)
            self._clear_interevent_result_display(metric)
        if invalidated:
            self.interevent_preview_parameters = None
        status_var = getattr(self, "interevent_status_var", None)
        if status_var is not None:
            status_var.set(status)

    def _on_interevent_parameter_changed(self, parameter_name: str) -> None:
        metrics_by_parameter = {
            "push_speed": ("D_Push",),
            "delay_sec": ("D_max", "D_reference"),
            "dmax_smooth_w": ("D_max",),
        }
        metrics = metrics_by_parameter.get(parameter_name)
        if metrics is not None:
            self._invalidate_interevent_preview(
                metrics,
                "Inter-event parameters changed — recompute preview",
            )

    def on_reference_signal_changed(self, event=None) -> None:
        signal_name = self.reference_signal_combobox.get()
        if signal_name in self.full_run_signal_candidates:
            self.interevent_bindings["reference"] = signal_name
        else:
            self.interevent_bindings["reference"] = None
        self._invalidate_interevent_preview(
            ("D_reference",),
            "Reference binding changed — recompute preview",
        )

    def on_dmax_signal_changed(self, event=None) -> None:
        signal_name = self.dmax_signal_combobox.get()
        if signal_name in self.full_run_signal_candidates:
            self.interevent_bindings["dmax"] = signal_name
        else:
            self.interevent_bindings["dmax"] = None
        self._invalidate_interevent_preview(
            ("D_max",),
            "D_max binding changed — recompute preview",
        )

    def _refresh_interevent_widgets(self) -> None:
        dmax_combobox = getattr(self, "dmax_signal_combobox", None)
        combobox = getattr(self, "reference_signal_combobox", None)
        button = getattr(self, "interevent_preview_button", None)
        status_var = getattr(self, "interevent_status_var", None)
        if (
            dmax_combobox is None
            or combobox is None
            or button is None
            or status_var is None
        ):
            return

        candidates = list(self.full_run_signal_candidates)
        dmax = self.interevent_bindings.get("dmax")
        if dmax not in candidates:
            dmax = None
            self.interevent_bindings["dmax"] = None
        reference = self.interevent_bindings.get("reference")
        if reference not in candidates:
            reference = None
            self.interevent_bindings["reference"] = None
        dmax_combobox.configure(values=candidates)
        dmax_combobox.set(dmax or "")
        combobox.configure(values=candidates)
        combobox.set(reference or "")
        self._clear_interevent_result_display()
        if self.current_event_time is None:
            button.configure(state="disabled")
            status_var.set("Current event has no finite event_time")
        else:
            button.configure(state="normal")
            status_var.set("Preview only — not saved")

    def _read_interevent_parameters(self) -> dict[str, Any]:
        variables = self.interevent_parameter_vars
        return parse_interevent_parameters(
            variables["push_speed"].get(),
            variables["delay_sec"].get(),
            variables["dmax_smooth_w"].get(),
        )

    def calculate_interevent_preview(self) -> dict[str, dict[str, Any]]:
        if self.current_event_time is None:
            raise ValueError("Current event has no finite event_time")
        parameters = self._read_interevent_parameters()
        dmax_signal = self._resolve_full_run_signal(
            self.interevent_bindings.get("dmax")
        )
        reference_signal = self._resolve_full_run_signal(
            self.interevent_bindings.get("reference")
        )
        has_run_signal = dmax_signal is not None or reference_signal is not None
        full_time = self._get_full_run_time() if has_run_signal else None
        if has_run_signal and full_time is None:
            raise ValueError(
                "Full-run time is unavailable for the selected displacement signal"
            )

        results = calculate_interevent_displacement_metrics(
            current_event_time=self.current_event_time,
            previous_event_time=self.previous_event_time,
            push_speed=parameters["push_speed"],
            time=full_time,
            lvdt_signal=dmax_signal,
            reference_displacement_signal=reference_signal,
            delay_sec=parameters["delay_sec"],
            lvdt_smooth_w=parameters["dmax_smooth_w"],
        )
        self.interevent_preview_results = results
        self.interevent_preview_parameters = parameters
        return results

    def recompute_interevent_preview(self) -> None:
        try:
            results = self.calculate_interevent_preview()
            self._clear_interevent_result_display()
            for metric in ("D_Push", "D_max", "D_reference"):
                display = format_interevent_result(results[metric])
                for key, variable in self.interevent_result_vars[metric].items():
                    variable.set(display[key])
            if self.previous_event_time is None:
                self.interevent_status_var.set(
                    "No previous event with a finite event_time"
                )
            else:
                self.interevent_status_var.set("Preview only — not saved")
        except ValueError as exc:
            self.interevent_preview_results = {}
            self.interevent_preview_parameters = None
            self._clear_interevent_result_display()
            self.interevent_status_var.set(str(exc))
            messagebox.showerror("Inter-event displacement", str(exc))

    def _refresh_event_widgets(self) -> None:
        names = list(self.signal_candidates)
        self.selected_signal_name = None
        self.preview_result = None
        self.preview_parameters = None
        self.signal_combobox.configure(values=names)
        self.signal_combobox.set("")
        if names:
            self.preview_button.configure(state="disabled")
            self.status_var.set("Select a signal to enable preview")
        else:
            self.preview_button.configure(state="disabled")
            self.status_var.set("No numeric one-dimensional event signal is available")
        self._clear_result_display()
        self._refresh_interevent_widgets()
        self._plot_preview()

    def _clear_result_display(self) -> None:
        for variable in self.result_vars.values():
            variable.set("—")

    def _invalidate_preview(self) -> None:
        self.preview_result = None
        self.preview_parameters = None
        self._clear_result_display()
        self.status_var.set("Fitting windows changed — recompute preview")

    def _on_parameter_changed(self) -> None:
        if not hasattr(self, "result_vars"):
            return
        self._invalidate_preview()
        if hasattr(self, "raw_ax"):
            self._plot_preview()

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
        signal_name = self.signal_combobox.get()
        if signal_name in self.signal_candidates:
            self.selected_signal_name = signal_name
            self.status_var.set("Preview only — not saved")
        else:
            self.selected_signal_name = None
        self.preview_button.configure(
            state="normal" if self.selected_signal_name else "disabled"
        )
        if self.selected_signal_name is None:
            self.status_var.set("Select a signal to enable preview")
        self._plot_preview()

    def _resolve_selected_signal(self) -> np.ndarray:
        signal_name = self.selected_signal_name
        if not signal_name or signal_name not in self.signal_candidates:
            raise ValueError("No available event signal is selected")
        return self.signal_candidates[signal_name]

    def _read_parameter_group(self, variables) -> dict[str, Any]:
        return parse_preview_parameters(
            variables["half_win"].get(),
            variables["pre_start"].get(),
            variables["pre_end"].get(),
            variables["post_start"].get(),
            variables["post_end"].get(),
            variables["smooth_w"].get(),
        )

    def calculate_preview(self) -> dict[str, dict[str, Any]]:
        """Calculate the selected event-local signal drop in memory."""
        signal = self._resolve_selected_signal()
        parameter_values = self._read_parameter_group(self.parameter_vars)
        signals = {"signal": signal}
        parameters = {"signal": parameter_values}
        results = calculate_event_drop_metrics(
            time=self.event["time"],
            event_time=self.event["event_time"],
            signals=signals,
            parameters=parameters,
        )
        self.preview_result = results["signal"]
        self.preview_parameters = parameter_values
        return self.preview_result

    def recompute_preview(self) -> None:
        try:
            result = self.calculate_preview()
            self._clear_result_display()
            display = format_preview_result(result)
            self.result_vars["signal_name"].set(self.selected_signal_name or "—")
            for key in ("valid", "delta", "magnitude", "val_pre_0", "val_post_0"):
                self.result_vars[key].set(display[key])
            self.status_var.set("Preview only — not saved")
            self._plot_preview()
        except ValueError as exc:
            self.preview_result = None
            self.preview_parameters = None
            self._clear_result_display()
            self.status_var.set(str(exc))
            messagebox.showerror("Event Drop Preview", str(exc))

    def _current_points(self) -> tuple[float, float, float, float] | None:
        variables = self.parameter_vars
        try:
            points = tuple(
                float(variables[key].get())
                for key in ("pre_start", "pre_end", "post_start", "post_end")
            )
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(point) for point in points):
            return None
        return points

    def _constrain_endpoint(self, endpoint: str, position: float) -> float:
        variables = self.parameter_vars
        try:
            half_win = float(variables["half_win"].get())
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

        signal_name = self.selected_signal_name
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

        preview_result = self.preview_result
        if preview_result and preview_result.get("valid") and points is not None:
            pre_x = np.linspace(min(points[0], points[1]), 0.0, 100)
            post_x = np.linspace(0.0, max(points[2], points[3]), 100)
            coeff_pre = preview_result["coeff_pre"]
            coeff_post = preview_result["coeff_post"]
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
