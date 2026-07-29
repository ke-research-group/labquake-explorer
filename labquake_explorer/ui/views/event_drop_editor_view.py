"""Preview explicitly bound scalar and ordered slip event-drop signals."""

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
    """Preview explicitly bound scalar and ordered slip event-drop signals."""

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
        self.metric_bindings: dict[str, str | None] = {
            "tau": None,
            "mu": None,
            "lvdt": None,
        }
        self.slip_bindings: list[str | None] = []
        self.slip_rows: list[dict[str, Any]] = []
        self.preview_results: dict[str, dict[str, Any]] = {}
        self.active_metric_role: str | None = None
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

        ttk.Label(controls, text="Tau signal:").grid(row=0, column=2, sticky="e")
        self.tau_signal_combobox = ttk.Combobox(
            controls, width=20, state="readonly"
        )
        self.tau_signal_combobox.grid(row=0, column=3, padx=(3, 10), sticky="w")
        self.tau_signal_combobox.bind(
            "<<ComboboxSelected>>", self.on_tau_signal_changed
        )

        ttk.Label(controls, text="Mu signal:").grid(row=0, column=4, sticky="e")
        self.mu_signal_combobox = ttk.Combobox(
            controls, width=20, state="readonly"
        )
        self.mu_signal_combobox.grid(row=0, column=5, padx=(3, 10), sticky="w")
        self.mu_signal_combobox.bind(
            "<<ComboboxSelected>>", self.on_mu_signal_changed
        )
        ttk.Label(controls, text="LVDT signal:").grid(row=0, column=6, sticky="e")
        self.lvdt_signal_combobox = ttk.Combobox(
            controls, width=20, state="readonly"
        )
        self.lvdt_signal_combobox.grid(row=0, column=7, padx=(3, 10), sticky="w")
        self.lvdt_signal_combobox.bind(
            "<<ComboboxSelected>>", self.on_lvdt_signal_changed
        )

        defaults = {
            "half_win": str(compute_half_win()),
            "pre_start": str(DEFAULT_POINTS[0]),
            "pre_end": str(DEFAULT_POINTS[1]),
            "post_start": str(DEFAULT_POINTS[2]),
            "post_end": str(DEFAULT_POINTS[3]),
            "smooth_w": "",
        }
        ttk.Label(controls, text="Tau / Mu fitting").grid(
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

        ttk.Label(controls, text="LVDT fitting").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(4, 0)
        )
        self.lvdt_parameter_vars: dict[str, tk.StringVar] = {}
        for column, (label, key) in enumerate(labels):
            ttk.Label(controls, text=label).grid(row=5, column=column, padx=3)
            variable = tk.StringVar(value=defaults[key])
            self.lvdt_parameter_vars[key] = variable
            ttk.Entry(controls, textvariable=variable, width=11).grid(
                row=6, column=column, padx=3
            )

        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=6, column=6, padx=(10, 3))

        results = ttk.LabelFrame(self, text="Preview Result")
        results.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.result_vars = {
            role: {
                key: tk.StringVar(value="—")
                for key in (
                    "valid",
                    "delta",
                    "magnitude",
                    "val_pre_0",
                    "val_post_0",
                )
            }
            for role in ("tau", "mu", "lvdt")
        }
        result_labels = (
            ("Valid", "valid"),
            ("Signed delta", "delta"),
            ("Magnitude", "magnitude"),
            ("Pre fit at t=0", "val_pre_0"),
            ("Post fit at t=0", "val_post_0"),
        )
        for row, role in enumerate(("tau", "mu", "lvdt")):
            display_name = "LVDT" if role == "lvdt" else role.capitalize()
            ttk.Label(results, text=display_name).grid(
                row=row, column=0, padx=(6, 10), pady=3, sticky="w"
            )
            for column, (label, key) in enumerate(result_labels):
                ttk.Label(results, text=f"{label}:").grid(
                    row=row, column=2 * column + 1, padx=(6, 2), pady=3
                )
                ttk.Label(
                    results, textvariable=self.result_vars[role][key]
                ).grid(
                    row=row, column=2 * column + 2, padx=(0, 8), pady=3
                )
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(results, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=11, padx=6, pady=(0, 3), sticky="w"
        )

        slip = ttk.LabelFrame(self, text="Slip sensors (display order only)")
        slip.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
        self.slip_rows_frame = ttk.Frame(slip)
        self.slip_rows_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(
            slip, text="Add slip sensor", command=self.add_slip_sensor
        ).pack(side=tk.TOP, anchor="w", padx=6, pady=3)

        slip_parameters = ttk.LabelFrame(slip, text="Slip fitting")
        slip_parameters.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        self.slip_parameter_vars: dict[str, tk.StringVar] = {}
        slip_labels = labels[:-1]
        for column, (label, key) in enumerate(slip_labels):
            ttk.Label(slip_parameters, text=label).grid(
                row=0, column=column, padx=3
            )
            variable = tk.StringVar(value=defaults[key])
            self.slip_parameter_vars[key] = variable
            ttk.Entry(
                slip_parameters, textvariable=variable, width=11
            ).grid(row=1, column=column, padx=3)

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
        self.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        self.slip_bindings = []
        self.preview_results = {}
        self.active_metric_role = None
        self.preview_parameters = None

    def _refresh_event_widgets(self) -> None:
        names = list(self.signal_candidates)
        self.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        self.slip_bindings = []
        self.preview_results = {}
        self.preview_parameters = None
        self.active_metric_role = None
        for combobox in (
            self.tau_signal_combobox,
            self.mu_signal_combobox,
            self.lvdt_signal_combobox,
        ):
            combobox.configure(values=names)
            combobox.set("")
        self._rebuild_slip_rows()
        if names:
            self.preview_button.configure(state="disabled")
            self.status_var.set("Select a signal to enable preview")
        else:
            self.preview_button.configure(state="disabled")
            self.status_var.set("No numeric one-dimensional event signal is available")
        self._clear_result_display()
        self._plot_preview()

    def _clear_result_display(self, role: str | None = None) -> None:
        if role and role.startswith("slip_"):
            index = int(role.split("_", 1)[1]) - 1
            if 0 <= index < len(self.slip_rows):
                for variable in self.slip_rows[index]["result_vars"].values():
                    variable.set("—")
            return
        roles = (role,) if role else ("tau", "mu", "lvdt")
        for metric_role in roles:
            for variable in self.result_vars[metric_role].values():
                variable.set("—")
        if role is None:
            for row in getattr(self, "slip_rows", []):
                for variable in row["result_vars"].values():
                    variable.set("—")

    def _invalidate_preview(self) -> None:
        if self.active_metric_role and self.active_metric_role.startswith("slip_"):
            roles = tuple(
                role for role in list(self.preview_results) if role.startswith("slip_")
            )
        elif self.active_metric_role == "lvdt":
            roles = ("lvdt",)
        else:
            roles = ("tau", "mu")
        for role in roles:
            self.preview_results.pop(role, None)
        self.preview_parameters = None
        for role in roles:
            self._clear_result_display(role)
        self.status_var.set("Fitting windows changed — recompute preview")

    def on_event_changed(self, event=None) -> None:
        try:
            event_idx = int(self.event_combobox.get())
            self._set_event(event_idx)
            self.title(f"Event Drop Preview - Event {event_idx}")
            self._refresh_event_widgets()
        except Exception as exc:
            messagebox.showerror("Event Drop Preview", f"Failed to load event: {exc}")

    def _on_metric_signal_changed(self, role: str, combobox) -> None:
        self.preview_results.pop(role, None)
        self.preview_parameters = None
        self._clear_result_display(role)
        signal_name = combobox.get()
        if signal_name in self.signal_candidates:
            self.metric_bindings[role] = signal_name
            self.active_metric_role = role
            self.status_var.set("Preview only — not saved")
        else:
            self.metric_bindings[role] = None
            if self.active_metric_role == role:
                self.active_metric_role = self._fallback_active_role()
        has_binding = self._has_any_binding()
        self.preview_button.configure(state="normal" if has_binding else "disabled")
        if not has_binding:
            self.status_var.set("Select a signal to enable preview")
        self._plot_preview()

    def on_tau_signal_changed(self, event=None) -> None:
        self._on_metric_signal_changed("tau", self.tau_signal_combobox)

    def on_mu_signal_changed(self, event=None) -> None:
        self._on_metric_signal_changed("mu", self.mu_signal_combobox)

    def on_lvdt_signal_changed(self, event=None) -> None:
        self._on_metric_signal_changed("lvdt", self.lvdt_signal_combobox)

    def _slip_role(self, index: int) -> str:
        return f"slip_{index + 1}"

    def _binding_for_role(self, role: str) -> str | None:
        if role.startswith("slip_"):
            index = int(role.split("_", 1)[1]) - 1
            return self.slip_bindings[index] if index < len(self.slip_bindings) else None
        return self.metric_bindings.get(role)

    def _fallback_active_role(self) -> str | None:
        for role in ("tau", "mu", "lvdt"):
            if self.metric_bindings[role]:
                return role
        for index, binding in enumerate(getattr(self, "slip_bindings", [])):
            if binding:
                return self._slip_role(index)
        return None

    def _has_any_binding(self) -> bool:
        return any(self.metric_bindings.values()) or any(
            getattr(self, "slip_bindings", [])
        )

    def add_slip_sensor(self) -> None:
        self.slip_bindings.append(None)
        self._rebuild_slip_rows()
        self.preview_button.configure(
            state="normal" if self._has_any_binding() else "disabled"
        )

    def remove_slip_sensor(self, index: int) -> None:
        del self.slip_bindings[index]
        self.preview_results = {
            role: result
            for role, result in self.preview_results.items()
            if not role.startswith("slip_")
        }
        if self.active_metric_role and self.active_metric_role.startswith("slip_"):
            self.active_metric_role = self._fallback_active_role()
        self._rebuild_slip_rows()
        self.preview_button.configure(
            state="normal" if self._has_any_binding() else "disabled"
        )
        self._plot_preview()

    def _rebuild_slip_rows(self) -> None:
        for row in getattr(self, "slip_rows", []):
            frame = row.get("frame")
            if frame is not None:
                frame.destroy()
        self.slip_rows = []
        if not hasattr(self, "slip_rows_frame"):
            return
        names = list(self.signal_candidates)
        result_keys = ("valid", "delta", "magnitude", "val_pre_0", "val_post_0")
        for index, binding in enumerate(self.slip_bindings):
            frame = ttk.Frame(self.slip_rows_frame)
            frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
            ttk.Label(frame, text=f"Slip {index + 1}").pack(side=tk.LEFT)
            selector = ttk.Combobox(frame, width=20, state="readonly", values=names)
            selector.pack(side=tk.LEFT, padx=4)
            selector.set(binding or "")
            selector.bind(
                "<<ComboboxSelected>>",
                lambda event, slot=index: self.on_slip_signal_changed(slot),
            )
            result_vars = {key: tk.StringVar(value="—") for key in result_keys}
            for key, label in (
                ("valid", "Valid"),
                ("delta", "Signed delta"),
                ("magnitude", "Magnitude"),
                ("val_pre_0", "Pre t=0"),
                ("val_post_0", "Post t=0"),
            ):
                ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT, padx=(5, 1))
                ttk.Label(frame, textvariable=result_vars[key]).pack(side=tk.LEFT)
            ttk.Button(
                frame,
                text="Remove",
                command=lambda slot=index: self.remove_slip_sensor(slot),
            ).pack(side=tk.RIGHT)
            self.slip_rows.append(
                {"frame": frame, "selector": selector, "result_vars": result_vars}
            )

    def on_slip_signal_changed(self, index: int) -> None:
        role = self._slip_role(index)
        signal_name = self.slip_rows[index]["selector"].get()
        self.preview_results.pop(role, None)
        self._clear_result_display(role)
        if signal_name in self.signal_candidates:
            self.slip_bindings[index] = signal_name
            self.active_metric_role = role
            self.status_var.set("Preview only — not saved")
        else:
            self.slip_bindings[index] = None
            if self.active_metric_role == role:
                self.active_metric_role = self._fallback_active_role()
        self.preview_parameters = None
        self.preview_button.configure(
            state="normal" if self._has_any_binding() else "disabled"
        )
        if not self._has_any_binding():
            self.status_var.set("Select a signal to enable preview")
        self._plot_preview()

    def _resolve_metric_signal(self, role: str) -> np.ndarray:
        signal_name = self.metric_bindings.get(role)
        if not signal_name or signal_name not in self.signal_candidates:
            raise ValueError(f"No available signal is bound to {role}")
        return self.signal_candidates[signal_name]

    def _resolve_bound_metrics(self) -> dict[str, np.ndarray]:
        signals = {}
        for role in ("tau", "mu", "lvdt"):
            if self.metric_bindings[role] is not None:
                signals[role] = self._resolve_metric_signal(role)
        for index, signal_name in enumerate(getattr(self, "slip_bindings", [])):
            if signal_name is None:
                continue
            role = self._slip_role(index)
            if signal_name not in self.signal_candidates:
                raise ValueError(f"No available signal is bound to {role}")
            signals[role] = self.signal_candidates[signal_name]
        if not signals:
            raise ValueError("Select at least one signal to enable preview")
        return signals

    def _read_parameter_group(self, variables) -> dict[str, Any]:
        return parse_preview_parameters(
            variables["half_win"].get(),
            variables["pre_start"].get(),
            variables["pre_end"].get(),
            variables["post_start"].get(),
            variables["post_end"].get(),
            variables["smooth_w"].get(),
        )

    def _read_bound_metric_parameters(self) -> dict[str, dict[str, Any]]:
        parameters = {}
        if self.metric_bindings["tau"] or self.metric_bindings["mu"]:
            shared = self._read_parameter_group(self.parameter_vars)
            for role in ("tau", "mu"):
                if self.metric_bindings[role]:
                    parameters[role] = dict(shared)
        if self.metric_bindings["lvdt"]:
            parameters["lvdt"] = dict(
                self._read_parameter_group(self.lvdt_parameter_vars)
            )
        if any(getattr(self, "slip_bindings", [])):
            slip = parse_preview_parameters(
                self.slip_parameter_vars["half_win"].get(),
                self.slip_parameter_vars["pre_start"].get(),
                self.slip_parameter_vars["pre_end"].get(),
                self.slip_parameter_vars["post_start"].get(),
                self.slip_parameter_vars["post_end"].get(),
                "",
            )
            for index, binding in enumerate(self.slip_bindings):
                if binding:
                    parameters[self._slip_role(index)] = dict(slip)
        return parameters

    def calculate_preview(self) -> dict[str, dict[str, Any]]:
        """Calculate all explicitly bound event-drop metrics in memory."""
        signals = self._resolve_bound_metrics()
        parameters = self._read_bound_metric_parameters()
        results = calculate_event_drop_metrics(
            time=self.event["time"],
            event_time=self.event["event_time"],
            signals=signals,
            parameters=parameters,
        )
        self.preview_results = results
        self.preview_parameters = parameters
        return results

    def recompute_preview(self) -> None:
        try:
            results = self.calculate_preview()
            self._clear_result_display()
            for role, result in results.items():
                display = format_preview_result(result)
                if role.startswith("slip_"):
                    index = int(role.split("_", 1)[1]) - 1
                    variables = self.slip_rows[index]["result_vars"]
                else:
                    variables = self.result_vars[role]
                for key, variable in variables.items():
                    variable.set(display[key])
            self.status_var.set("Preview only — not saved")
            self._plot_preview()
        except ValueError as exc:
            self.preview_results = {}
            self.preview_parameters = None
            self._clear_result_display()
            self.status_var.set(str(exc))
            messagebox.showerror("Event Drop Preview", str(exc))

    def _current_points(self) -> tuple[float, float, float, float] | None:
        variables = self._active_parameter_vars()
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
        variables = self._active_parameter_vars()
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
        self._active_parameter_vars()[endpoint].set(f"{position:.6g}")
        self._invalidate_preview()

    def _active_parameter_vars(self):
        if self.active_metric_role and self.active_metric_role.startswith("slip_"):
            return self.slip_parameter_vars
        if self.active_metric_role == "lvdt":
            return self.lvdt_parameter_vars
        return self.parameter_vars

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

        role = self.active_metric_role
        signal_name = self._binding_for_role(role) if role else None
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

        preview_result = self.preview_results.get(role) if role else None
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
