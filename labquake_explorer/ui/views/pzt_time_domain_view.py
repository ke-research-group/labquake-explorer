"""Read-only PZT time-domain preview for canonical event strain data."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Mapping, Sequence
from numbers import Real
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.analysis.pzt_analysis_seismology import (
    DEFAULT_POST_SEC,
    DEFAULT_PRE_SEC,
    DEFAULT_THRESHOLD,
    WINDOW_TYPE,
    BlockTrace,
    TimeWindowResult,
    compute_time_window,
)


def parse_time_domain_parameters(
    pre_sec: str,
    post_sec: str,
    threshold: str,
) -> dict[str, float]:
    """Parse the three parameters accepted by Tim's time-window helper."""
    parsed = {}
    for name, text in (
        ("pre_sec", pre_sec),
        ("post_sec", post_sec),
        ("threshold", threshold),
    ):
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        parsed[name] = value
    if parsed["pre_sec"] < 0:
        raise ValueError("pre_sec must be non-negative")
    if parsed["post_sec"] < 0:
        raise ValueError("post_sec must be non-negative")
    if parsed["threshold"] < 0:
        raise ValueError("threshold must be non-negative")
    return parsed


class PZTTimeDomainView(tk.Toplevel):
    """Preview Tim's BAC time-window preparation on explicitly bound strain."""

    def __init__(self, parent, run_idx: int, event_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"PZT Time Domain Preview - Event {event_idx}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data_manager = self.parent.data_manager
        self.run_idx = run_idx
        self.event_idx = event_idx
        self.events: Any = None
        self.event: Mapping[str, Any] | None = None
        self.strain_time: np.ndarray | None = None
        self.strain_raw: np.ndarray | None = None
        self.channel_candidates: list[int] = []
        self.selected_channel_index: int | None = None
        self.preview_result: TimeWindowResult | None = None
        self.preview_trace: BlockTrace | None = None
        self.preview_parameters: dict[str, float | str] | None = None
        self._initializing_parameters = True

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

    @staticmethod
    def _resolve_strain_context(
        event: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[int], str | None]:
        if not isinstance(event, Mapping):
            return None, None, [], "Event is not a mapping"
        strain = event.get("strain")
        if not isinstance(strain, Mapping):
            return None, None, [], "Event has no dynamic strain data"
        original = strain.get("original")
        if not isinstance(original, Mapping):
            return None, None, [], "Event strain has no original data"
        try:
            time = np.asarray(original.get("time"))
            raw = np.asarray(original.get("raw"))
        except (TypeError, ValueError) as exc:
            return None, None, [], f"Invalid dynamic strain arrays: {exc}"
        if time.ndim != 1 or time.size == 0:
            return None, None, [], "Dynamic strain time must be a non-empty 1-D array"
        if time.dtype.kind not in "iuf" or time.dtype.kind == "b":
            return None, None, [], "Dynamic strain time must be real numeric data"
        if raw.ndim not in (1, 2):
            return None, None, [], "Dynamic strain raw data must be 1-D or channel-by-sample 2-D"
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            return None, None, [], "Dynamic strain raw data must be real numeric data"
        sample_count = raw.shape[0] if raw.ndim == 1 else raw.shape[1]
        if sample_count != time.size:
            return None, None, [], "Dynamic strain time and channel lengths do not match"
        try:
            finite = np.all(np.isfinite(time)) and np.all(np.isfinite(raw))
        except TypeError:
            finite = False
        if not finite:
            return None, None, [], "Dynamic strain time and raw data must be finite"
        channel_count = 1 if raw.ndim == 1 else raw.shape[0]
        return time, raw, list(range(channel_count)), None

    def _set_event(self, event_idx: int) -> None:
        self.event_idx = event_idx
        self.events = self.data_manager.get_data(f"runs/[{self.run_idx}]/events")
        self.event = self.data_manager.get_data(
            f"runs/[{self.run_idx}]/events/[{self.event_idx}]"
        )
        (
            self.strain_time,
            self.strain_raw,
            self.channel_candidates,
            self.context_error,
        ) = self._resolve_strain_context(self.event)
        self.selected_channel_index = None
        self.preview_result = None
        self.preview_trace = None
        self.preview_parameters = None

    def _create_controls(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(controls, text="Event:").grid(row=0, column=0, sticky="e")
        self.event_combobox = ttk.Combobox(controls, width=8, state="readonly")
        self.event_combobox.grid(row=0, column=1, padx=(3, 10), sticky="w")
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_event_changed)

        ttk.Label(controls, text="Channel:").grid(row=0, column=2, sticky="e")
        self.channel_combobox = ttk.Combobox(controls, width=16, state="readonly")
        self.channel_combobox.grid(row=0, column=3, padx=(3, 10), sticky="w")
        self.channel_combobox.bind("<<ComboboxSelected>>", self.on_channel_changed)

        self.pre_sec_var = tk.StringVar(value=str(DEFAULT_PRE_SEC))
        self.post_sec_var = tk.StringVar(value=str(DEFAULT_POST_SEC))
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        for column, (label, variable) in enumerate(
            (
                ("Pre (s):", self.pre_sec_var),
                ("Post (s):", self.post_sec_var),
                ("Threshold:", self.threshold_var),
            ),
            start=4,
        ):
            ttk.Label(controls, text=label).grid(row=0, column=2 * column - 4, sticky="e")
            entry = ttk.Entry(controls, textvariable=variable, width=8)
            entry.grid(row=0, column=2 * column - 3, padx=(3, 8), sticky="w")
            variable.trace_add("write", self.on_parameters_changed)

        window_label = "Blackman-Harris" if WINDOW_TYPE.lower() == "bh" else WINDOW_TYPE
        ttk.Label(controls, text=f"Window: {window_label}").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )
        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=1, column=10, columnspan=2, sticky="e", pady=(5, 0))
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=2, column=0, columnspan=12, sticky="w", pady=(5, 0)
        )

    def _create_figure(self) -> None:
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _initialize_event_selector(self) -> None:
        count = len(self.events) if isinstance(self.events, Sequence) else 0
        self.event_combobox.configure(values=[str(index) for index in range(count)])
        self.event_combobox.set(str(self.event_idx))

    def _refresh_event_widgets(self) -> None:
        labels = [f"strain[{index}]" for index in self.channel_candidates]
        self.channel_combobox.configure(values=labels)
        self.channel_combobox.set("")
        self.channel_combobox.configure(state="readonly" if labels else "disabled")
        self.preview_button.configure(state="disabled")
        self._clear_preview(redraw=True)
        if self.context_error:
            self.status_var.set(self.context_error)
        elif self._coerce_event_time(self.event.get("event_time")) is None:
            self.status_var.set("Event has no finite event_time")
        else:
            self.status_var.set("Select a strain channel — preview only, not saved")

    def on_channel_changed(self, _event=None) -> None:
        value = self.channel_combobox.get()
        index = None
        if value.startswith("strain[") and value.endswith("]"):
            try:
                candidate = int(value[7:-1])
            except ValueError:
                candidate = -1
            if candidate in self.channel_candidates:
                index = candidate
        self.selected_channel_index = index
        self.preview_button.configure(state="normal" if index is not None else "disabled")
        self._clear_preview(redraw=True)
        self.status_var.set(
            "Ready — preview only, not saved"
            if index is not None
            else "Select a strain channel — preview only, not saved"
        )

    def on_event_changed(self, _event=None) -> None:
        try:
            event_idx = int(self.event_combobox.get())
            self._set_event(event_idx)
            self._refresh_event_widgets()
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            self._show_error(exc)

    def on_parameters_changed(self, *_args) -> None:
        if self._initializing_parameters:
            return
        self._clear_preview(redraw=True)
        self.status_var.set("Parameters changed — recompute preview")

    def _clear_preview(self, *, redraw: bool) -> None:
        self.preview_result = None
        self.preview_trace = None
        self.preview_parameters = None
        if redraw and hasattr(self, "ax"):
            self.ax.clear()
            self.canvas.draw_idle()

    def _build_selected_trace(self) -> BlockTrace:
        if self.selected_channel_index not in self.channel_candidates:
            raise ValueError("Select a valid strain channel")
        if self.strain_time is None or self.strain_raw is None:
            raise ValueError(self.context_error or "Dynamic strain data is unavailable")
        event_time = self._coerce_event_time(
            self.event.get("event_time") if isinstance(self.event, Mapping) else None
        )
        if event_time is None:
            raise ValueError("event_time must be a finite real scalar")

        time = self.strain_time
        signal = self.strain_raw if self.strain_raw.ndim == 1 else self.strain_raw[self.selected_channel_index]
        if time.ndim != 1 or signal.ndim != 1 or time.size != signal.size or time.size < 2:
            raise ValueError("Selected strain time and signal must be equal-length 1-D arrays")
        if time.dtype.kind not in "iuf" or signal.dtype.kind not in "iuf":
            raise ValueError("Selected strain time and signal must be real numeric arrays")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
            raise ValueError("Selected strain time and signal must be finite")
        differences = np.diff(time)
        if not np.all(differences > 0):
            raise ValueError("Dynamic strain time must be strictly increasing")

        sampling_rate = 1.0 / float(np.median(differences))
        trigger_sample = int(np.argmin(np.abs(time - event_time)))
        return BlockTrace(
            time=time,
            voltage=signal,
            sampling_rate=sampling_rate,
            trigger_time=event_time,
            trigger_sample=trigger_sample,
            block=self.event_idx,
            sensor=f"strain[{self.selected_channel_index}]",
        )

    def recompute_preview(self) -> None:
        try:
            parameters = parse_time_domain_parameters(
                self.pre_sec_var.get(), self.post_sec_var.get(), self.threshold_var.get()
            )
            trace = self._build_selected_trace()
            result = compute_time_window(trace, **parameters)
            self.preview_trace = trace
            self.preview_result = result
            self.preview_parameters = {**parameters, "window_type": WINDOW_TYPE}
            self._draw_preview(trace, result)
            self.status_var.set(
                f"Valid — {trace.sensor}, {trace.sampling_rate:.6g} Hz; preview only, not saved"
            )
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            self._show_error(exc)

    def _draw_preview(self, trace: BlockTrace, result: TimeWindowResult) -> None:
        self.ax.clear()
        relative_ms = (trace.time - trace.trigger_time) * 1e3
        signal_x = relative_ms[result.i0 : result.i1 + 1]
        self.ax.plot(relative_ms, trace.voltage, color="C0", linewidth=1.0, label="AE raw signal (t)")
        self.ax.plot(
            signal_x,
            result.voltage_windowed,
            color="green",
            linewidth=1.2,
            label="Windowed signal",
        )
        self.ax.plot(
            signal_x,
            result.window_scaled_voltage,
            color="C4",
            linestyle="--",
            linewidth=1.0,
            label="Blackman-Harris taper",
        )
        self.ax.axvspan(signal_x[0], signal_x[-1], color="C2", alpha=0.12, label="Signal window")
        noise_x = relative_ms[result.noise_start : result.noise_end]
        if noise_x.size:
            self.ax.axvspan(noise_x[0], noise_x[-1], color="C3", alpha=0.10, label="Noise window")
        self.ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, label="Canonical event")
        self.ax.set_xlabel("Time relative to event (ms)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title(f"PZT Time Domain - {trace.sensor}")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _show_error(self, exc: Exception) -> None:
        self._clear_preview(redraw=True)
        self.status_var.set(str(exc))
        messagebox.showerror("PZT time domain", str(exc), parent=self)

    def on_close(self) -> None:
        if self in self.parent.child_windows:
            self.parent.child_windows.remove(self)
        self.destroy()
