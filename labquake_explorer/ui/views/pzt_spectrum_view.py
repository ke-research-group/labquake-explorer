"""Read-only canonical-trigger PZT spectrum preview."""

from __future__ import annotations

import math
import re
import tkinter as tk
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pandas.errors import EmptyDataError, ParserError

from labquake_explorer.analysis.pzt_analysis_seismology import (
    DEFAULT_FIT_PARAMETER_LB,
    DEFAULT_FIT_PARAMETER_UB,
    DEFAULT_LNF_MAX_Q,
    DEFAULT_LNF_MIN_Q,
    DEFAULT_NFFT,
    DEFAULT_POST_SEC,
    DEFAULT_PRE_SEC,
    BlockTrace,
    FitResult,
    SpectrumResult,
    compute_spectrum_at_trigger,
    fit_omega_n_q,
    scale_spectrum_for_seismology_fit,
    validate_fit_parameter_bounds,
)


_EXPECTED_PREVIEW_EXCEPTIONS = (
    ValueError,
    TypeError,
    IndexError,
    KeyError,
    FileNotFoundError,
    OSError,
    UnicodeError,
    ParserError,
    EmptyDataError,
)

_EXPECTED_FIT_EXCEPTIONS = (ValueError, TypeError, IndexError, KeyError)


def parse_omega_n_fit_parameters(
    lnf_min: str,
    lnf_max: str,
    omega0_lb: str,
    fc_lb: str,
    n_lb: str,
    omega0_ub: str,
    fc_ub: str,
    n_ub: str,
) -> dict[str, Any]:
    """Parse and validate the explicit omega-n fitting controls."""
    fields = (
        ("ln(f) minimum", lnf_min),
        ("ln(f) maximum", lnf_max),
        ("Omega0 lower bound", omega0_lb),
        ("fc lower bound", fc_lb),
        ("n lower bound", n_lb),
        ("Omega0 upper bound", omega0_ub),
        ("fc upper bound", fc_ub),
        ("n upper bound", n_ub),
    )
    parsed = []
    for name, text in fields:
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        parsed.append(value)

    fit_min, fit_max, *bounds = parsed
    if fit_min >= fit_max:
        raise ValueError("ln(f) minimum must be smaller than ln(f) maximum")
    parameter_lb = tuple(bounds[:3])
    parameter_ub = tuple(bounds[3:])
    for name, lower, upper in zip(("Omega0", "fc", "n"), parameter_lb, parameter_ub):
        if lower <= 0:
            raise ValueError(f"{name} lower bound must be positive")
        if upper <= 0:
            raise ValueError(f"{name} upper bound must be positive")
        if lower >= upper:
            raise ValueError(f"{name} lower bound must be smaller than its upper bound")
    validate_fit_parameter_bounds(parameter_lb, parameter_ub)
    return {
        "lnf_min": fit_min,
        "lnf_max": fit_max,
        "parameter_lb": parameter_lb,
        "parameter_ub": parameter_ub,
    }


def parse_spectrum_parameters(
    pre_sec: str,
    post_sec: str,
    nfft: str,
    calibration_path: str,
    q_path: str,
) -> dict[str, Any]:
    """Parse spectrum controls without reading either CSV file."""
    parsed = {}
    for name, text in (("pre_sec", pre_sec), ("post_sec", post_sec)):
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        parsed[name] = value

    nfft_text = str(nfft).strip()
    if re.fullmatch(r"[1-9]\d*", nfft_text) is None:
        raise ValueError("nfft must be a positive integer")

    paths = {}
    for name, text in (("calibration_csv", calibration_path), ("q_csv", q_path)):
        if not str(text).strip():
            raise ValueError(f"{name} path must not be empty")
        path = Path(str(text).strip())
        if not path.exists():
            raise ValueError(f"{name} path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{name} path must be a file: {path}")
        paths[name] = path
    return {**parsed, "nfft": int(nfft_text), **paths}


class PZTSpectrumView(tk.Toplevel):
    """Preview Tim BAC spectra for an explicitly selected canonical channel."""

    def __init__(self, parent, run_idx: int, event_idx: int):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title(f"PZT Spectrum Preview - Event {event_idx}")
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
        self.context_error: str | None = None
        self.preview_trace: BlockTrace | None = None
        self.preview_result: SpectrumResult | None = None
        self.preview_parameters: dict[str, Any] | None = None
        self.fit_result: FitResult | None = None
        self.fit_parameters: dict[str, Any] | None = None
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
        if time.ndim != 1 or time.size == 0 or time.dtype.kind not in "iuf":
            return None, None, [], "Dynamic strain time must be a non-empty real numeric 1-D array"
        if raw.ndim not in (1, 2) or raw.dtype.kind not in "iuf":
            return None, None, [], "Dynamic strain raw data must be real numeric 1-D or channel-by-sample 2-D"
        sample_count = raw.shape[0] if raw.ndim == 1 else raw.shape[1]
        if sample_count != time.size:
            return None, None, [], "Dynamic strain time and channel lengths do not match"
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(raw)):
            return None, None, [], "Dynamic strain time and raw data must be finite"
        if time.size < 2 or not np.all(np.diff(time) > 0):
            return None, None, [], "Dynamic strain time must be strictly increasing"
        count = 1 if raw.ndim == 1 else raw.shape[0]
        return time, raw, list(range(count)), None

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
        self.preview_trace = None
        self.preview_result = None
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
        self.nfft_var = tk.StringVar(value=str(DEFAULT_NFFT))
        for column, (label, variable) in enumerate(
            (("Pre (s):", self.pre_sec_var), ("Post (s):", self.post_sec_var), ("NFFT:", self.nfft_var)),
            start=4,
        ):
            ttk.Label(controls, text=label).grid(row=0, column=2 * column - 4, sticky="e")
            ttk.Entry(controls, textvariable=variable, width=9).grid(
                row=0, column=2 * column - 3, padx=(3, 8), sticky="w"
            )
            variable.trace_add("write", self.on_parameters_changed)

        self.calibration_path_var = tk.StringVar(value="")
        self.q_path_var = tk.StringVar(value="")
        for row, (label, variable, command) in enumerate(
            (
                ("Calibration CSV:", self.calibration_path_var, self.browse_calibration),
                ("Q CSV:", self.q_path_var, self.browse_q),
            ),
            start=1,
        ):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="e", pady=(4, 0))
            ttk.Entry(controls, textvariable=variable, width=70).grid(
                row=row, column=1, columnspan=8, padx=(3, 5), pady=(4, 0), sticky="ew"
            )
            ttk.Button(controls, text="Browse", command=command).grid(
                row=row, column=9, pady=(4, 0), sticky="w"
            )
            variable.trace_add("write", self.on_parameters_changed)

        self.preview_button = ttk.Button(
            controls, text="Preview / Recompute", command=self.recompute_preview
        )
        self.preview_button.grid(row=1, column=10, columnspan=2, padx=(8, 0), sticky="e")
        self.status_var = tk.StringVar(value="Preview only — not saved")
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=12, sticky="w", pady=(5, 0)
        )

        fitting = ttk.LabelFrame(self, text="Omega-n fitting")
        fitting.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        self.lnf_min_var = tk.StringVar(value=str(DEFAULT_LNF_MIN_Q))
        self.lnf_max_var = tk.StringVar(value=str(DEFAULT_LNF_MAX_Q))
        lower = [tk.StringVar(value=str(value)) for value in DEFAULT_FIT_PARAMETER_LB]
        upper = [tk.StringVar(value=str(value)) for value in DEFAULT_FIT_PARAMETER_UB]
        self.omega0_lb_var, self.fc_lb_var, self.n_lb_var = lower
        self.omega0_ub_var, self.fc_ub_var, self.n_ub_var = upper
        fit_fields = (
            ("ln(f) min:", self.lnf_min_var),
            ("ln(f) max:", self.lnf_max_var),
            ("Omega0 LB:", self.omega0_lb_var),
            ("fc LB:", self.fc_lb_var),
            ("n LB:", self.n_lb_var),
            ("Omega0 UB:", self.omega0_ub_var),
            ("fc UB:", self.fc_ub_var),
            ("n UB:", self.n_ub_var),
        )
        for column, (label, variable) in enumerate(fit_fields):
            ttk.Label(fitting, text=label).grid(row=0, column=2 * column, sticky="e")
            ttk.Entry(fitting, textvariable=variable, width=9).grid(
                row=0, column=2 * column + 1, padx=(3, 7), sticky="w"
            )
            variable.trace_add("write", self.on_fit_parameters_changed)
        self.fit_button = ttk.Button(fitting, text="Fit / Recompute", command=self.recompute_fit)
        self.fit_button.grid(row=0, column=16, padx=(8, 0), sticky="e")
        self.fit_value_vars = {
            name: tk.StringVar(value="—")
            for name in ("Omega0", "fc", "n", "c", "R²", "ln(f)")
        }
        for column, (name, variable) in enumerate(self.fit_value_vars.items()):
            ttk.Label(fitting, text=f"{name}:").grid(row=1, column=2 * column, sticky="e")
            ttk.Label(fitting, textvariable=variable).grid(
                row=1, column=2 * column + 1, padx=(3, 10), sticky="w"
            )
        self.fit_status_var = tk.StringVar(value="Calculate a spectrum before fitting")
        ttk.Label(fitting, textvariable=self.fit_status_var).grid(
            row=2, column=0, columnspan=17, sticky="w", pady=(4, 0)
        )
        self._update_fit_button()

    def _create_figure(self) -> None:
        self.figure = Figure(figsize=(11, 8), dpi=100)
        self.time_ax, self.raw_ax, self.resampled_ax = self.figure.subplots(3, 1)
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
        self._clear_preview(redraw=True)
        self._update_preview_button()
        if self.context_error:
            self.status_var.set(self.context_error)
        elif self._event_time() is None:
            self.status_var.set("Event has no finite event_time")
        else:
            self.status_var.set("Select a channel and CSV files — preview only, not saved")

    def _event_time(self) -> float | None:
        return self._coerce_event_time(
            self.event.get("event_time") if isinstance(self.event, Mapping) else None
        )

    def _update_preview_button(self) -> None:
        enabled = (
            self.selected_channel_index in self.channel_candidates
            and self._event_time() is not None
            and bool(self.calibration_path_var.get().strip())
            and bool(self.q_path_var.get().strip())
        )
        self.preview_button.configure(state="normal" if enabled else "disabled")

    def _update_fit_button(self) -> None:
        self.fit_button.configure(state="normal" if self.preview_result is not None else "disabled")

    def on_channel_changed(self, _event=None) -> None:
        value = self.channel_combobox.get()
        selected = None
        if value.startswith("strain[") and value.endswith("]"):
            try:
                candidate = int(value[7:-1])
            except ValueError:
                candidate = -1
            if candidate in self.channel_candidates:
                selected = candidate
        self.selected_channel_index = selected
        self._invalidate_preview("Selection changed — recompute preview")

    def on_event_changed(self, _event=None) -> None:
        try:
            self._set_event(int(self.event_combobox.get()))
            self._refresh_event_widgets()
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            self._show_error(exc)

    def on_parameters_changed(self, *_args) -> None:
        if self._initializing_parameters:
            return
        self._invalidate_preview("Parameters changed — recompute preview")

    def on_fit_parameters_changed(self, *_args) -> None:
        if self._initializing_parameters:
            return
        self._clear_fit(redraw=True)
        self.fit_status_var.set("Fitting parameters changed — recompute fit")

    def _invalidate_preview(self, status: str) -> None:
        self._clear_preview(redraw=True)
        self._update_preview_button()
        self.status_var.set(status)

    def _clear_preview(self, *, redraw: bool) -> None:
        self.preview_trace = None
        self.preview_result = None
        self.preview_parameters = None
        self._clear_fit(redraw=False)
        if hasattr(self, "fit_button"):
            self._update_fit_button()
        if hasattr(self, "fit_status_var"):
            self.fit_status_var.set("Calculate a spectrum before fitting")
        if redraw and hasattr(self, "time_ax"):
            for axes in (self.time_ax, self.raw_ax, self.resampled_ax):
                axes.clear()
            self.canvas.draw_idle()

    def _clear_fit(self, *, redraw: bool) -> None:
        self.fit_result = None
        self.fit_parameters = None
        if hasattr(self, "fit_value_vars"):
            for variable in self.fit_value_vars.values():
                variable.set("—")
        if redraw and self.preview_trace is not None and self.preview_result is not None:
            self._draw_preview(self.preview_trace, self.preview_result)
        if hasattr(self, "fit_button"):
            self._update_fit_button()

    def browse_calibration(self) -> None:
        self._browse_path(self.calibration_path_var, "Select calibration CSV")

    def browse_q(self) -> None:
        self._browse_path(self.q_path_var, "Select Q CSV")

    def _browse_path(self, variable, title: str) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title=title,
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            variable.set(selected)
            self._invalidate_preview("Path changed — recompute preview")

    def _build_selected_trace(self) -> BlockTrace:
        if self.selected_channel_index not in self.channel_candidates:
            raise ValueError("Select a valid strain channel")
        if self.strain_time is None or self.strain_raw is None:
            raise ValueError(self.context_error or "Dynamic strain data is unavailable")
        event_time = self._event_time()
        if event_time is None:
            raise ValueError("event_time must be a finite real scalar")
        time = self.strain_time
        signal = self.strain_raw if self.strain_raw.ndim == 1 else self.strain_raw[self.selected_channel_index]
        differences = np.diff(time)
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
            parameters = parse_spectrum_parameters(
                self.pre_sec_var.get(),
                self.post_sec_var.get(),
                self.nfft_var.get(),
                self.calibration_path_var.get(),
                self.q_path_var.get(),
            )
            trace = self._build_selected_trace()
            result = compute_spectrum_at_trigger(trace, **parameters)
            self._clear_fit(redraw=False)
            self.preview_trace = trace
            self.preview_result = result
            self.preview_parameters = parameters
            self._draw_preview(trace, result)
            self._update_fit_button()
            self.fit_status_var.set("Spectrum ready — fit not yet computed")
            valid_count = int(np.count_nonzero(result.valid_cal_mask))
            self.status_var.set(
                f"{trace.sensor}; {trace.sampling_rate:.6g} Hz; NFFT {parameters['nfft']}; "
                f"{Path(parameters['calibration_csv']).name}; {Path(parameters['q_csv']).name}; "
                f"{valid_count} calibrated, {len(result.f7_hz)} resampled — preview only, not saved"
            )
        except _EXPECTED_PREVIEW_EXCEPTIONS as exc:
            self._show_error(exc)

    def recompute_fit(self) -> None:
        if self.preview_result is None:
            self._clear_fit(redraw=False)
            self.fit_status_var.set("Calculate a spectrum before fitting")
            self._update_fit_button()
            return
        try:
            parameters = parse_omega_n_fit_parameters(
                self.lnf_min_var.get(),
                self.lnf_max_var.get(),
                self.omega0_lb_var.get(),
                self.fc_lb_var.get(),
                self.n_lb_var.get(),
                self.omega0_ub_var.get(),
                self.fc_ub_var.get(),
                self.n_ub_var.get(),
            )
            fit_spectrum = scale_spectrum_for_seismology_fit(self.preview_result)
            result = fit_omega_n_q(fit_spectrum, **parameters)
            self.fit_result = result
            self.fit_parameters = parameters
            self._draw_preview(self.preview_trace, self.preview_result, redraw=False)
            self._draw_fit_overlay(self.preview_result, result, parameters)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self._display_fit_result(result)
            self.fit_status_var.set("Omega-n fit preview — not saved")
        except _EXPECTED_FIT_EXCEPTIONS as exc:
            self._clear_fit(redraw=True)
            self.fit_status_var.set(str(exc))
            messagebox.showerror("PZT omega-n fit", str(exc), parent=self)

    @staticmethod
    def _plot_positive(axes, x, y, *args, **kwargs) -> bool:
        x_values = np.asarray(x)
        y_values = np.asarray(y)
        mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
        if not np.any(mask):
            return False
        axes.loglog(x_values[mask], y_values[mask], *args, **kwargs)
        return True

    def _draw_preview(
        self, trace: BlockTrace, result: SpectrumResult, *, redraw: bool = True
    ) -> None:
        for axes in (self.time_ax, self.raw_ax, self.resampled_ax):
            axes.clear()

        window = result.time_window
        relative_ms = (window.full_time - window.peak_time) * 1e3
        signal_x = relative_ms[window.i0 : window.i1 + 1]
        self.time_ax.plot(relative_ms, window.full_voltage_corrected, label="Corrected waveform")
        self.time_ax.plot(signal_x, window.voltage_windowed, label="Windowed waveform")
        self.time_ax.axvspan(signal_x[0], signal_x[-1], color="C2", alpha=0.12, label="Signal window")
        noise_x = relative_ms[window.noise_start : window.noise_end]
        if noise_x.size:
            self.time_ax.axvspan(noise_x[0], noise_x[-1], color="C3", alpha=0.10, label="Noise window")
        self.time_ax.axvline(0.0, color="black", linestyle=":", label="Canonical trigger")
        self.time_ax.set_xlabel("Time relative to trigger (ms)")
        self.time_ax.set_ylabel("Voltage (V)")
        self.time_ax.legend(loc="best")
        self.time_ax.grid(True, alpha=0.3)

        self._plot_positive(self.raw_ax, result.freq_hz, result.amp_raw, label="Signal raw")
        self._plot_positive(self.raw_ax, result.freq_hz, result.amp_noise_raw, label="Noise raw")
        self._plot_positive(
            self.raw_ax,
            result.freq_hz[result.valid_cal_mask],
            result.amp_cal[result.valid_cal_mask],
            label="Signal calibrated",
        )
        self._plot_positive(
            self.raw_ax,
            result.freq_hz[result.valid_noise_cal_mask],
            result.amp_noise_cal[result.valid_noise_cal_mask],
            label="Noise calibrated",
        )
        self.raw_ax.set_xlabel("Frequency (Hz)")
        self.raw_ax.set_ylabel("Spectral amplitude")
        if self.raw_ax.lines:
            self.raw_ax.legend(loc="best")
        self.raw_ax.grid(True, which="both", alpha=0.3)

        self._plot_positive(self.resampled_ax, result.f7_hz, result.y7_cal, label="Calibrated")
        self._plot_positive(self.resampled_ax, result.f7_hz, result.y7_noise_cal, label="Noise")
        self._plot_positive(self.resampled_ax, result.f7_hz, result.y7_qcorr, label="Q-corrected")
        self.resampled_ax.set_xlabel("Frequency (Hz)")
        self.resampled_ax.set_ylabel("Resampled amplitude")
        if self.resampled_ax.lines:
            self.resampled_ax.legend(loc="best")
        self.resampled_ax.grid(True, which="both", alpha=0.3)
        self.figure.tight_layout()
        if redraw:
            self.canvas.draw_idle()

    def _draw_fit_overlay(
        self,
        preview_result: SpectrumResult,
        fit_result: FitResult,
        parameters: Mapping[str, Any],
    ) -> None:
        self._plot_positive(
            self.resampled_ax,
            preview_result.f7_hz[fit_result.mask_fit],
            preview_result.y7_qcorr[fit_result.mask_fit],
            linestyle="none",
            marker="o",
            markersize=3,
            label="Fit samples",
        )
        self._plot_positive(
            self.resampled_ax,
            fit_result.f_model_hz,
            fit_result.m_model_amp,
            linewidth=2,
            label="Omega-n model",
        )
        for value, label in (
            (math.exp(parameters["lnf_min"]), "Fit range"),
            (math.exp(parameters["lnf_max"]), None),
        ):
            if math.isfinite(value) and value > 0:
                self.resampled_ax.axvline(value, color="C4", linestyle=":", label=label)
        if self.resampled_ax.lines:
            self.resampled_ax.legend(loc="best")

    def _display_fit_result(self, result: FitResult) -> None:
        values = {
            "Omega0": result.omega0,
            "fc": result.fc_hz,
            "n": result.n,
            "c": result.c,
            "R²": result.r2,
        }
        for name, value in values.items():
            self.fit_value_vars[name].set(f"{value:.6g}")
        self.fit_value_vars["ln(f)"].set(f"{result.lnf_min:.6g} – {result.lnf_max:.6g}")

    def _show_error(self, exc: Exception) -> None:
        self._clear_preview(redraw=True)
        self.status_var.set(str(exc))
        messagebox.showerror("PZT spectrum", str(exc), parent=self)

    def on_close(self) -> None:
        if self in self.parent.child_windows:
            self.parent.child_windows.remove(self)
        self.destroy()
