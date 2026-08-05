"""Potter-compatible, schema-neutral event loading-stiffness analysis.

The public helper in this module calculates the signed loading slope of a
processed shear-stress signal against an explicitly selected slip signal in a
pre-trigger relative-time window.  The caller is responsible for signal
identity, calibration, and physical interpretation.  The core has no GUI,
schema, or persistence knowledge and performs no unit conversion; when stress
is supplied in MPa and slip in micrometres, the resulting slope is MPa/um.

Processing and regression order preserve Potter behavior.  RANSAC uses the
default sklearn estimator when importable (including its unfixed random state)
and selects Potter's deterministic fallback only on ``ImportError``.  Cutoffs
outside ``(0, Nyquist)`` mean no filter, and the returned stiffness is signed.
Only the explicit-array and structured-result boundary is Official-specific.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from .event_drop import moving_average


def calculate_event_loading_stiffness(
    *,
    time: Any,
    tau_signal: Any,
    slip_signal: Any,
    event_time: float,
    pre_start: float = -3.0,
    pre_end: float = -0.5,
    window_sec: float = 3.5,
    smooth_w: int = 100,
    highpass_freq: float = 0.0,
    lowpass_freq: float = 0.0,
    use_ransac: bool = False,
) -> dict[str, Any]:
    """Calculate signed event loading stiffness from selected signals.

    Full-run one-dimensional arrays are windowed inclusively around
    ``event_time``.  Stress and slip then receive the same processing order:
    moving average, optional fourth-order Butterworth high-pass, optional
    fourth-order Butterworth low-pass, and event-window baseline subtraction.
    A line ``processed_tau = k * processed_slip + intercept`` is fitted in the
    inclusive pre-trigger window.  ``k`` remains signed.

    Structural and parameter errors raise :class:`ValueError`.  Event-specific
    inability to complete the calculation (too few samples, filtering a window
    that is too short, or failed regression) returns
    ``{"valid": False}``.
    """
    time_array = _coerce_signal_array(time, "time")
    tau_array = _coerce_signal_array(tau_signal, "tau_signal")
    slip_array = _coerce_signal_array(slip_signal, "slip_signal")
    if not (time_array.size == tau_array.size == slip_array.size):
        raise ValueError("time, tau_signal, and slip_signal length mismatch")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("time must be strictly increasing")

    trigger = _coerce_real_scalar(event_time, "event_time")
    fit_start = _coerce_real_scalar(pre_start, "pre_start")
    fit_end = _coerce_real_scalar(pre_end, "pre_end")
    requested_window = _coerce_real_scalar(window_sec, "window_sec")
    highpass = _coerce_real_scalar(highpass_freq, "highpass_freq")
    lowpass = _coerce_real_scalar(lowpass_freq, "lowpass_freq")
    smoothing = _coerce_positive_integer(smooth_w, "smooth_w")
    if not isinstance(use_ransac, (bool, np.bool_)):
        raise ValueError("use_ransac must be a boolean")

    if requested_window <= 0:
        raise ValueError("window_sec must be greater than zero")
    if fit_start >= fit_end:
        raise ValueError("pre_start must be less than pre_end")
    if fit_start >= 0:
        raise ValueError("pre_start must be less than zero")
    if fit_end > 0:
        raise ValueError("pre_end must be at or before zero")
    half_win = max(requested_window, abs(fit_start) + 0.5)
    event_mask = (time_array >= trigger - half_win) & (
        time_array <= trigger + half_win
    )
    if np.count_nonzero(event_mask) < 20:
        return {"valid": False}

    selected_time = time_array[event_mask].copy()
    relative_time = selected_time - trigger
    raw_tau = tau_array[event_mask].copy()
    raw_slip = slip_array[event_mask].copy()

    dt = float(np.median(np.diff(selected_time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("event-window sampling interval must be finite and positive")
    sampling_rate = 1.0 / dt
    try:
        processed_tau = _process_signal(
            raw_tau, smoothing, highpass, lowpass, sampling_rate
        )
        processed_slip = _process_signal(
            raw_slip, smoothing, highpass, lowpass, sampling_rate
        )
    except ValueError:
        return {"valid": False}

    processed_tau = processed_tau - processed_tau[0]
    processed_slip = processed_slip - processed_slip[0]
    fit_mask = (relative_time >= fit_start) & (relative_time <= fit_end)
    if np.count_nonzero(fit_mask) < 6:
        return {"valid": False}

    fit_slip = processed_slip[fit_mask]
    fit_tau = processed_tau[fit_mask]
    if not np.all(np.isfinite(fit_slip)) or not np.all(np.isfinite(fit_tau)):
        return {"valid": False}
    try:
        if bool(use_ransac):
            coefficients = _fit_ransac(fit_slip, fit_tau)
        else:
            coefficients = np.polyfit(fit_slip, fit_tau, 1)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {"valid": False}

    coefficients = np.asarray(coefficients, dtype=float).copy()
    if coefficients.shape != (2,) or not np.all(np.isfinite(coefficients)):
        return {"valid": False}

    return {
        "valid": True,
        "k": float(coefficients[0]),
        "intercept": float(coefficients[1]),
        "coefficients": coefficients,
        "relative_time": relative_time.copy(),
        "raw_tau": raw_tau.copy(),
        "raw_slip": raw_slip.copy(),
        "processed_tau": processed_tau.copy(),
        "processed_slip": processed_slip.copy(),
        "fit_mask": fit_mask.copy(),
    }


def _coerce_signal_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must not be boolean")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real-valued numeric data")
    converted = np.asarray(array, dtype=float).copy()
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _coerce_real_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a finite real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite real scalar")
    return converted


def _coerce_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _process_signal(
    raw: np.ndarray,
    smooth_w: int,
    highpass_freq: float,
    lowpass_freq: float,
    sampling_rate: float,
) -> np.ndarray:
    """Apply Potter processing; nonpositive/out-of-Nyquist cutoffs are off."""
    processed = moving_average(raw, smooth_w)
    nyquist = sampling_rate / 2.0
    if 0 < highpass_freq < nyquist:
        b, a = butter(4, highpass_freq / nyquist, btype="high")
        processed = filtfilt(b, a, processed)
    if 0 < lowpass_freq < nyquist:
        b, a = butter(4, lowpass_freq / nyquist, btype="low")
        processed = filtfilt(b, a, processed)
    return np.asarray(processed, dtype=float).copy()


def _fit_ransac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Use Potter's sklearn-first RANSAC, with its fallback when unavailable.

    The sklearn estimator intentionally uses all library defaults, including
    its unfixed ``random_state``.  Only an unavailable sklearn import selects
    Potter's deterministic NumPy fallback.
    """
    try:
        from sklearn.linear_model import RANSACRegressor
    except ImportError:
        return _fit_ransac_fallback(x, y)

    model = RANSACRegressor()
    model.fit(x.reshape(-1, 1), y)
    return np.asarray(
        [model.estimator_.coef_[0], model.estimator_.intercept_], dtype=float
    )


def _fit_ransac_fallback(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Potter's deterministic NumPy fallback for an unavailable sklearn."""
    sample_count = len(x)
    if sample_count < 5:
        return np.asarray(np.polyfit(x, y, 1), dtype=float)

    try:
        initial = np.polyfit(x, y, 1)
        residuals = np.abs(y - (initial[0] * x + initial[1]))
        threshold = max(1e-6, float(np.median(residuals)) * 1.5)
    except Exception:
        threshold = 0.05

    rng = np.random.default_rng(42)
    best_inlier_count = -1
    best_coefficients = None

    for _ in range(100):
        indices = rng.choice(x.size, 2, replace=False)
        dx = float(x[indices[1]] - x[indices[0]])
        if abs(dx) < 1e-12:
            continue
        slope = float(y[indices[1]] - y[indices[0]]) / dx
        intercept = float(y[indices[0]]) - slope * float(x[indices[0]])
        inliers = np.abs(y - (slope * x + intercept)) < threshold
        inlier_count = int(np.sum(inliers))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            if inlier_count >= 2:
                try:
                    coefficients = np.polyfit(x[inliers], y[inliers], 1)
                except Exception:
                    coefficients = np.array([slope, intercept])
            else:
                coefficients = np.array([slope, intercept])
            best_coefficients = coefficients

    if best_coefficients is not None:
        return np.asarray(best_coefficients, dtype=float)
    return np.asarray(np.polyfit(x, y, 1), dtype=float)
