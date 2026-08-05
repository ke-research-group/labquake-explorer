"""Tim EGF numerical analysis with an explicit in-memory trace boundary.

The numerical implementation is ported from the latest student-tim module and
intentionally preserves its legacy EGF behavior.  Dataset, sensor, and file
lookup are replaced by caller-supplied ``BlockTrace`` and calibration paths.
Legacy APIs keep automatic peak selection; explicit-trigger APIs replace only
that selection and preserve trim, baseline, filter, taper, pre-zero padding,
calibration, log interpolation, ``EGF_SCALE_FIX``, fitting, M0, and Mw behavior.

The caller supplies identity and calibration inputs.  Spectral pairing, ratios,
source-time functions, project infrastructure, and persistence are outside this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, find_peaks

from .pzt_analysis_seismology import (
    DEFAULT_NFFT,
    DEFAULT_POST_SEC,
    DEFAULT_PRE_SEC,
    DEFAULT_THRESHOLD,
    DEFAULT_TRIM_HEAD_SEC,
    FMIN_VALID_HZ,
    POINTS_PER_DECADE,
    BlockTrace,
    blackman_harris_window,
    load_csv_gain,
    tukey_window,
    validate_fit_parameter_bounds,
)


EGF_DEFAULT_LNF_MIN = 6.9
EGF_DEFAULT_LNF_MAX = 11.3
EGF_DEFAULT_FIT_PARAMETER_LB = (1e-15, 1e-9, 1.0)
EGF_DEFAULT_FIT_PARAMETER_UB = (1e3, 3e5, 8.0)
EGF_SCALE_FIX = 1
EGF_SPECTRUM_DISPLAY_MAX_HZ = 1_000_000.0
C_FM = 4628.0

WINDOW_TYPE = "bh"
BH_TERMS = 4
TUKEY_ALPHA = 1.0
BASELINE_MODE = "linear"
HP_FC_HZ = 200.0
USE_EXTRA_FILTER = False
FILTER_MODE = "bandpass"
FILTER_ORDER = 3
FILTER_HP_HZ = 1e3
FILTER_LP_HZ = 5e5
USE_OMEGA_IN_FREQ = False


@dataclass
class EGFTimeWindowResult:
    """Tim-compatible EGF signal/noise window and sampling metadata."""
    x_ms: np.ndarray
    signal_windowed: np.ndarray
    window_scaled_signal: np.ndarray
    window: np.ndarray
    raw_voltage_segment: np.ndarray
    full_time: np.ndarray
    full_voltage_corrected: np.ndarray
    i0: int
    i1: int
    noise_start: int
    noise_end: int
    dt: float
    coherent_gain: float
    peak_time: float
    block: int
    sensor: str


@dataclass
class EGFSpectrumResult:
    """Tim-compatible raw, calibrated, and log-resampled EGF spectra."""
    time_window: EGFTimeWindowResult
    freq_hz: np.ndarray
    amp_raw: np.ndarray
    amp_noise_raw: np.ndarray
    amp_cal: np.ndarray
    amp_noise_cal: np.ndarray
    valid_cal_mask: np.ndarray
    valid_noise_cal_mask: np.ndarray
    f7_hz: np.ndarray
    y7_cal: np.ndarray
    y7_noise_cal: np.ndarray
    calibration_path: Path


@dataclass
class EGFFitResult:
    """Omega-n fit, model curve, M0, and Mw derived by Tim's EGF flow."""
    lnf_min: float
    lnf_max: float
    mask_fit: np.ndarray
    omega0: float
    fc_hz: float
    n: float
    r2: float
    f_model_hz: np.ndarray
    m_model_amp: np.ndarray
    m0: float
    mw: float


def egf_calib_interp_for(
    freq_hz: np.ndarray,
    calibration_csv: Path,
    fmin_valid: float = FMIN_VALID_HZ,
    fmax_valid: float = EGF_SPECTRUM_DISPLAY_MAX_HZ,
) -> np.ndarray:
    """Interpolate Tim's EGF calibration from an explicit caller path."""
    f_csv, gain_csv, _ = load_csv_gain(str(calibration_csv), None)
    cal = np.full_like(freq_hz, np.nan, dtype=float)
    if freq_hz.size == 0:
        return cal
    lo = max(float(np.nanmin(f_csv)), float(fmin_valid))
    hi = min(float(fmax_valid), float(np.nanmax(freq_hz)))
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if np.any(mask):
        cal[mask] = np.interp(
            freq_hz[mask], f_csv, gain_csv, left=gain_csv[0], right=gain_csv[-1]
        )
    return cal


def prezero_pad(values: np.ndarray, target_length: int) -> np.ndarray:
    """Prepend zeros using Tim's EGF FFT padding convention."""
    if target_length > len(values):
        return np.concatenate(
            [np.zeros(target_length - len(values), dtype=values.dtype), values]
        )
    return values


def apply_extra_filter(values: np.ndarray, dt: float) -> np.ndarray:
    """Apply Tim's module-configured optional EGF filter when enabled."""
    if (not USE_EXTRA_FILTER) or values.size < 3:
        return values
    fs = 1.0 / dt
    nyq = 0.5 * fs
    mode = FILTER_MODE.lower()
    if mode in ("bp", "band", "bandpass"):
        low = max(FILTER_HP_HZ / nyq, 1e-6)
        high = min(FILTER_LP_HZ / nyq, 0.999999)
        if not low < high:
            return values
        wn = [low, high]
        btype = "bandpass"
    elif mode in ("hp", "high", "highpass"):
        wn = min(max(FILTER_HP_HZ / nyq, 1e-6), 0.999999)
        btype = "highpass"
    elif mode in ("lp", "low", "lowpass"):
        wn = min(max(FILTER_LP_HZ / nyq, 1e-6), 0.999999)
        btype = "lowpass"
    else:
        return values
    b, a = butter(FILTER_ORDER, wn, btype=btype)
    return filtfilt(b, a, values)


def _finish_egf_time_window(
    trace: BlockTrace,
    t_full: np.ndarray,
    v_full: np.ndarray,
    peak_index: int,
    pre_sec: float,
    post_sec: float,
) -> EGFTimeWindowResult:
    """Apply Tim's window processing after the peak sample is known."""
    t_peak = float(t_full[peak_index])
    t_start = t_peak - pre_sec
    t_end = t_peak + post_sec

    mask_sig = (t_full >= t_start) & (t_full <= t_end)
    idx_sig = np.where(mask_sig)[0]
    if len(idx_sig) < 2:
        raise ValueError("EGF signal window is too short")
    i0 = int(idx_sig[0])
    i1 = int(idx_sig[-1])
    win_len = i1 - i0 + 1

    if i0 - win_len >= 0:
        noise_start, noise_end = i0 - win_len, i0
    elif i1 + 1 + win_len <= len(t_full):
        noise_start, noise_end = i1 + 1, i1 + 1 + win_len
    else:
        take = min(win_len, len(t_full))
        noise_start = max(0, i0 - take)
        noise_end = noise_start + take

    dt_full = float(np.median(np.diff(t_full)))
    fs_full = 1.0 / dt_full
    if BASELINE_MODE == "linear" and noise_end - noise_start >= 2:
        p = np.polyfit(t_full[noise_start:noise_end], v_full[noise_start:noise_end], 1)
        v_full = v_full - np.polyval(p, t_full)
    elif BASELINE_MODE == "dc" and noise_end > noise_start:
        v_full = v_full - np.median(v_full[noise_start:noise_end])
    elif BASELINE_MODE == "highpass":
        wn = min(max(HP_FC_HZ / (0.5 * fs_full), 1e-6), 0.999999)
        b, a = butter(2, wn, btype="highpass")
        v_full = filtfilt(b, a, v_full)

    v_signal = v_full[i0 : i1 + 1]
    v_noise_raw = v_full[noise_start:noise_end]
    v_signal = apply_extra_filter(v_signal, dt_full)
    v_noise_raw = apply_extra_filter(v_noise_raw, dt_full)
    if len(v_noise_raw) < win_len:
        v_noise = np.concatenate(
            [v_noise_raw, np.zeros(win_len - len(v_noise_raw), dtype=v_noise_raw.dtype)]
        )
    else:
        v_noise = v_noise_raw[:win_len]

    if WINDOW_TYPE.lower() == "bh":
        window = blackman_harris_window(win_len, terms=BH_TERMS)
    elif WINDOW_TYPE.lower() == "tukey":
        window = tukey_window(win_len, alpha=TUKEY_ALPHA)
    else:
        raise ValueError("WINDOW_TYPE must be 'bh' or 'tukey'")

    coherent_gain = float(window.mean())
    signal_windowed = v_signal * window
    noise_windowed = v_noise * window
    x_ms = (t_full[i0 : i1 + 1] - t_full[i0]) * 1e3
    scale = np.max(np.abs(v_signal)) if np.max(np.abs(v_signal)) > 0 else 1.0
    result = EGFTimeWindowResult(
        x_ms=x_ms,
        signal_windowed=signal_windowed,
        window_scaled_signal=window * scale,
        window=window,
        raw_voltage_segment=v_full[i0 : i1 + 1],
        full_time=t_full,
        full_voltage_corrected=v_full,
        i0=i0,
        i1=i1,
        noise_start=int(noise_start),
        noise_end=int(noise_end),
        dt=dt_full,
        coherent_gain=coherent_gain,
        peak_time=t_peak,
        block=trace.block,
        sensor=trace.sensor,
    )
    # Tim's spectrum stage consumes this runtime-only compatibility attribute.
    result._noise_windowed = noise_windowed  # type: ignore[attr-defined]
    return result


def _prepare_egf_time_window(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
    trim_head_sec: float = DEFAULT_TRIM_HEAD_SEC,
) -> EGFTimeWindowResult:
    """Prepare Tim's EGF window after caller-side dataset and sensor selection."""
    t_full = np.asarray(trace.time, dtype=float)
    v_full = np.asarray(trace.voltage, dtype=float)
    t_full = t_full - t_full[0]

    if trim_head_sec > 0.0:
        mask = t_full >= trim_head_sec
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"{trace.sensor} has too little data after trim_head_sec={trim_head_sec}"
            )
        t_full = t_full[mask] - trim_head_sec
        v_full = v_full[mask]

    peaks, _ = find_peaks(np.abs(v_full), height=threshold)
    peak_index = int(peaks[0]) if len(peaks) else int(np.nanargmax(np.abs(v_full)))
    return _finish_egf_time_window(trace, t_full, v_full, peak_index, pre_sec, post_sec)


def _prepare_egf_time_window_at_trigger(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    trim_head_sec: float = DEFAULT_TRIM_HEAD_SEC,
) -> EGFTimeWindowResult:
    """Prepare Tim-compatible processing around an absolute canonical trigger."""
    trigger = trace.trigger_time
    if isinstance(trigger, (bool, np.bool_, np.ndarray)) or not isinstance(trigger, Real):
        raise ValueError("trigger_time must be a finite real scalar")
    trigger = float(trigger)
    if not np.isfinite(trigger):
        raise ValueError("trigger_time must be a finite real scalar")

    t_original = np.asarray(trace.time, dtype=float)
    v_full = np.asarray(trace.voltage, dtype=float)
    if trigger < float(t_original[0]) or trigger > float(t_original[-1]):
        raise ValueError("trigger_time is outside the original trace range")
    trigger_relative_original = trigger - float(t_original[0])
    t_full = t_original - t_original[0]

    if trim_head_sec > 0.0:
        if trigger_relative_original < trim_head_sec:
            raise ValueError("trigger_time lies in the trimmed-away trace region")
        mask = t_full >= trim_head_sec
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"{trace.sensor} has too little data after trim_head_sec={trim_head_sec}"
            )
        t_full = t_full[mask] - trim_head_sec
        v_full = v_full[mask]
        # Convert the absolute trigger into the retained, trimmed coordinates.
        trigger_relative = trigger_relative_original - trim_head_sec
    else:
        trigger_relative = trigger_relative_original

    if trigger_relative < float(t_full[0]) or trigger_relative > float(t_full[-1]):
        raise ValueError("trigger_time is outside the retained trace range")
    peak_index = int(np.argmin(np.abs(t_full - trigger_relative)))
    return _finish_egf_time_window(trace, t_full, v_full, peak_index, pre_sec, post_sec)


def compute_egf_time_window(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
) -> EGFTimeWindowResult:
    """Compute Tim's legacy first-threshold-peak EGF window.

    Inputs use seconds through ``BlockTrace`` and explicit window parameters;
    the result contains processed arrays, indices, and sampling metadata.  This
    API performs no schema traversal, plotting, or persistence.
    """
    return _prepare_egf_time_window(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
        threshold=threshold,
    )


def compute_egf_time_window_at_trigger(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
) -> EGFTimeWindowResult:
    """Compute an EGF window at an absolute ``BlockTrace.trigger_time``.

    The trigger is converted relative to the trace start and adjusted for
    ``DEFAULT_TRIM_HEAD_SEC``.  No automatic peak detection or fallback is used;
    all remaining processing retains Tim-compatible EGF behavior.
    """
    return _prepare_egf_time_window_at_trigger(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
    )


def compute_egf_spectrum(
    trace: BlockTrace,
    calibration_csv: Path,
    nfft: int = DEFAULT_NFFT,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
) -> EGFSpectrumResult:
    """Compute Tim's legacy auto-peak spectrum from explicit trace and CSV.

    The structured result contains raw, calibrated, and resampled spectrum
    arrays.  Signal discovery, plotting, pairing, and persistence are excluded.
    """
    window_result = _prepare_egf_time_window(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
        threshold=threshold,
    )
    return _compute_egf_spectrum_from_window(window_result, calibration_csv, nfft)


def compute_egf_spectrum_at_trigger(
    trace: BlockTrace,
    calibration_csv: Path,
    nfft: int = DEFAULT_NFFT,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
) -> EGFSpectrumResult:
    """Compute a Tim-compatible EGF spectrum at the explicit canonical trigger.

    Window selection uses ``BlockTrace.trigger_time`` without automatic peak
    detection or fallback.  FFT, calibration, and resampling remain unchanged.
    """
    window_result = _prepare_egf_time_window_at_trigger(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
    )
    return _compute_egf_spectrum_from_window(window_result, calibration_csv, nfft)


def _compute_egf_spectrum_from_window(
    window_result: EGFTimeWindowResult,
    calibration_csv: Path,
    nfft: int,
) -> EGFSpectrumResult:
    """Apply Tim's spectrum body to an already-selected EGF window."""
    win_len = len(window_result.signal_windowed)
    n_use = int(nfft) if nfft and nfft > win_len else win_len

    v_sig = prezero_pad(window_result.signal_windowed, n_use)
    v_noise = prezero_pad(window_result._noise_windowed, n_use)  # type: ignore[attr-defined]
    v_sig_fft = np.fft.rfft(v_sig, n=n_use)
    v_noise_fft = np.fft.rfft(v_noise, n=n_use)
    freq = np.fft.rfftfreq(n_use, window_result.dt)

    amp_raw = np.abs(v_sig_fft) * window_result.dt
    amp_noise_raw = np.abs(v_noise_fft) * window_result.dt
    if USE_OMEGA_IN_FREQ:
        omega = 2.0 * np.pi * freq
        amp_raw = amp_raw * omega
        amp_noise_raw = amp_noise_raw * omega

    calibration = egf_calib_interp_for(freq, calibration_csv, fmin_valid=FMIN_VALID_HZ)
    amp_cal = np.divide(
        amp_raw,
        calibration,
        out=np.full_like(amp_raw, np.nan),
        where=np.isfinite(calibration),
    )
    amp_noise_cal = np.divide(
        amp_noise_raw,
        calibration,
        out=np.full_like(amp_noise_raw, np.nan),
        where=np.isfinite(calibration),
    )

    valid_cal = np.isfinite(amp_cal) & (amp_cal > 0) & (freq >= FMIN_VALID_HZ)
    valid_noise = (
        np.isfinite(amp_noise_cal) & (amp_noise_cal > 0) & (freq >= FMIN_VALID_HZ)
    )
    f_valid = freq[valid_cal]
    y_valid = amp_cal[valid_cal]
    noise_valid = amp_noise_cal[valid_cal]
    if f_valid.size < 2:
        raise ValueError("Not enough EGF calibrated spectrum points for resampling")

    f_lo = max(FMIN_VALID_HZ, float(np.nanmin(f_valid)))
    f_hi = float(np.nanmax(f_valid))
    decades = np.log10(f_hi) - np.log10(f_lo)
    n_points = int(np.ceil(decades * POINTS_PER_DECADE)) + 1
    f7 = np.logspace(np.log10(f_lo), np.log10(f_hi), n_points)

    pos = (y_valid > 0) & (noise_valid > 0)
    if np.count_nonzero(pos) < 2:
        raise ValueError("Not enough positive EGF signal/noise spectrum points")
    f_src = f_valid[pos]
    y_src = y_valid[pos]
    noise_src = noise_valid[pos]
    y7 = 10.0 ** np.interp(np.log10(f7), np.log10(f_src), np.log10(y_src))
    y7_noise = 10.0 ** np.interp(
        np.log10(f7), np.log10(f_src), np.log10(noise_src)
    )

    return EGFSpectrumResult(
        time_window=window_result,
        freq_hz=freq,
        amp_raw=amp_raw,
        amp_noise_raw=amp_noise_raw,
        amp_cal=amp_cal,
        amp_noise_cal=amp_noise_cal,
        valid_cal_mask=valid_cal,
        valid_noise_cal_mask=valid_noise,
        f7_hz=f7,
        y7_cal=y7,
        y7_noise_cal=y7_noise,
        calibration_path=Path(calibration_csv),
    )


def model_ln_amp(params: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Evaluate Tim's EGF omega-n model in natural-log coordinates."""
    omega0, fc, n = params
    freq = np.exp(x_values)
    fc = max(float(fc), 1e-12)
    denom = 1.0 + (freq / fc) ** n
    denom = np.maximum(denom, np.finfo(float).tiny)
    return np.log(abs(float(omega0))) - np.log(denom)


def fit_egf_omega_n(
    spectrum: EGFSpectrumResult,
    lnf_min: float = EGF_DEFAULT_LNF_MIN,
    lnf_max: float = EGF_DEFAULT_LNF_MAX,
    parameter_lb=None,
    parameter_ub=None,
) -> EGFFitResult:
    """Fit Tim's EGF omega-n model to an existing calibrated spectrum.

    Fit limits are natural-log frequencies.  ``EGF_SCALE_FIX`` and Tim's M0/Mw
    conversion remain part of the numerical contract; no spectrum recompute,
    plotting, schema traversal, or persistence occurs.
    """
    f7 = np.asarray(spectrum.f7_hz, dtype=float)
    y7 = np.asarray(spectrum.y7_cal, dtype=float) * EGF_SCALE_FIX
    y7 = np.maximum(y7, np.finfo(float).tiny)
    x7 = np.log(f7)
    y7_ln = np.log(y7)
    mask_finite = np.isfinite(x7) & np.isfinite(y7_ln)
    mask_fit = mask_finite & (x7 >= float(lnf_min)) & (x7 <= float(lnf_max))
    if np.count_nonzero(mask_fit) < 10:
        raise ValueError(
            f"EGF fit needs at least 10 points in ln(f) range {lnf_min:.3g}-{lnf_max:.3g}"
        )

    x_fit = x7[mask_fit]
    y_fit = y7_ln[mask_fit]
    fc0 = float(np.exp(np.mean(x_fit)))
    omega0_0 = float(np.exp(np.median(y_fit)))
    n0 = 2.0
    if parameter_lb is None:
        parameter_lb = EGF_DEFAULT_FIT_PARAMETER_LB
    if parameter_ub is None:
        parameter_ub = EGF_DEFAULT_FIT_PARAMETER_UB
    lb, ub = validate_fit_parameter_bounds(parameter_lb, parameter_ub)
    p0 = np.array([omega0_0, max(1e-9, fc0), n0], dtype=float)
    p0 = np.minimum(np.maximum(p0, lb), ub)

    res = least_squares(
        lambda params: y_fit - model_ln_amp(params, x_fit),
        p0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
    )
    if not res.success:
        p0b = np.array([omega0_0, min(1e4, ub[1]), n0], dtype=float)
        p0b = np.minimum(np.maximum(p0b, lb), ub)
        res = least_squares(
            lambda params: y_fit - model_ln_amp(params, x_fit),
            p0b,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=10000,
        )
    if not res.success:
        raise RuntimeError(f"EGF ω⁻ⁿ fitting failed: {res.message}")

    omega0, fc, n = res.x
    y_hat = model_ln_amp(res.x, x_fit)
    ss_res = float(np.sum((y_fit - y_hat) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    f_all = np.asarray(spectrum.freq_hz[1:], dtype=float)
    f_all = f_all[np.isfinite(f_all) & (f_all > 0)]
    if f_all.size:
        f_model_min = float(f_all.min())
        f_model_max = float(f_all.max())
    else:
        f_model_min, f_model_max = 1e2, 4e5
    f_fit = f7[mask_fit]
    if f_fit.size:
        f_model_max = min(f_model_max, float(np.nanmax(f_fit)))
    f_model = np.logspace(np.log10(f_model_min), np.log10(f_model_max), 400)
    m_model = np.exp(model_ln_amp(res.x, np.log(f_model)))

    amp0 = float(abs(float(omega0)))
    m0 = float(amp0 * C_FM)
    mw = float((2.0 / 3.0) * np.log10(max(m0, 1e-50)) - 6.067)

    return EGFFitResult(
        lnf_min=float(lnf_min),
        lnf_max=float(lnf_max),
        mask_fit=mask_fit,
        omega0=float(omega0),
        fc_hz=float(fc),
        n=float(n),
        r2=float(r2),
        f_model_hz=f_model,
        m_model_amp=m_model,
        m0=m0,
        mw=mw,
    )


def calc_mw_from_amp(value: float) -> float:
    """Convert an explicit amplitude to Mw using Tim's fixed EGF constant."""
    return (2.0 / 3.0) * np.log10(max(float(value), 1e-50) * C_FM) - 6.067
