"""Tim BAC/PZT numerical analysis operating on in-memory ``BlockTrace`` data.

The numerical implementation was migrated from the latest student-tim
``pzt_analysis_seismology.py`` and intentionally preserves its processing
order and legacy numerical assumptions.  Dataset and file discovery belong to
the caller.  Legacy APIs retain Tim's automatic peak selection; canonical-
trigger APIs replace only the peak source with ``BlockTrace.trigger_time`` and
do not fall back.  Subsequent baseline, filter, taper, FFT, calibration, Q
correction, and fitting steps are shared with the Tim flow.

The Official caller owns waveform and channel identity.  This module does not
know about DataManager, schemas, persistence, or TPC5 files.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, find_peaks


TIME_SCALE = 1.0
SCALE_FACTOR = 1.0
SEISMOLOGY_FIT_SPECTRUM_SCALE = 1

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

DEFAULT_PRE_SEC = 0.1
DEFAULT_POST_SEC = 0.1
DEFAULT_THRESHOLD = 0.05
DEFAULT_NFFT = 10000
DEFAULT_TRIM_HEAD_SEC = 0.0
FMIN_VALID_HZ = 100.0
POINTS_PER_DECADE = 1000

USE_Q_ATTEN_CORR = True
Q_FREQ_IS_KHZ = True
Q_CORR_FMIN_KHZ = 1.0
Q_CORR_FMAX_KHZ = 450.0
Q_CORR_TRAVEL_TIME_S = 5.5e-5

DEFAULT_LNF_MIN_Q = 7.3
DEFAULT_LNF_MAX_Q = 10.5
DEFAULT_FIT_PARAMETER_LB = (1e-15, 1e-9, 1.0)
DEFAULT_FIT_PARAMETER_UB = (1e-4, 3e5, 8.0)


@dataclass
class BlockTrace:
    """Caller-resolved in-memory waveform and explicit trigger metadata."""
    time: np.ndarray
    voltage: np.ndarray
    sampling_rate: float
    trigger_time: float
    trigger_sample: int
    block: int
    sensor: str


@dataclass
class TimeWindowResult:
    """Tim-compatible processed signal/noise window and sampling metadata."""
    x_ms: np.ndarray
    voltage_windowed: np.ndarray
    displacement_windowed: np.ndarray
    window_scaled_voltage: np.ndarray
    window_scaled_displacement: np.ndarray
    window: np.ndarray
    raw_voltage_segment: np.ndarray
    displacement_segment: np.ndarray
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
class SpectrumResult:
    """Tim-compatible raw, calibrated, resampled, and Q-corrected spectra."""
    time_window: TimeWindowResult
    freq_hz: np.ndarray
    amp_raw: np.ndarray
    amp_noise_raw: np.ndarray
    amp_voltage_raw: np.ndarray
    amp_noise_voltage_raw: np.ndarray
    amp_cal: np.ndarray
    amp_noise_cal: np.ndarray
    valid_cal_mask: np.ndarray
    valid_noise_cal_mask: np.ndarray
    f7_hz: np.ndarray
    y7_cal: np.ndarray
    y7_noise_cal: np.ndarray
    y7_qcorr: np.ndarray
    calibration_path: Path
    q_path: Path


@dataclass
class FitResult:
    """Omega-n fit parameters, log-space fit mask, model curve, and R-squared."""
    lnf_min: float
    lnf_max: float
    mask_fit: np.ndarray
    omega0: float
    fc_hz: float
    n: float
    c: float
    r2: float
    f_model_hz: np.ndarray
    m_model_amp: np.ndarray


def scale_spectrum_for_seismology_fit(
    spectrum: SpectrumResult,
    scale: float = SEISMOLOGY_FIT_SPECTRUM_SCALE,
) -> SpectrumResult:
    """Return a spectrum with Tim fit-amplitude fields scaled by ``scale``."""
    scale = float(scale)
    if scale == 1.0:
        return spectrum
    return replace(
        spectrum,
        amp_raw=spectrum.amp_raw * scale,
        amp_noise_raw=spectrum.amp_noise_raw * scale,
        amp_cal=spectrum.amp_cal * scale,
        amp_noise_cal=spectrum.amp_noise_cal * scale,
        y7_cal=spectrum.y7_cal * scale,
        y7_noise_cal=spectrum.y7_noise_cal * scale,
        y7_qcorr=spectrum.y7_qcorr * scale,
    )


def general_cosine(length: int, coeffs: Iterable[float]) -> np.ndarray:
    """Construct Tim's general-cosine window for explicit coefficients."""
    if length <= 0:
        return np.array([], dtype=float)
    n = np.arange(length, dtype=float)
    window = np.zeros(length, dtype=float)
    for k, coeff in enumerate(coeffs):
        if k == 0:
            window += coeff
        else:
            window += coeff * np.cos(2.0 * np.pi * k * n / (length - 1))
    return window


def blackman_harris_window(length: int, terms: int = 4) -> np.ndarray:
    """Return Tim's 3-, 4-, or 7-term Blackman-Harris window."""
    if terms == 3:
        coeffs = np.array([0.42, -0.50, 0.08], dtype=float)
    elif terms == 4:
        coeffs = np.array([0.35875, -0.48829, 0.14128, -0.01168], dtype=float)
    elif terms == 7:
        coeffs = np.array(
            [
                0.2712203606,
                -0.4334446123,
                0.21800412,
                -0.0657853433,
                0.0107618673,
                -0.000770012,
                0.00001368088,
            ],
            dtype=float,
        )
    else:
        raise ValueError("BH terms must be 3, 4, or 7.")
    return general_cosine(length, coeffs)


def tukey_window(length: int, alpha: float = 0.5) -> np.ndarray:
    """Return the Tukey window used by Tim's waveform preparation."""
    if length <= 0:
        return np.array([], dtype=float)
    if alpha <= 0:
        return np.ones(length, dtype=float)
    if alpha >= 1:
        n = np.arange(length, dtype=float)
        return 0.5 * (1.0 - np.cos(2 * np.pi * n / (length - 1)))
    n = np.arange(length, dtype=float)
    window = np.ones(length, dtype=float)
    edge = alpha * (length - 1) / 2.0
    left = n < edge
    window[left] = 0.5 * (1 + np.cos(np.pi * (2 * n[left] / (alpha * (length - 1)) - 1)))
    right = n > (length - 1) * (1 - alpha / 2.0)
    window[right] = 0.5 * (
        1 + np.cos(np.pi * (2 * n[right] / (alpha * (length - 1)) - 2 / alpha + 1))
    )
    return window


def postzero_pad(values: np.ndarray, target_length: int) -> np.ndarray:
    """Append zeros to Tim's FFT input without truncating longer input."""
    if target_length > len(values):
        return np.concatenate([values, np.zeros(target_length - len(values), dtype=values.dtype)])
    return values


def apply_extra_filter(values: np.ndarray, dt: float) -> np.ndarray:
    """Apply Tim's module-configured optional filter, or return input unchanged."""
    if (not USE_EXTRA_FILTER) or values.size < 3:
        return values

    fs = 1.0 / dt
    nyq = 0.5 * fs
    mode = FILTER_MODE.lower()

    if mode in ("bp", "band", "bandpass"):
        if FILTER_HP_HZ is None or FILTER_LP_HZ is None:
            raise ValueError("bandpass requires FILTER_HP_HZ and FILTER_LP_HZ")
        low = max(FILTER_HP_HZ / nyq, 1e-6)
        high = min(FILTER_LP_HZ / nyq, 0.999999)
        if not low < high:
            raise ValueError("Invalid bandpass cutoff range")
        wn = [low, high]
        btype = "bandpass"
    elif mode in ("hp", "high", "highpass"):
        if FILTER_HP_HZ is None:
            raise ValueError("highpass requires FILTER_HP_HZ")
        wn = min(max(FILTER_HP_HZ / nyq, 1e-6), 0.999999)
        btype = "highpass"
    elif mode in ("lp", "low", "lowpass"):
        if FILTER_LP_HZ is None:
            raise ValueError("lowpass requires FILTER_LP_HZ")
        wn = min(max(FILTER_LP_HZ / nyq, 1e-6), 0.999999)
        btype = "lowpass"
    else:
        return values

    b, a = butter(FILTER_ORDER, wn, btype=btype)
    return filtfilt(b, a, values)


def _auto_is_db(values: np.ndarray, amp_col_name: str = "") -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    spread = np.nanmax(finite) - np.nanmin(finite)
    looks_db = np.nanmedian(finite) < 0 and spread > 20
    if amp_col_name:
        looks_db = looks_db or ("db" in amp_col_name.lower())
    return bool(looks_db)


@lru_cache(maxsize=8)
def load_csv_gain(csv_path: str, csv_is_db: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load and normalize Tim calibration gain columns from an explicit CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {path}")

    df = pd.read_csv(path)
    cols_lower = {col: str(col).lower() for col in df.columns}
    freq_candidates = [
        col for col in df.columns if any(key in cols_lower[col] for key in ("f", "freq", "frequency", "hz"))
    ]
    if not freq_candidates:
        raise ValueError("Calibration CSV has no frequency column")

    freq_col = None
    for col in freq_candidates:
        if df[col].dtype.kind in "fc" and df[col].notna().sum() > 0:
            freq_col = col
            break
    if freq_col is None:
        freq_col = freq_candidates[0]

    amp_candidates = [
        col
        for col in df.columns
        if col != freq_col
        and any(key in cols_lower[col] for key in ("mag", "amp", "gain", "response", "ratio", "calib"))
    ]
    if amp_candidates:
        amp_col = amp_candidates[0]
    else:
        others = [col for col in df.columns if col != freq_col]
        if not others:
            raise ValueError("Calibration CSV has no amplitude column")
        amp_col = others[-1]

    f_raw = np.asarray(df[freq_col], dtype=float)
    g_raw = np.asarray(df[amp_col], dtype=float)
    ok = np.isfinite(f_raw) & np.isfinite(g_raw) & (f_raw > 0)
    f = f_raw[ok]
    g = g_raw[ok]
    order = np.argsort(f)
    f = f[order]
    g = g[order]

    is_db = _auto_is_db(g, str(amp_col)) if csv_is_db is None else bool(csv_is_db)
    gain = 10.0 ** (g / 20.0) if is_db else g.astype(float)
    gain = np.clip(gain, np.finfo(float).tiny, None)
    return f, gain, str(amp_col)


def calib_interp_for(
    freq_hz: np.ndarray,
    calibration_csv: Path,
    fmin_valid: float = FMIN_VALID_HZ,
) -> np.ndarray:
    """Interpolate calibration gain within Tim's valid frequency interval."""
    f_csv, gain_csv, _ = load_csv_gain(str(calibration_csv), None)
    cal = np.full_like(freq_hz, np.nan, dtype=float)
    if freq_hz.size == 0:
        return cal
    lo = max(float(np.nanmin(f_csv)), float(fmin_valid))
    hi = float(np.nanmax(f_csv))
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if np.any(mask):
        cal[mask] = np.interp(freq_hz[mask], f_csv, gain_csv)
    return cal


@lru_cache(maxsize=4)
def load_q_data(q_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load positive frequency and inverse-Q arrays from an explicit CSV."""
    path = Path(q_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Q CSV not found: {path}")
    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)

    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError("Q CSV requires at least two columns: frequency and Qp_inv")
    fq = data[:, 0] * 1e3 if Q_FREQ_IS_KHZ else data[:, 0]
    qinv = data[:, 1]
    ok = np.isfinite(fq) & np.isfinite(qinv) & (fq > 0) & (qinv > 0)
    if np.count_nonzero(ok) < 2:
        raise ValueError("Q CSV does not contain enough valid points")
    fq = fq[ok]
    qinv = qinv[ok]
    order = np.argsort(fq)
    return fq[order], qinv[order]


def apply_q_correction(
    freq_hz: np.ndarray,
    amp: np.ndarray,
    q_csv_path: Path,
) -> np.ndarray:
    """Apply Tim's configured log-frequency Q attenuation correction."""
    corrected = np.asarray(amp, dtype=float).copy()
    if not USE_Q_ATTEN_CORR:
        return corrected

    fq, qinv = load_q_data(str(q_csv_path))
    q_min = Q_CORR_FMIN_KHZ * 1e3
    q_max = Q_CORR_FMAX_KHZ * 1e3
    mask = (
        np.isfinite(corrected)
        & np.isfinite(freq_hz)
        & (freq_hz > 0)
        & (freq_hz >= q_min)
        & (freq_hz <= q_max)
    )
    if not np.any(mask):
        return corrected

    log_qinv = np.log10(qinv)
    log_interp = np.interp(np.log10(freq_hz[mask]), np.log10(fq), log_qinv)
    qinv_interp = 10.0 ** log_interp
    factor = np.exp(np.pi * freq_hz[mask] * float(Q_CORR_TRAVEL_TIME_S) * qinv_interp)
    corrected[mask] *= factor
    return corrected


def _prepare_time_window(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
    trim_head_sec: float = DEFAULT_TRIM_HEAD_SEC,
) -> TimeWindowResult:
    t_full = np.asarray(trace.time, dtype=float)
    v_full = np.asarray(trace.voltage, dtype=float)
    t_full = t_full - t_full[0]

    if trim_head_sec > 0.0:
        mask = t_full >= trim_head_sec
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"{trace.sensor} has too little data after trim_head_sec={trim_head_sec}")
        t_full = t_full[mask] - trim_head_sec
        v_full = v_full[mask]

    peaks, _ = find_peaks(np.abs(v_full), height=threshold)
    peak_index = int(peaks[0]) if len(peaks) else int(np.nanargmax(np.abs(v_full)))
    t_peak = float(t_full[peak_index])
    t_start = t_peak - pre_sec
    t_end = t_peak + post_sec

    mask_sig = (t_full >= t_start) & (t_full <= t_end)
    idx_sig = np.where(mask_sig)[0]
    if len(idx_sig) < 2:
        raise ValueError("Signal window is too short")
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

    v_sig_voltage = apply_extra_filter(v_full[i0 : i1 + 1], dt_full)
    v_noise_voltage_raw = apply_extra_filter(v_full[noise_start:noise_end], dt_full)
    v_sig_disp = v_sig_voltage.copy()
    v_noise_disp_raw = v_noise_voltage_raw.copy()
    if len(v_noise_disp_raw) < win_len:
        v_noise = np.concatenate(
            [v_noise_disp_raw, np.zeros(win_len - len(v_noise_disp_raw), dtype=v_noise_disp_raw.dtype)]
        )
        v_noise_voltage = np.concatenate(
            [
                v_noise_voltage_raw,
                np.zeros(win_len - len(v_noise_voltage_raw), dtype=v_noise_voltage_raw.dtype),
            ]
        )
    else:
        v_noise = v_noise_disp_raw[:win_len]
        v_noise_voltage = v_noise_voltage_raw[:win_len]

    if WINDOW_TYPE.lower() == "bh":
        window = blackman_harris_window(win_len, terms=BH_TERMS)
    elif WINDOW_TYPE.lower() == "tukey":
        window = tukey_window(win_len, alpha=TUKEY_ALPHA)
    else:
        raise ValueError("WINDOW_TYPE must be 'bh' or 'tukey'")

    coherent_gain = float(window.mean())
    voltage_windowed = v_sig_voltage * window
    displacement_windowed = v_sig_disp * window
    x_ms = (t_full[i0 : i1 + 1] - t_full[i0]) * 1e3
    v_scale = np.max(np.abs(v_sig_voltage)) if np.max(np.abs(v_sig_voltage)) > 0 else 1.0
    d_scale = np.max(np.abs(v_sig_disp)) if np.max(np.abs(v_sig_disp)) > 0 else 1.0

    result = TimeWindowResult(
        x_ms=x_ms,
        voltage_windowed=voltage_windowed,
        displacement_windowed=displacement_windowed,
        window_scaled_voltage=window * v_scale,
        window_scaled_displacement=window * d_scale,
        window=window,
        raw_voltage_segment=v_sig_voltage,
        displacement_segment=v_sig_disp,
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
    # Tim's spectrum stage consumes these runtime-only compatibility arrays.
    result._noise_displacement = v_noise * window  # type: ignore[attr-defined]
    result._noise_voltage = v_noise_voltage * window  # type: ignore[attr-defined]
    return result


def _prepare_time_window_with_peak(
    trace: BlockTrace,
    peak_time_rel: float,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
    trim_head_sec: float = DEFAULT_TRIM_HEAD_SEC,
) -> TimeWindowResult:
    """Tim's explicit-peak preparation path, preserved for numerical parity."""
    t_full = np.asarray(trace.time, dtype=float)
    v_full = np.asarray(trace.voltage, dtype=float)
    t_full = t_full - t_full[0]

    if trim_head_sec > 0.0:
        mask = t_full >= trim_head_sec
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"{trace.sensor} has too little data after trim_head_sec={trim_head_sec}")
        t_full = t_full[mask] - trim_head_sec
        v_full = v_full[mask]
        peak_time_rel = float(peak_time_rel) - trim_head_sec

    if not np.isfinite(peak_time_rel) or peak_time_rel < t_full[0] or peak_time_rel > t_full[-1]:
        return _prepare_time_window(
            trace,
            pre_sec=pre_sec,
            post_sec=post_sec,
            threshold=threshold,
            trim_head_sec=trim_head_sec,
        )

    peak_index = int(np.nanargmin(np.abs(t_full - float(peak_time_rel))))
    t_peak = float(t_full[peak_index])
    t_start = t_peak - pre_sec
    t_end = t_peak + post_sec

    mask_sig = (t_full >= t_start) & (t_full <= t_end)
    idx_sig = np.where(mask_sig)[0]
    if len(idx_sig) < 2:
        raise ValueError("Signal window is too short")
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

    v_sig_voltage = apply_extra_filter(v_full[i0 : i1 + 1], dt_full)
    v_noise_voltage_raw = apply_extra_filter(v_full[noise_start:noise_end], dt_full)
    v_sig_disp = v_sig_voltage.copy()
    v_noise_disp_raw = v_noise_voltage_raw.copy()
    if len(v_noise_disp_raw) < win_len:
        v_noise = np.concatenate(
            [v_noise_disp_raw, np.zeros(win_len - len(v_noise_disp_raw), dtype=v_noise_disp_raw.dtype)]
        )
        v_noise_voltage = np.concatenate(
            [
                v_noise_voltage_raw,
                np.zeros(win_len - len(v_noise_voltage_raw), dtype=v_noise_voltage_raw.dtype),
            ]
        )
    else:
        v_noise = v_noise_disp_raw[:win_len]
        v_noise_voltage = v_noise_voltage_raw[:win_len]

    if WINDOW_TYPE.lower() == "bh":
        window = blackman_harris_window(win_len, terms=BH_TERMS)
    elif WINDOW_TYPE.lower() == "tukey":
        window = tukey_window(win_len, alpha=TUKEY_ALPHA)
    else:
        raise ValueError("WINDOW_TYPE must be 'bh' or 'tukey'")

    coherent_gain = float(window.mean())
    voltage_windowed = v_sig_voltage * window
    displacement_windowed = v_sig_disp * window
    x_ms = (t_full[i0 : i1 + 1] - t_full[i0]) * 1e3
    v_scale = np.max(np.abs(v_sig_voltage)) if np.max(np.abs(v_sig_voltage)) > 0 else 1.0
    d_scale = np.max(np.abs(v_sig_disp)) if np.max(np.abs(v_sig_disp)) > 0 else 1.0

    result = TimeWindowResult(
        x_ms=x_ms,
        voltage_windowed=voltage_windowed,
        displacement_windowed=displacement_windowed,
        window_scaled_voltage=window * v_scale,
        window_scaled_displacement=window * d_scale,
        window=window,
        raw_voltage_segment=v_sig_voltage,
        displacement_segment=v_sig_disp,
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
    # Tim's spectrum stage consumes these runtime-only compatibility arrays.
    result._noise_displacement = v_noise * window  # type: ignore[attr-defined]
    result._noise_voltage = v_noise_voltage * window  # type: ignore[attr-defined]
    return result


def compute_time_window(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
) -> TimeWindowResult:
    """Prepare Tim's auto-peak window after caller-side trace resolution.

    The first threshold peak is used, with Tim's legacy fallback when none is
    found.  Inputs use seconds and the result contains millisecond plot time,
    processed arrays, indices, and sampling metadata.  No schema or persistence
    operation occurs here.
    """
    return _prepare_time_window(trace, pre_sec=pre_sec, post_sec=post_sec, threshold=threshold)


def compute_time_window_at_trigger(
    trace: BlockTrace,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
) -> TimeWindowResult:
    """Prepare a window at the caller-specified trigger, without peak fallback.

    ``trace.trigger_time`` must use the same time coordinates as ``trace.time``.
    The nearest sample is the explicit peak used by Tim's preparation flow.
    """
    time = np.asarray(trace.time, dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("trace.time must be a non-empty 1-D array")
    if not np.all(np.isfinite(time)) or (time.size > 1 and not np.all(np.diff(time) > 0)):
        raise ValueError("trace.time must be finite and strictly increasing")
    if isinstance(trace.trigger_time, (bool, np.bool_, np.ndarray)):
        raise ValueError("trace.trigger_time must be finite")
    try:
        trigger_time = float(trace.trigger_time)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("trace.trigger_time must be finite") from exc
    if not np.isfinite(trigger_time):
        raise ValueError("trace.trigger_time must be finite")
    if trigger_time < time[0] or trigger_time > time[-1]:
        raise ValueError("trace.trigger_time must lie within trace.time")
    return _prepare_time_window_with_peak(
        trace,
        peak_time_rel=trigger_time - float(time[0]),
        pre_sec=pre_sec,
        post_sec=post_sec,
    )


def compute_spectrum(
    trace: BlockTrace,
    calibration_csv: Path,
    q_csv: Path,
    nfft: int = DEFAULT_NFFT,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
    threshold: float = DEFAULT_THRESHOLD,
) -> SpectrumResult:
    """Compute Tim's auto-peak BAC spectrum from explicit trace and CSV paths.

    The result contains raw, calibrated, resampled, and Q-corrected arrays; file
    discovery, plotting, schema traversal, and persistence remain caller work.
    """
    window_result = compute_time_window(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
        threshold=threshold,
    )
    return _compute_spectrum_from_window(
        window_result,
        calibration_csv,
        q_csv,
        nfft,
    )


def compute_spectrum_at_trigger(
    trace: BlockTrace,
    calibration_csv: Path,
    q_csv: Path,
    nfft: int = DEFAULT_NFFT,
    pre_sec: float = DEFAULT_PRE_SEC,
    post_sec: float = DEFAULT_POST_SEC,
) -> SpectrumResult:
    """Compute a Tim-compatible spectrum using ``BlockTrace.trigger_time``.

    Peak auto-detection is not performed; all processing after time-window
    selection is identical to :func:`compute_spectrum`.
    """
    window_result = compute_time_window_at_trigger(
        trace,
        pre_sec=pre_sec,
        post_sec=post_sec,
    )
    return _compute_spectrum_from_window(
        window_result,
        calibration_csv,
        q_csv,
        nfft,
    )


def _compute_spectrum_from_window(
    window_result: TimeWindowResult,
    calibration_csv: Path,
    q_csv: Path,
    nfft: int,
) -> SpectrumResult:
    """Compute the Tim BAC spectrum from a prepared time window."""
    win_len = len(window_result.displacement_windowed)
    n_use = int(nfft) if nfft and nfft > win_len else win_len

    v_sig = postzero_pad(window_result.displacement_windowed, n_use)
    v_noise = postzero_pad(window_result._noise_displacement, n_use)  # type: ignore[attr-defined]
    voltage_sig = postzero_pad(window_result.voltage_windowed, n_use)
    voltage_noise = postzero_pad(window_result._noise_voltage, n_use)  # type: ignore[attr-defined]

    v_sig_fft = np.fft.rfft(v_sig, n=n_use)
    v_noise_fft = np.fft.rfft(v_noise, n=n_use)
    voltage_sig_fft = np.fft.rfft(voltage_sig, n=n_use)
    voltage_noise_fft = np.fft.rfft(voltage_noise, n=n_use)
    freq = np.fft.rfftfreq(n_use, window_result.dt)

    # Preserve Tim's asymmetric coherent-gain correction (noise branches only).
    amp_raw = np.abs(v_sig_fft) * window_result.dt
    amp_noise_raw = (np.abs(v_noise_fft) * window_result.dt) / max(window_result.coherent_gain, np.finfo(float).tiny)
    amp_voltage_raw = np.abs(voltage_sig_fft) * window_result.dt
    amp_noise_voltage_raw = (
        np.abs(voltage_noise_fft) * window_result.dt
    ) / max(window_result.coherent_gain, np.finfo(float).tiny)

    if USE_OMEGA_IN_FREQ:
        omega = 2.0 * np.pi * freq
        amp_raw = amp_raw * omega
        amp_noise_raw = amp_noise_raw * omega
        amp_voltage_raw = amp_voltage_raw * omega
        amp_noise_voltage_raw = amp_noise_voltage_raw * omega

    calibration = calib_interp_for(freq, calibration_csv, fmin_valid=FMIN_VALID_HZ)
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
    valid_noise = np.isfinite(amp_noise_cal) & (amp_noise_cal > 0) & (freq >= FMIN_VALID_HZ)

    f_valid = freq[valid_cal]
    y_valid = amp_cal[valid_cal]
    noise_valid = amp_noise_cal[valid_cal]
    if f_valid.size < 2:
        raise ValueError("Not enough calibrated spectrum points for resampling")

    f_lo = max(FMIN_VALID_HZ, float(np.nanmin(f_valid)))
    f_hi = float(np.nanmax(f_valid))
    decades = np.log10(f_hi) - np.log10(f_lo)
    n_points = int(np.ceil(decades * POINTS_PER_DECADE)) + 1
    f7 = np.logspace(np.log10(f_lo), np.log10(f_hi), n_points)

    pos = (y_valid > 0) & (noise_valid > 0)
    if np.count_nonzero(pos) < 2:
        raise ValueError("Not enough positive signal/noise spectrum points for log resampling")

    f_src = f_valid[pos]
    y_src = y_valid[pos]
    noise_src = noise_valid[pos]
    y7 = 10.0 ** np.interp(np.log10(f7), np.log10(f_src), np.log10(y_src))
    y7_noise = 10.0 ** np.interp(np.log10(f7), np.log10(f_src), np.log10(noise_src))
    y7_qcorr = apply_q_correction(f7, y7, q_csv)

    return SpectrumResult(
        time_window=window_result,
        freq_hz=freq,
        amp_raw=amp_raw,
        amp_noise_raw=amp_noise_raw,
        amp_voltage_raw=amp_voltage_raw,
        amp_noise_voltage_raw=amp_noise_voltage_raw,
        amp_cal=amp_cal,
        amp_noise_cal=amp_noise_cal,
        valid_cal_mask=valid_cal,
        valid_noise_cal_mask=valid_noise,
        f7_hz=f7,
        y7_cal=y7,
        y7_noise_cal=y7_noise,
        y7_qcorr=y7_qcorr,
        calibration_path=Path(calibration_csv),
        q_path=Path(q_csv),
    )


def model_ln_amp(params: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Evaluate Tim's omega-n amplitude model in natural-log coordinates."""
    omega0, fc, n = params
    freq = np.exp(x_values)
    fc = max(float(fc), 1e-12)
    denom = 1.0 + (freq / fc) ** n
    denom = np.maximum(denom, np.finfo(float).tiny)
    return np.log(abs(float(omega0))) - np.log(denom)


def validate_fit_parameter_bounds(
    parameter_lb=None,
    parameter_ub=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated positive ``(omega0, fc, n)`` lower and upper bounds."""
    lb = np.asarray(
        DEFAULT_FIT_PARAMETER_LB if parameter_lb is None else parameter_lb,
        dtype=float,
    )
    ub = np.asarray(
        DEFAULT_FIT_PARAMETER_UB if parameter_ub is None else parameter_ub,
        dtype=float,
    )
    if lb.shape != (3,) or ub.shape != (3,):
        raise ValueError("parameter bounds must contain Omega0, fc, and n")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("parameter bounds must be finite")
    if np.any(lb <= 0) or np.any(lb >= ub):
        raise ValueError("each parameter LB must be positive and smaller than its UB")
    return lb, ub


def _initial_guess(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    fc0 = float(np.exp(np.mean(x_values)))
    omega0_0 = float(np.exp(np.percentile(y_values, 95)))
    return np.array([omega0_0, max(1e-9, fc0), 2.0], dtype=float)


def fit_omega_n_q(
    spectrum: SpectrumResult,
    lnf_min: float = DEFAULT_LNF_MIN_Q,
    lnf_max: float = DEFAULT_LNF_MAX_Q,
    parameter_lb=None,
    parameter_ub=None,
) -> FitResult:
    """Fit Tim's omega-n model to the existing Q-corrected spectrum.

    ``lnf_min`` and ``lnf_max`` bound natural-log frequency.  The returned
    parameters, model curve, mask, and log-space R-squared are computed without
    reloading data, plotting, schema traversal, or persistence.
    """
    eps = np.finfo(float).tiny
    f7 = np.asarray(spectrum.f7_hz, dtype=float)
    y = np.maximum(np.asarray(spectrum.y7_qcorr, dtype=float), eps)
    x7 = np.log(np.maximum(f7, eps))
    y_ln = np.log(y)

    valid = np.isfinite(x7) & np.isfinite(y_ln) & (f7 > 0)
    mask_fit = valid & (x7 >= float(lnf_min)) & (x7 <= float(lnf_max))
    if np.count_nonzero(mask_fit) < 10:
        raise ValueError(f"Q fit needs at least 10 points in ln(f) range {lnf_min:.3g}-{lnf_max:.3g}")

    x_fit = x7[mask_fit]
    y_fit = y_ln[mask_fit]
    lb, ub = validate_fit_parameter_bounds(parameter_lb, parameter_ub)

    p0 = np.minimum(np.maximum(_initial_guess(x_fit, y_fit), lb), ub)
    res = least_squares(
        lambda params: y_fit - model_ln_amp(params, x_fit),
        p0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=10000,
        x_scale="jac",
    )
    if not res.success:
        p0b = np.array([p0[0], 10000.0, 2.0], dtype=float)
        p0b = np.minimum(np.maximum(p0b, lb), ub)
        res = least_squares(
            lambda params: y_fit - model_ln_amp(params, x_fit),
            p0b,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=10000,
            x_scale="jac",
        )

    omega0, fc, n = res.x
    y_hat = model_ln_amp(res.x, x_fit)
    ss_res = float(np.sum((y_fit - y_hat) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    positive_freq = np.asarray(spectrum.freq_hz[1:], dtype=float)
    positive_freq = positive_freq[np.isfinite(positive_freq) & (positive_freq > 0)]
    if positive_freq.size:
        f_model_min = float(np.nanmin(positive_freq))
        f_model_max = float(np.nanmax(positive_freq))
    else:
        f_model_min, f_model_max = 1e2, 4e5

    max_fit_freq = float(np.exp(np.max(x_fit)))
    f_model_max = min(f_model_max, max_fit_freq)
    f_model = np.logspace(np.log10(f_model_min), np.log10(f_model_max), 400)
    m_model = np.exp(model_ln_amp(res.x, np.log(f_model)))

    return FitResult(
        lnf_min=float(lnf_min),
        lnf_max=float(lnf_max),
        mask_fit=mask_fit,
        omega0=float(omega0),
        fc_hz=float(fc),
        n=float(n),
        c=0.0,
        r2=float(r2),
        f_model_hz=f_model,
        m_model_amp=m_model,
    )
