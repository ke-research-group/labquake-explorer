"""Pure signal-processing helpers used by event-drop analysis.

This module intentionally contains no event traversal, GUI, persistence, or
Labquake Explorer data-schema logic.  In particular, ``calculate_trend_drop``
returns a signed delta; deciding whether a higher-level result is a magnitude
belongs to the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_PRE_WINDOW = (-1.0, -0.5)
DEFAULT_POST_WINDOW = (0.5, 1.0)


def moving_average(
    values: Any,
    window_size: int,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Return an edge-padded moving average with the input length preserved.

    The padding and convolution match Potter's implementation.  NaN values
    are not ignored and therefore propagate through windows that contain them.
    """
    if isinstance(window_size, (bool, np.bool_)) or not isinstance(
        window_size, (int, np.integer)
    ):
        raise ValueError("window_size must be a positive integer")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    array = np.ravel(np.asarray(values, dtype=float))
    if window_size == 1:
        return array.copy()
    if array.size == 0:
        raise ValueError("cannot smooth an empty input with window_size > 1")

    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad
    padded = np.pad(array, (left_pad, right_pad), mode=pad_mode)
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(padded, kernel, mode="valid")


def compute_half_win(config: Mapping[str, Sequence[float]] | None = None) -> float:
    """Compute Potter's rounded extraction half-window.

    The total span is rounded up to a multiple of three using the original
    formula.  ``config`` may provide ``pre_win`` and ``post_win`` pairs.
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    pre_window = _coerce_window(config.get("pre_win", DEFAULT_PRE_WINDOW), "pre_win")
    post_window = _coerce_window(
        config.get("post_win", DEFAULT_POST_WINDOW), "post_win"
    )
    base = max(abs(pre_window[0]), abs(post_window[1]))
    return math.ceil(2 * base / 3) * 1.5


def calculate_2pt_trend_drop(
    t_rel: Any,
    y: Any,
    points: Sequence[float],
) -> dict[str, Any]:
    """Compatibility wrapper for Potter's historical function name.

    Despite its name, Potter redirected this function to the two-window
    least-squares calculation.  That behavior is retained here.
    """
    return calculate_trend_drop(t_rel, y, points)


def calculate_trend_drop(
    t_rel: Any,
    y: Any,
    points_or_pre_window: Sequence[float],
    post_window: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit pre/post linear trends and return their signed jump at ``t=0``.

    The windows may be supplied as one four-value sequence
    ``(pre_start, pre_end, post_start, post_end)`` or as separate two-value
    pre and post sequences.  Window endpoints are inclusive and may be given
    in either order.

    The returned ``delta`` is ``val_pre_0 - val_post_0``.  No absolute value
    is applied.  If either window has fewer than two finite samples at distinct
    times, the function returns ``{"valid": False}``.
    """
    time = _coerce_1d_float_array(t_rel, "t_rel")
    signal = _coerce_1d_float_array(y, "y")
    if time.size != signal.size:
        raise ValueError("t_rel and y must have the same length")

    pre_window, normalized_post_window = _coerce_trend_windows(
        points_or_pre_window, post_window
    )
    if time.size == 0:
        return {"valid": False}

    pre_fit = _fit_window(time, signal, pre_window)
    post_fit = _fit_window(time, signal, normalized_post_window)
    if pre_fit is None or post_fit is None:
        return {"valid": False}

    val_pre_0 = float(np.polyval(pre_fit, 0.0))
    val_post_0 = float(np.polyval(post_fit, 0.0))
    if not np.isfinite(val_pre_0) or not np.isfinite(val_post_0):
        return {"valid": False}

    return {
        "valid": True,
        "coeff_pre": pre_fit,
        "coeff_post": post_fit,
        "delta": val_pre_0 - val_post_0,
        "val_pre_0": val_pre_0,
        "val_post_0": val_post_0,
    }


def _coerce_1d_float_array(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _coerce_window(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two numeric endpoints")
    try:
        if len(values) != 2:
            raise ValueError
        start, end = float(values[0]), float(values[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"{name} must contain exactly two numeric endpoints"
        ) from exc
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{name} endpoints must be finite")
    return start, end


def _coerce_trend_windows(
    points_or_pre_window: Sequence[float],
    post_window: Sequence[float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if post_window is None:
        if isinstance(points_or_pre_window, (str, bytes)):
            raise ValueError("points must contain four numeric endpoints")
        try:
            if len(points_or_pre_window) != 4:
                raise ValueError
            pre_values = points_or_pre_window[:2]
            post_values = points_or_pre_window[2:]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("points must contain four numeric endpoints") from exc
    else:
        pre_values = points_or_pre_window
        post_values = post_window

    pre_start, pre_end = _coerce_window(pre_values, "pre_window")
    post_start, post_end = _coerce_window(post_values, "post_window")
    return (
        (min(pre_start, pre_end), max(pre_start, pre_end)),
        (min(post_start, post_end), max(post_start, post_end)),
    )


def _fit_window(
    time: np.ndarray,
    signal: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray | None:
    start, end = window
    mask = (
        (time >= start)
        & (time <= end)
        & np.isfinite(time)
        & np.isfinite(signal)
    )
    selected_time = time[mask]
    selected_signal = signal[mask]
    if selected_time.size < 2 or np.unique(selected_time).size < 2:
        return None
    try:
        coefficients = np.polyfit(selected_time, selected_signal, 1)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coefficients)):
        return None
    return coefficients
