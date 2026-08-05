"""Tim-compatible power-law scaling calculations.

The functions preserve the current student-tim BAC/AMP production behavior:
positive-pair filtering followed by an unweighted log10-space regression,
``coefficient = 10**intercept``, log10-space R-squared, and logarithmically
spaced fit-curve sampling.  The BAC/AMP guard behavior is retained.  This
module has no paper-record aggregation, schema traversal, plotting, or
persistence responsibilities.
"""

import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np


def _finite_real_scalar(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _one_dimensional_values(
    name: str,
    values: Real | Sequence[Real] | np.ndarray,
) -> np.ndarray:
    if isinstance(values, (bool, np.bool_)):
        raise ValueError(f"{name} must contain real numeric values")
    if isinstance(values, Real):
        array = np.asarray([values])
    else:
        if isinstance(values, np.ndarray):
            if values.dtype.kind == "b":
                raise ValueError(f"{name} must contain real numeric values")
        elif any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"{name} must contain real numeric values")
        array = np.asarray(values)
        if array.ndim == 0:
            raise ValueError(f"{name} must be one-dimensional")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, (bool, np.bool_)) for value in array.tolist()):
        raise ValueError(f"{name} must contain real numeric values")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    converted = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _paired_values(
    x: Real | Sequence[Real] | np.ndarray,
    y: Real | Sequence[Real] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = _one_dimensional_values("x", x)
    y_values = _one_dimensional_values("y", y)
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same length")
    return x_values, y_values


def fit_power_law(
    *,
    x: Real | Sequence[Real] | np.ndarray,
    y: Real | Sequence[Real] | np.ndarray,
) -> tuple[float, float] | None:
    """Fit ``y = coefficient * x**exponent`` in log10 space.

    Pairs for which either value is not greater than zero are discarded,
    matching Tim's production filtering order.  This is an unweighted linear
    regression of ``log10(y)`` on ``log10(x)``.
    """

    x_values, y_values = _paired_values(x, y)
    valid_pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if x_value > 0 and y_value > 0
    ]
    if len(valid_pairs) < 2:
        return None

    log_x = [math.log10(x_value) for x_value, _ in valid_pairs]
    log_y = [math.log10(y_value) for _, y_value in valid_pairs]
    count = len(valid_pairs)

    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xx = sum(x_value * x_value for x_value in log_x)
    sum_xy = sum(
        x_value * y_value for x_value, y_value in zip(log_x, log_y)
    )
    denominator = count * sum_xx - sum_x * sum_x

    if abs(denominator) < 1e-12:
        return None

    exponent = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - exponent * sum_x) / count
    coefficient = 10**intercept
    return float(coefficient), float(exponent)


def compute_power_law_r_squared(
    *,
    x: Real | Sequence[Real] | np.ndarray,
    y: Real | Sequence[Real] | np.ndarray,
    coefficient: float,
    exponent: float,
) -> float | None:
    """Calculate Tim's coefficient of determination in log10(y) space."""

    x_values, y_values = _paired_values(x, y)
    coefficient = _finite_real_scalar("coefficient", coefficient)
    exponent = _finite_real_scalar("exponent", exponent)
    valid_pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if x_value > 0 and y_value > 0
    ]
    if len(valid_pairs) < 2:
        return None

    log_y = [math.log10(y_value) for _, y_value in valid_pairs]
    predicted_log_y = [
        math.log10(coefficient * (x_value**exponent))
        for x_value, _ in valid_pairs
        if coefficient * (x_value**exponent) > 0
    ]
    if len(predicted_log_y) != len(log_y):
        return None

    mean_log_y = sum(log_y) / len(log_y)
    total_sum_squares = sum(
        (value - mean_log_y) ** 2 for value in log_y
    )
    residual_sum_squares = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(log_y, predicted_log_y)
    )
    if total_sum_squares <= 1e-12:
        return 1.0 if residual_sum_squares <= 1e-12 else 0.0
    return float(1.0 - residual_sum_squares / total_sum_squares)


def build_fit_curve(
    *,
    x_min: float,
    x_max: float,
    coefficient: float,
    exponent: float,
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Tim power-law fit at logarithmically spaced x values."""

    x_min = _finite_real_scalar("x_min", x_min)
    x_max = _finite_real_scalar("x_max", x_max)
    coefficient = _finite_real_scalar("coefficient", coefficient)
    exponent = _finite_real_scalar("exponent", exponent)
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, Integral):
        raise ValueError("points must be an integer")
    points = int(points)

    if x_min <= 0 or x_max <= 0:
        raise ValueError("x_min and x_max must be positive")
    if points < 2:
        points = 2

    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    x_fit = np.asarray(
        [
            10 ** (log_min + (log_max - log_min) * index / (points - 1))
            for index in range(points)
        ],
        dtype=float,
    )
    y_fit = np.asarray(
        [coefficient * (x_value**exponent) for x_value in x_fit],
        dtype=float,
    )
    return x_fit, y_fit
