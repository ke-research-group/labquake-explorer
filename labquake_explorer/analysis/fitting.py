"""Straight-line fits with diagnostics.

``linear_fit`` never raises on degenerate input; it returns a ``LinearFit``
with ``valid=False`` and NaN coefficients so callers can display "n/a"
instead of crashing inside a Tk callback.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

METHODS = ("ols", "theilsen")


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r2: float
    stderr: float          # standard error of the slope (NaN for theilsen)
    n: int
    x_range: float
    method: str
    valid: bool
    reason: str = ""

    def __call__(self, x):
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def as_dict(self) -> dict:
        return asdict(self)


def _invalid(method: str, n: int, reason: str, x_range: float = float("nan")) -> LinearFit:
    nan = float("nan")
    return LinearFit(nan, nan, nan, nan, int(n), x_range, method, False, reason)


def linear_fit(x, y, method: str = "ols") -> LinearFit:
    """Fit ``y = slope * x + intercept``.

    ``method`` is ``"ols"`` (ordinary least squares via ``scipy.stats.linregress``)
    or ``"theilsen"`` (median of pairwise slopes, robust to outliers,
    deterministic).  Non-finite pairs are dropped.  Fewer than two distinct x
    values gives an invalid fit.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n < 2:
        return _invalid(method, n, "fewer than two finite samples")
    x_range = float(x.max() - x.min())
    if x_range == 0.0:
        return _invalid(method, n, "all x values are identical", x_range)

    if method == "ols":
        res = stats.linregress(x, y)
        slope, intercept = float(res.slope), float(res.intercept)
        stderr = float(res.stderr) if n > 2 else float("nan")
    else:
        res = stats.theilslopes(y, x)
        slope, intercept = float(res.slope), float(res.intercept)
        stderr = float("nan")

    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return _invalid(method, n, "fit produced non-finite coefficients", x_range)
    return LinearFit(slope, intercept, float(r2), stderr, n, x_range, method, True)
