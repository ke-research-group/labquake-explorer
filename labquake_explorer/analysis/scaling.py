"""Power-law scaling between source parameters (e.g. corner frequency vs moment).

``fit_power_law`` fits ``y = coefficient * x**exponent`` as a straight line in
log10-log10 space using ``scipy.stats.linregress``.  Only pairs where both
values are positive and finite enter the fit (a logarithm is undefined
otherwise).  As elsewhere in this package the fit never raises on degenerate
data: it returns a ``PowerLawFit`` with ``valid=False``, NaN coefficients and
a human-readable ``reason``.

``bootstrap_exponent`` gives a deterministic 16-84 percentile interval of the
exponent by resampling pairs with replacement; ``reference_line`` evaluates a
power law of a prescribed exponent through an anchor point (e.g. the
``fc ~ M0**(-1/3)`` constant-stress-drop line).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

MIN_PAIRS = 3


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float            # slope of log10(y) vs log10(x)
    log10_coefficient: float   # intercept of log10(y) vs log10(x)
    coefficient: float         # 10 ** log10_coefficient
    exponent_stderr: float     # standard error of the exponent
    r2: float                  # coefficient of determination in log10 space
    n: int                     # number of positive finite pairs used
    valid: bool
    reason: str = ""

    def __call__(self, x):
        """Evaluate ``coefficient * x**exponent`` (NaN when invalid)."""
        x = np.asarray(x, dtype=float)
        return self.coefficient * np.power(x, self.exponent)

    def as_dict(self) -> dict:
        return asdict(self)


def _invalid(n: int, reason: str) -> PowerLawFit:
    nan = float("nan")
    return PowerLawFit(nan, nan, nan, nan, nan, int(n), False, reason)


def _positive_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs where both values are finite and > 0 (as arrays)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[keep], y[keep]


def fit_power_law(x, y) -> PowerLawFit:
    """Fit ``y = coefficient * x**exponent`` by OLS of log10(y) on log10(x).

    Pairs with a non-finite or non-positive value are dropped.  Fewer than
    ``MIN_PAIRS`` (3) remaining pairs, or all x values identical, gives an
    invalid fit rather than an exception.
    """
    x, y = _positive_pairs(x, y)
    n = int(x.size)
    if n < MIN_PAIRS:
        return _invalid(n, f"fewer than {MIN_PAIRS} positive finite pairs")
    lx, ly = np.log10(x), np.log10(y)
    if np.ptp(lx) == 0.0:
        return _invalid(n, "all x values are identical")

    res = stats.linregress(lx, ly)
    exponent, intercept = float(res.slope), float(res.intercept)
    stderr = float(res.stderr)
    if not (np.isfinite(exponent) and np.isfinite(intercept)):
        return _invalid(n, "fit produced non-finite coefficients")
    if not np.isfinite(stderr):
        stderr = 0.0   # exact data can give a non-finite stderr from round-off; report 0

    resid = ly - (exponent * lx + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return PowerLawFit(exponent, intercept, float(10.0 ** intercept), stderr,
                       float(r2), n, True)


def bootstrap_exponent(x, y, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """16th and 84th percentiles of the exponent over ``n_boot`` pair resamples.

    Pairs are resampled with replacement (deterministically, from ``seed``)
    and refitted with ``fit_power_law``; resamples that give an invalid fit
    (e.g. all identical x) are skipped.  Returns ``(nan, nan)`` when the full
    data set does not give a valid fit or fewer than half of the resamples
    do.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    x, y = _positive_pairs(x, y)
    if not fit_power_law(x, y).valid:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = x.size
    exponents = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        fit = fit_power_law(x[idx], y[idx])
        if fit.valid:
            exponents.append(fit.exponent)
    if len(exponents) < max(1, n_boot // 2):
        return float("nan"), float("nan")
    lo, hi = np.percentile(exponents, [16.0, 84.0])
    return float(lo), float(hi)


def reference_line(x, exponent: float, anchor_x: float, anchor_y: float) -> np.ndarray:
    """``y = anchor_y * (x / anchor_x)**exponent``: a power law through an anchor.

    Used to draw e.g. the constant-stress-drop ``fc ~ M0**(-1/3)`` line
    through a chosen point of an fc-M0 plot.  ``anchor_x`` and ``anchor_y``
    must be positive and finite.
    """
    if not (np.isfinite(anchor_x) and np.isfinite(anchor_y) and anchor_x > 0 and anchor_y > 0):
        raise ValueError("anchor_x and anchor_y must be positive and finite")
    if not np.isfinite(exponent):
        raise ValueError("exponent must be finite")
    x = np.asarray(x, dtype=float)
    return float(anchor_y) * np.power(x / float(anchor_x), float(exponent))
