"""Power-law scaling between source parameters (e.g. corner frequency vs moment).

``fit_power_law`` fits ``y = coefficient * x**exponent`` as a straight line in
log10-log10 space using ``scipy.stats.linregress``.  Only pairs where both
values are positive and finite enter the fit (a logarithm is undefined
otherwise).  As elsewhere in this package the fit never raises on degenerate
data: it returns a ``PowerLawFit`` with ``valid=False``, NaN coefficients and
a human-readable ``reason``.

Direction of regression.  The reported ``exponent`` is the ordinary
least-squares slope of log10(y) on log10(x): it treats x as error-free and
minimises the vertical (y) scatter only.  When x itself scatters (in an
fc-M0 plot M0 is derived from omega0 with a relative error comparable to that
of fc) the OLS slope is *attenuated* toward zero (regression dilution), and
regressing x on y gives a different line.  Compare against literature values
using the same axis assignment.  ``exponent_rma`` is the reduced-major-axis
(geometric-mean) slope ``sign(r) * std(log10 y) / std(log10 x)``, the usual
errors-in-both-variables estimator for scaling laws; it equals the OLS slope
when r2 = 1 and is always at least as steep (``|rma| = |ols| / |r|``).

``bootstrap_exponent`` gives a deterministic 16-84 percentile interval of the
OLS exponent by resampling pairs with replacement; ``reference_line``
evaluates a power law of a prescribed exponent through an anchor point (e.g.
the ``fc ~ M0**(-1/3)`` constant-stress-drop line).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

MIN_PAIRS = 3


@dataclass(frozen=True)
class PowerLawFit:
    exponent: float            # OLS slope of log10(y) vs log10(x)
    log10_coefficient: float   # intercept of log10(y) vs log10(x)
    coefficient: float         # 10 ** log10_coefficient (inf / 0.0 when it over/underflows a float)
    exponent_stderr: float     # standard error of the OLS exponent
    r2: float                  # coefficient of determination in log10 space
    exponent_rma: float        # reduced-major-axis slope: sign(r) * std(log10 y) / std(log10 x)
    n: int                     # number of positive finite pairs used
    valid: bool
    reason: str = ""

    def __call__(self, x):
        """Evaluate ``coefficient * x**exponent`` (NaN when invalid).

        Evaluated in log space where the direct product is not finite, so a
        fit whose ``coefficient`` overflowed to inf (or underflowed to 0) still
        gives finite values at the x of the data.
        """
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            direct = self.coefficient * np.power(x, self.exponent)
            via_log = np.power(10.0, self.log10_coefficient + self.exponent * np.log10(x))
        return np.where(np.isfinite(direct), direct, via_log)

    def as_dict(self) -> dict:
        return asdict(self)


def _invalid(n: int, reason: str) -> PowerLawFit:
    nan = float("nan")
    return PowerLawFit(nan, nan, nan, nan, nan, nan, int(n), False, reason)


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

    ``exponent`` is the OLS slope of log10(y) on log10(x): x is assumed
    error-free, and the slope is attenuated toward zero when x scatters
    (regression dilution) -- see the module docstring.  ``exponent_rma`` is
    the reduced-major-axis slope for the errors-in-both-variables case.

    Pairs with a non-finite or non-positive value are dropped.  Fewer than
    ``MIN_PAIRS`` (3) remaining pairs, or all x values identical, gives an
    invalid fit rather than an exception.  ``coefficient`` is ``10 **
    log10_coefficient`` evaluated with numpy, so an intercept beyond the
    float range gives inf (or 0.0) instead of an exception; the fit stays
    valid and ``log10_coefficient`` / ``__call__`` remain usable.
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

    # Reduced-major-axis slope; r = 0 (flat y) gives 0 with a positive sign.
    sign = -1.0 if res.rvalue < 0 else 1.0
    rma = sign * float(np.std(ly) / np.std(lx))

    with np.errstate(over="ignore", under="ignore"):
        coefficient = float(np.power(10.0, intercept))
    return PowerLawFit(exponent, intercept, coefficient, stderr, float(r2), rma, n, True)


def bootstrap_exponent(x, y, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """16th and 84th percentiles of the OLS exponent over ``n_boot`` pair resamples.

    Pairs are resampled with replacement (deterministically: index draws are
    ``numpy.random.default_rng(seed).integers(0, n, size=n)`` per resample,
    in order) and refitted with ``fit_power_law``; resamples that give an
    invalid fit (e.g. all identical x) are skipped.  Returns ``(nan, nan)``
    when the full data set does not give a valid fit or fewer than half of
    the resamples do.
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
