import warnings

import numpy as np
import pytest
from scipy import stats

from labquake_explorer.analysis.scaling import (
    PowerLawFit, fit_power_law, bootstrap_exponent, reference_line,
)

FIELDS = {"exponent", "log10_coefficient", "coefficient", "exponent_stderr",
          "r2", "exponent_rma", "n", "valid", "reason"}


def test_exact_power_law_recovered():
    x = np.logspace(-3, 3, 25)
    y = 4.0 * x ** (-1.0 / 3.0)
    fit = fit_power_law(x, y)
    assert isinstance(fit, PowerLawFit)
    assert fit.valid and fit.reason == ""
    assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-10)
    assert fit.coefficient == pytest.approx(4.0, rel=1e-9)
    assert fit.log10_coefficient == pytest.approx(np.log10(4.0), abs=1e-10)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.n == 25
    assert fit(8.0) == pytest.approx(4.0 * 8.0 ** (-1.0 / 3.0), rel=1e-9)
    assert fit.exponent_stderr == pytest.approx(0.0, abs=1e-9)
    # RMA slope coincides with OLS on exact data
    assert fit.exponent_rma == pytest.approx(-1.0 / 3.0, abs=1e-10)


def test_exact_fit_stderr_is_finite_zero_and_as_dict_roundtrips():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    fit = fit_power_law(x, 2.0 * x ** 1.5)
    assert np.isfinite(fit.exponent_stderr) and fit.exponent_stderr < 1e-9
    d = fit.as_dict()
    assert d["exponent"] == pytest.approx(1.5) and d["valid"] is True
    assert d["exponent_rma"] == pytest.approx(1.5)
    assert set(d) == FIELDS


def test_noisy_data_exponent_within_three_stderr_and_stderr_r2_pinned():
    rng = np.random.default_rng(3)
    x = np.logspace(0, 4, 200)
    y = 10.0 ** (0.7 + (-1.0 / 3.0) * np.log10(x) + rng.normal(0, 0.1, x.size))
    fit = fit_power_law(x, y)
    assert fit.valid
    assert fit.exponent_stderr > 0
    assert abs(fit.exponent - (-1.0 / 3.0)) < 3 * fit.exponent_stderr
    assert 0.5 < fit.r2 < 1.0
    # pin stderr against the textbook OLS formula with n-2 degrees of freedom
    lx, ly = np.log10(x), np.log10(y)
    n = x.size
    resid = ly - (fit.exponent * lx + fit.log10_coefficient)
    sxx = np.sum((lx - lx.mean()) ** 2)
    expected_se = np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)
    assert fit.exponent_stderr == pytest.approx(expected_se, rel=1e-9)
    # a wrong dof (n-1) or an inflated stderr must not match
    assert fit.exponent_stderr != pytest.approx(np.sqrt(np.sum(resid ** 2) / (n - 1) / sxx), rel=1e-9)
    # r2, slope and intercept pinned to scipy's own linregress
    ref = stats.linregress(lx, ly)
    assert fit.r2 == pytest.approx(ref.rvalue ** 2, rel=1e-12)
    assert fit.exponent == pytest.approx(ref.slope, rel=1e-12)
    assert fit.log10_coefficient == pytest.approx(ref.intercept, rel=1e-12)


def test_rma_slope_identity_and_steeper_than_ols():
    rng = np.random.default_rng(5)
    x = np.logspace(0, 4, 150)
    y = 10.0 ** (0.7 - 0.5 * np.log10(x) + rng.normal(0, 0.2, x.size))
    fit = fit_power_law(x, y)
    lx, ly = np.log10(x), np.log10(y)
    r = stats.linregress(lx, ly).rvalue
    assert r < 0
    # RMA = sign(r) * std(ly)/std(lx) = OLS / |r|
    assert fit.exponent_rma == pytest.approx(np.sign(r) * np.std(ly) / np.std(lx), rel=1e-12)
    assert fit.exponent_rma == pytest.approx(fit.exponent / abs(r), rel=1e-9)
    assert abs(fit.exponent_rma) > abs(fit.exponent)
    assert np.sign(fit.exponent_rma) == np.sign(fit.exponent)


def test_regression_dilution_with_scatter_in_x():
    # y is an exact -1/3 power of the TRUE x; the observed x scatters.
    # OLS of log y on observed log x is attenuated toward 0; RMA is less so.
    rng = np.random.default_rng(7)
    lx_true = rng.uniform(0, 4, 400)
    lx_obs = lx_true + rng.normal(0, 0.5, lx_true.size)
    y = 10.0 ** (1.0 - lx_true / 3.0)
    fit = fit_power_law(10.0 ** lx_obs, y)
    assert fit.valid
    assert abs(fit.exponent) < abs(fit.exponent_rma) < 1.0 / 3.0
    assert abs(fit.exponent) < 1.0 / 3.0 - 5 * fit.exponent_stderr   # clearly attenuated
    # the reverse regression (x on y) gives a different exponent for the same data
    rev = fit_power_law(y, 10.0 ** lx_obs)
    assert abs(1.0 / rev.exponent - fit.exponent) > 0.02


def test_coefficient_overflow_and_underflow_do_not_raise():
    # intercept +400: 10**400 overflows a float; the fit must stay usable
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = fit_power_law([1e-300, 1e-250, 1e-200], [1e100, 1e150, 1e200])
        assert fit.valid and fit.reason == ""
        assert fit.exponent == pytest.approx(1.0, abs=1e-9)
        assert fit.log10_coefficient == pytest.approx(400.0, abs=1e-6)
        assert np.isposinf(fit.coefficient)
        assert fit.exponent_rma == pytest.approx(1.0, abs=1e-9)
        # evaluation falls back to log space and gives finite, correct values
        assert fit(1e-300) == pytest.approx(1e100, rel=1e-6)
        assert fit(1e-200) == pytest.approx(1e200, rel=1e-6)
        assert np.all(np.isfinite(fit(np.array([1e-300, 1e-250, 1e-200]))))
        # intercept -400: underflows to 0.0 coefficient, still usable
        fit = fit_power_law([1e100, 1e150, 1e200], [1e-300, 1e-250, 1e-200])
        assert fit.valid and fit.coefficient == 0.0
        assert fit.log10_coefficient == pytest.approx(-400.0, abs=1e-6)
        assert fit(1e100) == pytest.approx(1e-300, rel=1e-6)
        assert fit.as_dict()["coefficient"] == 0.0


def test_call_matches_direct_evaluation_when_finite():
    fit = fit_power_law(np.logspace(0, 3, 10), 3.0 * np.logspace(0, 3, 10) ** 2.0)
    xs = np.array([0.0, 0.5, 1.0, 10.0, 1e3])
    assert np.allclose(fit(xs), 3.0 * xs ** 2.0, rtol=1e-9)
    assert fit(0.0) == pytest.approx(0.0)
    # zero exponent: constant everywhere, including x = 0
    flat = fit_power_law([1.0, 10.0, 100.0], [0.4, 0.4, 0.4])
    assert flat.valid and flat.exponent == pytest.approx(0.0, abs=1e-12)
    assert flat(0.0) == pytest.approx(0.4) and flat(7.0) == pytest.approx(0.4)
    assert flat.exponent_rma == pytest.approx(0.0, abs=1e-12)


def test_non_positive_and_non_finite_pairs_are_dropped():
    x = np.array([1.0, 10.0, 100.0, -5.0, 1000.0, np.nan, 1e4, 1e5])
    y = np.array([2.0, 20.0, 200.0, 7.0, 2000.0, 1.0, 0.0, np.inf])
    fit = fit_power_law(x, y)
    assert fit.valid and fit.n == 4
    assert fit.exponent == pytest.approx(1.0, abs=1e-10)
    assert fit.coefficient == pytest.approx(2.0, rel=1e-9)


def test_invalid_on_fewer_than_three_pairs():
    fit = fit_power_law([1.0, 10.0], [1.0, 10.0])
    assert not fit.valid and fit.n == 2 and "fewer than 3" in fit.reason
    assert np.isnan(fit.exponent) and np.isnan(fit.coefficient)
    assert np.isnan(fit.exponent_rma) and np.isnan(fit.exponent_stderr)
    assert np.isnan(fit(5.0))
    assert set(fit.as_dict()) == FIELDS
    assert not fit_power_law([], []).valid
    # three pairs but one is non-positive -> only two usable
    fit = fit_power_law([1.0, 10.0, 100.0], [1.0, 10.0, -1.0])
    assert not fit.valid and fit.n == 2


def test_invalid_on_all_non_positive_and_identical_x():
    fit = fit_power_law([-1.0, -2.0, 0.0], [1.0, 2.0, 3.0])
    assert not fit.valid and fit.n == 0
    fit = fit_power_law([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert not fit.valid and "identical" in fit.reason and fit.n == 4


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fit_power_law([1.0, 2.0, 3.0], [1.0, 2.0])


def _bootstrap_data():
    rng = np.random.default_rng(11)
    x = np.logspace(0, 4, 80)
    y = 10.0 ** (1.0 - 0.5 * np.log10(x) + rng.normal(0, 0.15, x.size))
    return x, y


def test_bootstrap_is_deterministic_and_brackets_truth():
    x, y = _bootstrap_data()
    lo, hi = bootstrap_exponent(x, y, n_boot=400, seed=0)
    lo2, hi2 = bootstrap_exponent(x, y, n_boot=400, seed=0)
    assert (lo, hi) == (lo2, hi2)
    assert lo < hi
    fit = fit_power_law(x, y)
    assert lo < fit.exponent < hi
    # 16-84 half-width is close to +-1 stderr of the OLS fit (observed ratio ~1.04);
    # a 95% or 90% interval (ratio ~1.9 / ~1.6) or an IQR (~0.66) would fail
    assert 0.85 < (hi - lo) / 2 / fit.exponent_stderr < 1.25
    lo3, hi3 = bootstrap_exponent(x, y, n_boot=400, seed=1)
    assert (lo3, hi3) != (lo, hi)


def test_bootstrap_matches_explicit_16_84_percentiles_of_resamples():
    x, y = _bootstrap_data()
    n_boot, seed = 300, 0
    rng = np.random.default_rng(seed)
    exponents = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        exponents.append(fit_power_law(x[idx], y[idx]).exponent)
    expected = np.percentile(exponents, [16.0, 84.0])
    lo, hi = bootstrap_exponent(x, y, n_boot=n_boot, seed=seed)
    assert (lo, hi) == (float(expected[0]), float(expected[1]))
    # and it is not any of the other common interval choices
    for pair in ([2.5, 97.5], [5.0, 95.0], [10.0, 90.0], [25.0, 75.0]):
        other = np.percentile(exponents, pair)
        assert (lo, hi) != (float(other[0]), float(other[1]))


def test_bootstrap_exact_data_collapses_and_invalid_gives_nan():
    x = np.logspace(0, 3, 12)
    lo, hi = bootstrap_exponent(x, 3.0 * x ** 2.0, n_boot=50)
    assert lo == pytest.approx(2.0, abs=1e-8) and hi == pytest.approx(2.0, abs=1e-8)
    lo, hi = bootstrap_exponent([1.0, 2.0], [1.0, 2.0])
    assert np.isnan(lo) and np.isnan(hi)
    with pytest.raises(ValueError):
        bootstrap_exponent(x, x, n_boot=0)


def test_reference_line_passes_through_anchor_with_given_slope():
    x = np.logspace(9, 15, 7)
    y = reference_line(x, -1.0 / 3.0, anchor_x=1e12, anchor_y=2e4)
    assert y[3] == pytest.approx(2e4)
    slope = np.diff(np.log10(y)) / np.diff(np.log10(x))
    assert np.allclose(slope, -1.0 / 3.0)
    assert reference_line(1e12, -1.0 / 3.0, 1e12, 2e4) == pytest.approx(2e4)
    # one decade in x -> 10**(-1/3) in y
    assert reference_line(1e13, -1.0 / 3.0, 1e12, 2e4) == pytest.approx(2e4 * 10 ** (-1.0 / 3.0))


def test_reference_line_rejects_bad_anchor():
    for bad in [(0.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf)]:
        with pytest.raises(ValueError):
            reference_line([1.0, 2.0], -1.0 / 3.0, *bad)
    with pytest.raises(ValueError):
        reference_line([1.0], np.nan, 1.0, 1.0)
