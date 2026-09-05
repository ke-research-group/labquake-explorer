import numpy as np
import pytest

from labquake_explorer.analysis.scaling import (
    PowerLawFit, fit_power_law, bootstrap_exponent, reference_line,
)


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


def test_exact_fit_stderr_is_finite_zero_and_as_dict_roundtrips():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    fit = fit_power_law(x, 2.0 * x ** 1.5)
    assert np.isfinite(fit.exponent_stderr) and fit.exponent_stderr < 1e-9
    d = fit.as_dict()
    assert d["exponent"] == pytest.approx(1.5) and d["valid"] is True
    assert set(d) == {"exponent", "log10_coefficient", "coefficient",
                      "exponent_stderr", "r2", "n", "valid", "reason"}


def test_noisy_data_exponent_within_three_stderr():
    rng = np.random.default_rng(3)
    x = np.logspace(0, 4, 200)
    y = 10.0 ** (0.7 + (-1.0 / 3.0) * np.log10(x) + rng.normal(0, 0.1, x.size))
    fit = fit_power_law(x, y)
    assert fit.valid
    assert fit.exponent_stderr > 0
    assert abs(fit.exponent - (-1.0 / 3.0)) < 3 * fit.exponent_stderr
    assert 0.5 < fit.r2 < 1.0


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
    assert np.isnan(fit(5.0))
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


def test_bootstrap_is_deterministic_and_brackets_truth():
    rng = np.random.default_rng(11)
    x = np.logspace(0, 4, 80)
    y = 10.0 ** (1.0 - 0.5 * np.log10(x) + rng.normal(0, 0.15, x.size))
    lo, hi = bootstrap_exponent(x, y, n_boot=400, seed=0)
    lo2, hi2 = bootstrap_exponent(x, y, n_boot=400, seed=0)
    assert (lo, hi) == (lo2, hi2)
    assert lo < hi
    fit = fit_power_law(x, y)
    assert lo < fit.exponent < hi
    # 16-84 interval width is comparable to +-1 stderr of the OLS fit
    assert 0.5 * fit.exponent_stderr < (hi - lo) / 2 < 2.0 * fit.exponent_stderr
    lo3, hi3 = bootstrap_exponent(x, y, n_boot=400, seed=1)
    assert (lo3, hi3) != (lo, hi)


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
