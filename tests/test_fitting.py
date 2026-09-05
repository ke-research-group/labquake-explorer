import numpy as np
import pytest

from labquake_explorer.analysis.fitting import linear_fit


def test_ols_recovers_line_exactly():
    x = np.linspace(0, 10, 101)
    y = 2.5 * x - 1.0
    fit = linear_fit(x, y)
    assert fit.valid
    assert fit.slope == pytest.approx(2.5)
    assert fit.intercept == pytest.approx(-1.0)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.n == 101
    assert fit.x_range == pytest.approx(10.0)
    assert fit(4.0) == pytest.approx(9.0)


def test_ols_noise_stderr_and_r2():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 500)
    y = 3.0 * x + rng.normal(0, 0.1, x.size)
    fit = linear_fit(x, y)
    assert fit.slope == pytest.approx(3.0, abs=5 * fit.stderr)
    assert 0.9 < fit.r2 < 1.0
    assert fit.stderr > 0


def test_theilsen_ignores_outliers():
    x = np.linspace(0, 10, 101)
    y = 2.0 * x + 1.0
    y[-10:] += 50.0  # 10% gross outliers at one end (biases OLS)
    ols = linear_fit(x, y, "ols")
    ts = linear_fit(x, y, "theilsen")
    assert ts.valid and ts.method == "theilsen"
    assert ts.slope == pytest.approx(2.0, abs=1e-6)
    assert abs(ols.slope - 2.0) > 0.1
    assert np.isnan(ts.stderr)


def test_degenerate_inputs_are_invalid_not_exceptions():
    assert not linear_fit([], []).valid
    assert not linear_fit([1.0], [2.0]).valid
    fit = linear_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert not fit.valid and "identical" in fit.reason
    assert np.isnan(fit.slope) and np.isnan(fit(0.0))
    fit = linear_fit([0.0, np.nan, 1.0], [0.0, 5.0, 1.0])
    assert fit.valid and fit.n == 2 and fit.slope == pytest.approx(1.0)


def test_two_points_have_nan_stderr_but_valid():
    fit = linear_fit([0.0, 1.0], [0.0, 2.0])
    assert fit.valid and fit.slope == 2.0 and np.isnan(fit.stderr)


def test_bad_method_and_shape():
    with pytest.raises(ValueError):
        linear_fit([0, 1], [0, 1], method="ransac")
    with pytest.raises(ValueError):
        linear_fit([0, 1, 2], [0, 1])
