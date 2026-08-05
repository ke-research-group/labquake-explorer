import math
from pathlib import Path
import unittest

import numpy as np

from labquake_explorer.analysis import (
    build_fit_curve,
    compute_power_law_r_squared,
    fit_power_law,
)


def tim_fit_power_law(xs, ys):
    valid_pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(valid_pairs) < 2:
        return None
    log_x = [math.log10(x) for x, _ in valid_pairs]
    log_y = [math.log10(y) for _, y in valid_pairs]
    count = len(valid_pairs)
    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xx = sum(x * x for x in log_x)
    sum_xy = sum(x * y for x, y in zip(log_x, log_y))
    denom = count * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None
    exponent = (count * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - exponent * sum_x) / count
    return 10**intercept, exponent


def tim_bac_amp_r_squared(xs, ys, coefficient, exponent):
    valid_pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(valid_pairs) < 2:
        return None
    log_y = [math.log10(y) for _, y in valid_pairs]
    predicted_log_y = [
        math.log10(coefficient * (x**exponent))
        for x, _ in valid_pairs
        if coefficient * (x**exponent) > 0
    ]
    if len(predicted_log_y) != len(log_y):
        return None
    mean_log_y = sum(log_y) / len(log_y)
    ss_tot = sum((value - mean_log_y) ** 2 for value in log_y)
    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(log_y, predicted_log_y)
    )
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def tim_egf_r_squared(xs, ys, coefficient, exponent):
    valid_pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(valid_pairs) < 2:
        return None
    log_y = [math.log10(y) for _, y in valid_pairs]
    predicted_log_y = [
        math.log10(coefficient * (x**exponent)) for x, _ in valid_pairs
    ]
    mean_log_y = sum(log_y) / len(log_y)
    ss_tot = sum((value - mean_log_y) ** 2 for value in log_y)
    ss_res = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(log_y, predicted_log_y)
    )
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def tim_bac_amp_curve(x_min, x_max, coefficient, exponent, points=200):
    if x_min <= 0 or x_max <= 0:
        raise ValueError("Power-law fit requires positive x values.")
    if points < 2:
        points = 2
    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    x_fit = [
        10 ** (log_min + (log_max - log_min) * index / (points - 1))
        for index in range(points)
    ]
    y_fit = [coefficient * (x**exponent) for x in x_fit]
    return x_fit, y_fit


def tim_egf_curve(x_min, x_max, coefficient, exponent, points=200):
    log_min = math.log10(x_min)
    log_max = math.log10(x_max)
    x_fit = [
        10 ** (log_min + (log_max - log_min) * index / (points - 1))
        for index in range(points)
    ]
    y_fit = [coefficient * (x**exponent) for x in x_fit]
    return x_fit, y_fit


class PZTSourceScalingTests(unittest.TestCase):
    def assert_common_parity(self, x, y):
        expected_fit = tim_fit_power_law(x, y)
        actual_fit = fit_power_law(x=x, y=y)
        self.assertEqual(actual_fit, expected_fit)
        self.assertIsNotNone(actual_fit)
        coefficient, exponent = actual_fit
        expected_bac_r2 = tim_bac_amp_r_squared(
            x, y, coefficient, exponent
        )
        expected_egf_r2 = tim_egf_r_squared(x, y, coefficient, exponent)
        self.assertEqual(expected_bac_r2, expected_egf_r2)
        self.assertEqual(
            compute_power_law_r_squared(
                x=x,
                y=y,
                coefficient=coefficient,
                exponent=exponent,
            ),
            expected_bac_r2,
        )
        expected_bac_curve = tim_bac_amp_curve(
            min(value for value in x if value > 0),
            max(value for value in x if value > 0),
            coefficient,
            exponent,
        )
        expected_egf_curve = tim_egf_curve(
            min(value for value in x if value > 0),
            max(value for value in x if value > 0),
            coefficient,
            exponent,
        )
        np.testing.assert_array_equal(expected_bac_curve, expected_egf_curve)
        actual_curve = build_fit_curve(
            x_min=min(value for value in x if value > 0),
            x_max=max(value for value in x if value > 0),
            coefficient=coefficient,
            exponent=exponent,
        )
        np.testing.assert_array_equal(actual_curve, expected_bac_curve)

    def test_1d_bac_parity(self):
        self.assert_common_parity(
            [35.0, 50.0, 75.0, 100.0],
            [1.3e5, 2.2e5, 4.8e5, 7.1e5],
        )

    def test_1d_egf_parity(self):
        self.assert_common_parity(
            [35.0, 50.0, 75.0, 100.0],
            [8.0e4, 1.9e5, 3.6e5, 6.3e5],
        )

    def test_2d_amp_parity(self):
        self.assert_common_parity(
            [100.0, 125.0, 150.0, 200.0],
            [2.0e6, 3.1e6, 5.2e6, 8.7e6],
        )

    def test_2d_egf_parity(self):
        self.assert_common_parity(
            [100.0, 125.0, 150.0, 200.0],
            [1.7e6, 2.9e6, 4.6e6, 7.8e6],
        )

    def test_randomized_parity(self):
        rng = np.random.default_rng(20260805)
        for _ in range(100):
            x = np.sort(10 ** rng.uniform(-5.0, 5.0, 8))
            coefficient = float(10 ** rng.uniform(-6.0, 6.0))
            exponent = float(rng.uniform(-3.0, 3.0))
            y = coefficient * x**exponent * 10 ** rng.normal(0.0, 0.1, 8)
            expected = tim_fit_power_law(x, y)
            actual = fit_power_law(x=x, y=y)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(
                compute_power_law_r_squared(
                    x=x,
                    y=y,
                    coefficient=actual[0],
                    exponent=actual[1],
                ),
                tim_bac_amp_r_squared(x, y, expected[0], expected[1]),
            )

    def test_constant_y_behavior(self):
        x = [1.0, 2.0, 4.0]
        y = [5.0, 5.0, 5.0]
        coefficient, exponent = fit_power_law(x=x, y=y)
        self.assertEqual((coefficient, exponent), tim_fit_power_law(x, y))
        self.assertEqual(
            compute_power_law_r_squared(
                x=x, y=y, coefficient=coefficient, exponent=exponent
            ),
            1.0,
        )
        self.assertEqual(
            compute_power_law_r_squared(
                x=x, y=y, coefficient=coefficient * 2.0, exponent=exponent
            ),
            0.0,
        )

    def test_insufficient_all_negative_and_zero_values(self):
        self.assertIsNone(fit_power_law(x=1.0, y=2.0))
        self.assertIsNone(fit_power_law(x=[-1.0, -2.0], y=[3.0, 4.0]))
        self.assertIsNone(fit_power_law(x=[0.0, 1.0], y=[2.0, 0.0]))

    def test_mixed_values_use_only_positive_pairs(self):
        x = [-5.0, 1.0, 2.0, 4.0, 0.0]
        y = [99.0, 3.0, 12.0, 48.0, 100.0]
        self.assertEqual(fit_power_law(x=x, y=y), tim_fit_power_law(x, y))

    def test_near_zero_positive_and_degenerate_x(self):
        x = [1.0e-300, 2.0e-300, 4.0e-300]
        y = [1.0e-200, 2.0e-200, 8.0e-200]
        self.assertEqual(fit_power_law(x=x, y=y), tim_fit_power_law(x, y))
        self.assertIsNone(fit_power_law(x=[2.0, 2.0], y=[3.0, 4.0]))

    def test_r_squared_nonpositive_prediction_matches_bac_amp(self):
        self.assertIsNone(
            compute_power_law_r_squared(
                x=[1.0, 2.0],
                y=[2.0, 4.0],
                coefficient=-1.0,
                exponent=1.0,
            )
        )

    def test_curve_endpoints_count_and_minimum_points(self):
        x_fit, y_fit = build_fit_curve(
            x_min=1.0,
            x_max=1000.0,
            coefficient=3.0,
            exponent=2.0,
            points=7,
        )
        self.assertEqual(len(x_fit), 7)
        self.assertEqual(x_fit[0], 1.0)
        self.assertEqual(x_fit[-1], 1000.0)
        np.testing.assert_array_equal(y_fit, 3.0 * x_fit**2)
        short_x, short_y = build_fit_curve(
            x_min=1.0,
            x_max=10.0,
            coefficient=2.0,
            exponent=1.0,
            points=1,
        )
        self.assertEqual(len(short_x), 2)
        self.assertEqual(len(short_y), 2)

    def test_inputs_are_not_modified_and_curve_arrays_are_new(self):
        x = np.array([1.0, 2.0, 4.0])
        y = np.array([2.0, 8.0, 32.0])
        original_x = x.copy()
        original_y = y.copy()
        fit_power_law(x=x, y=y)
        np.testing.assert_array_equal(x, original_x)
        np.testing.assert_array_equal(y, original_y)
        first = build_fit_curve(
            x_min=1.0, x_max=4.0, coefficient=2.0, exponent=2.0
        )
        second = build_fit_curve(
            x_min=1.0, x_max=4.0, coefficient=2.0, exponent=2.0
        )
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])

    def test_structural_validation(self):
        invalid_pairs = [
            ({"x": [], "y": []}, "x"),
            ({"x": [1.0], "y": [1.0, 2.0]}, "same length"),
            ({"x": np.array(1.0), "y": [1.0]}, "x"),
            ({"x": [[1.0, 2.0]], "y": [1.0, 2.0]}, "x"),
            ({"x": [True, 2.0], "y": [1.0, 2.0]}, "x"),
            ({"x": [1.0 + 1.0j, 2.0], "y": [1.0, 2.0]}, "x"),
            ({"x": [np.nan, 2.0], "y": [1.0, 2.0]}, "x"),
            ({"x": [1.0, 2.0], "y": [1.0, np.inf]}, "y"),
        ]
        for arguments, message in invalid_pairs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fit_power_law(**arguments)
        with self.assertRaisesRegex(ValueError, "coefficient"):
            compute_power_law_r_squared(
                x=[1.0, 2.0], y=[2.0, 4.0], coefficient=np.nan, exponent=1.0
            )
        with self.assertRaisesRegex(ValueError, "points"):
            build_fit_curve(
                x_min=1.0,
                x_max=2.0,
                coefficient=1.0,
                exponent=1.0,
                points=True,
            )

    def test_module_has_no_forbidden_dependencies(self):
        source = (
            Path(__file__).parents[1]
            / "labquake_explorer"
            / "analysis"
            / "pzt_source_scaling.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "tkinter",
            "DataManager",
            "matplotlib",
            "pandas",
            "TPC5Dataset",
            "h5py",
            "Path(",
            "open(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
