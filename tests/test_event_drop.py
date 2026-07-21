"""Deterministic tests for the pure event-drop calculation helpers."""

import unittest

import numpy as np

from labquake_explorer.analysis.event_drop import (
    calculate_2pt_trend_drop,
    calculate_trend_drop,
    compute_half_win,
    moving_average,
)


class CalculateTrendDropTests(unittest.TestCase):
    def test_normal_drop_preserves_positive_direction(self):
        time = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        signal = np.where(time < 0, 2.0 * time + 10.0, -time + 6.0)

        result = calculate_trend_drop(time, signal, (-2.0, -0.5, 0.5, 2.0))

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["val_pre_0"], 10.0)
        self.assertAlmostEqual(result["val_post_0"], 6.0)
        self.assertAlmostEqual(result["delta"], 4.0)

    def test_reverse_change_preserves_negative_direction(self):
        time = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        signal = np.where(time < 0, time + 3.0, -2.0 * time + 8.0)

        result = calculate_trend_drop(time, signal, (-2.0, -0.5), (0.5, 2.0))

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["val_pre_0"], 3.0)
        self.assertAlmostEqual(result["val_post_0"], 8.0)
        self.assertAlmostEqual(result["delta"], -5.0)

    def test_window_forms_and_reversed_endpoints_are_equivalent(self):
        time = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        signal = np.where(time < 0, 0.5 * time + 7.0, 1.5 * time + 2.0)

        four_points = calculate_trend_drop(time, signal, (-1.0, -3.0, 3.0, 1.0))
        two_windows = calculate_trend_drop(time, signal, (-3.0, -1.0), (1.0, 3.0))

        self.assertTrue(four_points["valid"])
        self.assertAlmostEqual(four_points["delta"], two_windows["delta"])
        np.testing.assert_allclose(four_points["coeff_pre"], two_windows["coeff_pre"])
        np.testing.assert_allclose(four_points["coeff_post"], two_windows["coeff_post"])

    def test_window_endpoints_are_inclusive(self):
        time = np.array([-1.0, -0.5, 0.5, 1.0])
        signal = np.array([4.0, 5.0, 1.0, 2.0])

        result = calculate_trend_drop(time, signal, (-1.0, -0.5, 0.5, 1.0))

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], 6.0)

    def test_insufficient_samples_return_invalid(self):
        time = np.array([-1.0, 0.5, 1.0])
        signal = np.array([2.0, 1.0, 1.5])

        self.assertEqual(
            calculate_trend_drop(time, signal, (-1.0, -0.5, 0.5, 1.0)),
            {"valid": False},
        )

    def test_duplicate_times_do_not_count_as_two_fit_points(self):
        time = np.array([-1.0, -1.0, 0.5, 1.0])
        signal = np.array([2.0, 2.0, 1.0, 1.5])

        self.assertEqual(
            calculate_trend_drop(time, signal, (-1.0, -0.5, 0.5, 1.0)),
            {"valid": False},
        )

    def test_nan_and_infinity_are_filtered_pairwise(self):
        time = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        signal = np.where(time < 0, time + 5.0, -time + 2.0)
        signal[1] = np.nan
        signal[6] = np.inf

        result = calculate_trend_drop(time, signal, (-2.0, -0.5, 0.5, 2.0))

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], 3.0)

    def test_nonfinite_data_can_make_window_invalid(self):
        time = np.array([-1.0, -0.5, 0.5, 1.0])
        signal = np.array([np.nan, 2.0, 1.0, 2.0])

        self.assertEqual(
            calculate_trend_drop(time, signal, (-1.0, -0.5, 0.5, 1.0)),
            {"valid": False},
        )

    def test_structurally_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            calculate_trend_drop([0.0, 1.0], [1.0], (-1.0, -0.5, 0.5, 1.0))
        with self.assertRaises(ValueError):
            calculate_trend_drop([[0.0, 1.0]], [[1.0, 2.0]], (-1.0, -0.5, 0.5, 1.0))
        with self.assertRaises(ValueError):
            calculate_trend_drop([0.0, 1.0], [1.0, 2.0], (-1.0, 1.0, 2.0))

    def test_compatibility_wrapper_uses_same_least_squares_result(self):
        time = np.array([-2.0, -1.0, 1.0, 2.0])
        signal = np.where(time < 0, time + 5.0, time + 2.0)
        points = (-2.0, -1.0, 1.0, 2.0)

        direct = calculate_trend_drop(time, signal, points)
        compatible = calculate_2pt_trend_drop(time, signal, points)

        self.assertEqual(direct["valid"], compatible["valid"])
        self.assertAlmostEqual(direct["delta"], compatible["delta"])
        np.testing.assert_allclose(direct["coeff_pre"], compatible["coeff_pre"])
        np.testing.assert_allclose(direct["coeff_post"], compatible["coeff_post"])


class MovingAverageTests(unittest.TestCase):
    def test_reflect_padding_at_edges(self):
        result = moving_average([1.0, 2.0, 3.0], 3)
        np.testing.assert_allclose(result, [5.0 / 3.0, 2.0, 7.0 / 3.0])

    def test_even_window_matches_potter_padding(self):
        result = moving_average([1.0, 2.0, 3.0], 2)
        np.testing.assert_allclose(result, [1.5, 1.5, 2.5])

    def test_window_one_returns_float_copy(self):
        original = np.array([1, 2, 3], dtype=int)
        result = moving_average(original, 1)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        self.assertEqual(result.dtype.kind, "f")
        self.assertIsNot(result, original)

    def test_nan_propagates_through_affected_windows(self):
        result = moving_average([1.0, np.nan, 3.0], 3)
        self.assertTrue(np.isnan(result).all())

    def test_invalid_window_sizes_raise_value_error(self):
        for window_size in (0, -1, 1.5, True):
            with self.subTest(window_size=window_size):
                with self.assertRaises(ValueError):
                    moving_average([1.0, 2.0], window_size)

    def test_empty_input_with_nontrivial_window_raises_value_error(self):
        with self.assertRaises(ValueError):
            moving_average([], 3)


class ComputeHalfWindowTests(unittest.TestCase):
    def test_defaults_match_potter_rounding(self):
        self.assertEqual(compute_half_win(), 1.5)

    def test_custom_window_rounds_total_span_to_multiple_of_three(self):
        config = {"pre_win": (-4.0, -3.0), "post_win": (0.2, 2.0)}
        self.assertEqual(compute_half_win(config), 4.5)
        self.assertEqual((2.0 * compute_half_win(config)) % 3.0, 0.0)

    def test_invalid_config_or_window_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_half_win([])
        with self.assertRaises(ValueError):
            compute_half_win({"pre_win": (-1.0,)})
        with self.assertRaises(ValueError):
            compute_half_win({"post_win": (0.5, np.nan)})


if __name__ == "__main__":
    unittest.main()
