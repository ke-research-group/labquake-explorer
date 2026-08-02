import ast
import inspect
import unittest
from unittest import mock

import numpy as np

from labquake_explorer.analysis import calculate_event_loading_stiffness
from labquake_explorer.analysis import k_stiffness


class EventLoadingStiffnessTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(-5.0, 5.0, 101)
        self.slip = 0.5 * self.time + 4.0
        self.tau = 3.0 * self.slip - 2.0

    def calculate(self, **overrides):
        arguments = {
            "time": self.time,
            "tau_signal": self.tau,
            "slip_signal": self.slip,
            "event_time": 0.0,
            "pre_start": -3.0,
            "pre_end": -0.5,
            "window_sec": 3.5,
            "smooth_w": 1,
        }
        arguments.update(overrides)
        return calculate_event_loading_stiffness(**arguments)

    def test_exact_positive_linear_k_and_python_scalars(self):
        result = self.calculate()
        self.assertTrue(result["valid"])
        self.assertIsInstance(result["k"], float)
        self.assertIsInstance(result["intercept"], float)
        self.assertAlmostEqual(result["k"], 3.0)

    def test_exact_negative_signed_k_is_not_absolute(self):
        result = self.calculate(tau_signal=-2.5 * self.slip + 7.0)
        self.assertAlmostEqual(result["k"], -2.5)

    def test_nonzero_intercept_and_coefficients_are_correct(self):
        tau = 4.0 * self.slip + 0.2 * self.time**2
        result = self.calculate(tau_signal=tau)
        expected = np.polyfit(
            result["processed_slip"][result["fit_mask"]],
            result["processed_tau"][result["fit_mask"]],
            1,
        )
        self.assertNotAlmostEqual(result["intercept"], 0.0)
        np.testing.assert_allclose(result["coefficients"], expected)

    def test_each_call_owns_result_and_arrays(self):
        first = self.calculate()
        second = self.calculate()
        self.assertIsNot(first, second)
        for name in (
            "coefficients", "relative_time", "raw_tau", "raw_slip",
            "processed_tau", "processed_slip", "fit_mask",
        ):
            self.assertIsNot(first[name], second[name])

    def test_returned_arrays_do_not_share_input_memory(self):
        result = self.calculate()
        for name in ("raw_tau", "raw_slip", "processed_tau", "processed_slip"):
            self.assertFalse(np.shares_memory(result[name], self.tau))
            self.assertFalse(np.shares_memory(result[name], self.slip))

    def test_event_mask_is_inclusive(self):
        result = self.calculate(window_sec=3.0, pre_start=-2.0)
        self.assertEqual(result["relative_time"][0], -3.0)
        self.assertEqual(result["relative_time"][-1], 3.0)

    def test_pre_mask_is_inclusive(self):
        result = self.calculate(pre_start=-2.0, pre_end=-1.0)
        selected = result["relative_time"][result["fit_mask"]]
        self.assertEqual(selected[0], -2.0)
        self.assertEqual(selected[-1], -1.0)

    def test_half_window_uses_pre_start_requirement(self):
        result = self.calculate(window_sec=1.0, pre_start=-4.0, pre_end=-3.0)
        self.assertEqual(result["relative_time"][0], -4.5)
        self.assertEqual(result["relative_time"][-1], 4.5)

    def test_nineteen_event_samples_are_invalid(self):
        time = np.linspace(-0.9, 0.9, 19)
        self.assertEqual(
            self.calculate(time=time, slip_signal=time, tau_signal=2 * time,
                           window_sec=0.9, pre_start=-0.8, pre_end=-0.2),
            {"valid": False},
        )

    def test_twenty_event_samples_continue(self):
        time = np.linspace(-0.95, 0.95, 20)
        result = self.calculate(time=time, slip_signal=time, tau_signal=2 * time,
                                window_sec=0.95, pre_start=-0.9, pre_end=-0.1)
        self.assertTrue(result["valid"])

    def test_five_fit_samples_are_invalid(self):
        result = self.calculate(pre_start=-0.9, pre_end=-0.5)
        self.assertEqual(result, {"valid": False})

    def test_six_fit_samples_can_fit(self):
        result = self.calculate(pre_start=-1.0, pre_end=-0.5)
        self.assertTrue(result["valid"])

    def test_invalid_windows_raise_value_error(self):
        cases = (
            {"pre_start": -0.5, "pre_end": -1.0},
            {"pre_start": -1.0, "pre_end": 0.1},
            {"pre_start": 0.0, "pre_end": 0.0},
            {"window_sec": 0.0},
            {"window_sec": -1.0},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.calculate(**arguments)

    def test_processing_baselines_after_moving_average(self):
        result = self.calculate(smooth_w=3)
        expected_tau = k_stiffness.moving_average(
            result["raw_tau"], 3
        )
        expected_tau -= expected_tau[0]
        np.testing.assert_allclose(result["processed_tau"], expected_tau)
        self.assertEqual(result["processed_tau"][0], 0.0)
        self.assertEqual(result["processed_slip"][0], 0.0)

    def test_raw_arrays_are_not_baselined(self):
        result = self.calculate()
        self.assertNotEqual(result["raw_tau"][0], 0.0)
        self.assertNotEqual(result["raw_slip"][0], 0.0)

    def test_tau_and_slip_receive_same_processing_order(self):
        calls = []

        def smooth(values, window):
            calls.append(("smooth", float(values[0])))
            return np.asarray(values, dtype=float) + 10.0

        def filter_signal(b, a, values):
            calls.append(("filter", float(values[0])))
            return np.asarray(values) + 20.0

        with mock.patch.object(k_stiffness, "moving_average", side_effect=smooth), \
             mock.patch.object(k_stiffness, "butter", return_value=(np.ones(2), np.ones(2))), \
             mock.patch.object(k_stiffness, "filtfilt", side_effect=filter_signal):
            result = self.calculate(highpass_freq=0.5, lowpass_freq=1.0)
        self.assertTrue(result["valid"])
        self.assertEqual([item[0] for item in calls], [
            "smooth", "filter", "filter", "smooth", "filter", "filter"
        ])

    def test_processing_is_applied_after_event_window_selection(self):
        lengths = []

        def smooth(values, window):
            lengths.append(len(values))
            return np.asarray(values, dtype=float).copy()

        with mock.patch.object(k_stiffness, "moving_average", side_effect=smooth):
            result = self.calculate(window_sec=2.0, pre_start=-1.0)
        self.assertTrue(result["valid"])
        self.assertEqual(lengths, [41, 41])
        self.assertLess(lengths[0], len(self.time))

    def test_inputs_are_not_modified(self):
        originals = (self.time.copy(), self.tau.copy(), self.slip.copy())
        self.calculate(smooth_w=3)
        np.testing.assert_array_equal(self.time, originals[0])
        np.testing.assert_array_equal(self.tau, originals[1])
        np.testing.assert_array_equal(self.slip, originals[2])

    def test_zero_cutoffs_do_not_call_filters(self):
        with mock.patch.object(k_stiffness, "butter") as butter_mock, \
             mock.patch.object(k_stiffness, "filtfilt") as filter_mock:
            result = self.calculate()
        self.assertTrue(result["valid"])
        butter_mock.assert_not_called()
        filter_mock.assert_not_called()

    def test_highpass_and_lowpass_use_fourth_order_in_order(self):
        butter_calls = []

        def make_filter(order, normalized, btype):
            butter_calls.append((order, btype))
            return np.ones(2), np.ones(2)

        with mock.patch.object(k_stiffness, "butter", side_effect=make_filter), \
             mock.patch.object(k_stiffness, "filtfilt", side_effect=lambda b, a, x: x):
            result = self.calculate(highpass_freq=0.5, lowpass_freq=1.0)
        self.assertTrue(result["valid"])
        self.assertEqual(butter_calls, [
            (4, "high"), (4, "low"), (4, "high"), (4, "low")
        ])

    def test_real_filtering_is_deterministic(self):
        time = np.linspace(-4.0, 4.0, 401)
        slip = time + 0.05 * np.sin(2 * np.pi * 8 * time)
        tau = 2.0 * time + 0.08 * np.sin(2 * np.pi * 8 * time)
        first = self.calculate(time=time, slip_signal=slip, tau_signal=tau,
                               highpass_freq=0.2, lowpass_freq=10.0,
                               smooth_w=3)
        second = self.calculate(time=time, slip_signal=slip, tau_signal=tau,
                                highpass_freq=0.2, lowpass_freq=10.0,
                                smooth_w=3)
        self.assertTrue(first["valid"])
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])

    def test_invalid_cutoffs_raise(self):
        for arguments in (
            {"highpass_freq": -1.0}, {"lowpass_freq": -1.0},
            {"highpass_freq": 5.0}, {"lowpass_freq": 5.0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.calculate(**arguments)

    def test_filter_window_too_short_is_invalid(self):
        time = np.linspace(-1.0, 1.0, 20)
        with mock.patch.object(
            k_stiffness, "filtfilt", side_effect=ValueError("padlen")
        ):
            result = self.calculate(
                time=time, tau_signal=2 * time, slip_signal=time,
                event_time=0.0, pre_start=-0.9, pre_end=-0.1,
                window_sec=1.0, highpass_freq=0.5,
            )
        self.assertEqual(result, {"valid": False})

    def test_constant_and_near_constant_slip_are_invalid(self):
        constant = np.ones_like(self.slip)
        near = 1.0 + np.linspace(0.0, np.finfo(float).eps, self.slip.size)
        self.assertEqual(self.calculate(slip_signal=constant), {"valid": False})
        self.assertEqual(self.calculate(slip_signal=near), {"valid": False})

    def test_nonfinite_regression_coefficients_are_invalid(self):
        with mock.patch.object(np, "polyfit", return_value=np.array([np.nan, 0.0])):
            self.assertEqual(self.calculate(), {"valid": False})

    def test_ransac_handles_outlier_differently_from_ols(self):
        time = np.linspace(-5.0, 5.0, 101)
        slip = time.copy()
        tau = 2.0 * slip + 1.0
        tau[np.argmin(np.abs(time + 1.5))] += 50.0
        ols = self.calculate(time=time, slip_signal=slip, tau_signal=tau)
        robust = self.calculate(time=time, slip_signal=slip, tau_signal=tau,
                                use_ransac=True)
        self.assertTrue(robust["valid"])
        self.assertAlmostEqual(robust["k"], 2.0, places=10)
        self.assertGreater(abs(ols["k"] - 2.0), 0.1)

    def test_ransac_is_deterministic_and_preserves_negative_sign(self):
        tau = -4.0 * self.slip + 3.0
        tau = tau.copy()
        tau[25] += 20.0
        first = self.calculate(tau_signal=tau, use_ransac=True)
        second = self.calculate(tau_signal=tau, use_ransac=True)
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
        self.assertAlmostEqual(first["k"], -4.0, places=10)

    def test_ransac_failure_does_not_fallback_to_ols(self):
        with mock.patch.object(k_stiffness, "_fit_ransac", side_effect=ValueError):
            result = self.calculate(use_ransac=True)
        self.assertEqual(result, {"valid": False})

    def test_length_mismatch_is_named(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self.calculate(tau_signal=self.tau[:-1])

    def test_invalid_arrays_raise_with_argument_name(self):
        cases = (
            ("time", []), ("tau_signal", np.ones((2, 2))),
            ("slip_signal", np.array([True] * self.time.size)),
            ("tau_signal", np.array(["x"] * self.time.size)),
            ("slip_signal", np.r_[self.slip[:-1], np.nan]),
            ("tau_signal", np.r_[self.tau[:-1], np.inf]),
        )
        for name, value in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.calculate(**{name: value})

    def test_time_must_be_strictly_increasing(self):
        decreasing = self.time.copy()
        decreasing[50], decreasing[51] = decreasing[51], decreasing[50]
        duplicate = self.time.copy()
        duplicate[51] = duplicate[50]
        for value in (decreasing, duplicate):
            with self.subTest(), self.assertRaisesRegex(ValueError, "strictly increasing"):
                self.calculate(time=value)

    def test_bool_and_array_scalars_are_rejected(self):
        scalar_names = (
            "event_time", "pre_start", "pre_end", "window_sec",
            "highpass_freq", "lowpass_freq",
        )
        for name in scalar_names:
            for value in (True, np.array([1.0])):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.calculate(**{name: value})

    def test_invalid_smooth_window_is_rejected(self):
        for value in (True, 100.0, 0, -1, np.array(1)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "smooth_w"):
                self.calculate(smooth_w=value)

    def test_invalid_ransac_flag_is_rejected(self):
        for value in (1, 0.0, "yes", np.array(True)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "use_ransac"):
                self.calculate(use_ransac=value)

    def test_numpy_scalar_parameters_are_accepted(self):
        result = self.calculate(
            event_time=np.float64(0.0), pre_start=np.float64(-3.0),
            pre_end=np.float64(-0.5), window_sec=np.float64(3.5),
            smooth_w=np.int64(1), highpass_freq=np.float64(0.0),
            lowpass_freq=np.float64(0.0), use_ransac=np.bool_(False),
        )
        self.assertTrue(result["valid"])

    def test_module_has_no_forbidden_imports(self):
        tree = ast.parse(inspect.getsource(k_stiffness))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {"tkinter", "h5py", "labquake_explorer.data.data_manager"}
        self.assertTrue(imported.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
