import ast
import builtins
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from labquake_explorer.analysis import calculate_event_loading_stiffness
from labquake_explorer.analysis import k_stiffness


def _load_potter_k_stiffness():
    """Directly load Potter's pure production analysis modules."""
    analysis_path = (
        Path(__file__).resolve().parents[2]
        / "student-potter"
        / "labquake_explorer"
        / "analysis"
    )
    source_path = analysis_path / "k_stiffness_analyzer.py"
    dependency_name = "labquake_explorer.analysis.event_drop_analyzer"
    previous = sys.modules.get(dependency_name)
    try:
        dependency_spec = importlib.util.spec_from_file_location(
            dependency_name, analysis_path / "event_drop_analyzer.py"
        )
        dependency = importlib.util.module_from_spec(dependency_spec)
        sys.modules[dependency_name] = dependency
        dependency_spec.loader.exec_module(dependency)
        spec = importlib.util.spec_from_file_location(
            "_potter_k_stiffness_characterization", source_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(dependency_name, None)
        else:
            sys.modules[dependency_name] = previous


POTTER_K_STIFFNESS = _load_potter_k_stiffness()


def _potter_fallback_ransac(x, y):
    """Call Potter production while making its optional sklearn import fail."""
    original_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name == "sklearn.linear_model":
            raise ImportError("forced test-only fallback")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=import_without_sklearn):
        return POTTER_K_STIFFNESS.robust_fit_ransac(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )


def _potter_sklearn_ransac_seeded(x, y, seed=20260806):
    """Stabilize characterization without changing Potter's production API."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return POTTER_K_STIFFNESS.robust_fit_ransac(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
    finally:
        np.random.set_state(state)


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


class PotterLoadingStiffnessParityTests(unittest.TestCase):
    """Characterize parity and known differences against Potter production."""

    def setUp(self):
        self.time = np.linspace(-5.0, 5.0, 1001)
        self.slip = 0.4 * self.time + 0.03 * np.sin(2 * np.pi * 2 * self.time)
        self.tau = 2.75 * self.slip + 0.2
        self.common = {
            "event_time": 0.0,
            "pre_start": -3.0,
            "pre_end": -0.5,
            "window_sec": 3.5,
            "smooth_w": 1,
            "highpass_freq": 0.0,
            "lowpass_freq": 0.0,
        }

    def official(self, **overrides):
        arguments = dict(self.common)
        arguments.update(
            time=self.time,
            tau_signal=self.tau,
            slip_signal=self.slip,
        )
        arguments.update(overrides)
        return calculate_event_loading_stiffness(**arguments)

    def potter(self, **overrides):
        arguments = {
            "k_pre_start": self.common["pre_start"],
            "k_pre_end": self.common["pre_end"],
            "k_window_sec": self.common["window_sec"],
            "k_smooth_w": self.common["smooth_w"],
            "k_highpass_freq": self.common["highpass_freq"],
            "k_lowpass_freq": self.common["lowpass_freq"],
            "k_use_ransac": False,
            "k_slip_source": "LVDT",
        }
        arguments.update(overrides)
        return POTTER_K_STIFFNESS.analyze_single_k(
            {
                "time": self.time,
                "tau_local": self.tau,
                "LP_displacement": self.slip,
            },
            [{"event_time": 0.0}],
            0,
            arguments,
        )

    def assert_ols_parity(self, **overrides):
        official_overrides = dict(overrides)
        potter_overrides = {
            {
                "smooth_w": "k_smooth_w",
                "highpass_freq": "k_highpass_freq",
                "lowpass_freq": "k_lowpass_freq",
                "pre_start": "k_pre_start",
                "pre_end": "k_pre_end",
                "window_sec": "k_window_sec",
            }.get(name, name): value
            for name, value in overrides.items()
        }
        official = self.official(**official_overrides)
        potter = self.potter(**potter_overrides)
        self.assertTrue(official["valid"])
        self.assertFalse(potter["skipped"])
        np.testing.assert_allclose(
            official["coefficients"], potter["k_coeffs"], rtol=1e-12, atol=1e-12
        )
        self.assertEqual(official["k"], official["coefficients"][0])
        self.assertEqual(official["intercept"], official["coefficients"][1])
        return official, potter

    def test_ols_positive_negative_intercept_and_smoothing_parity(self):
        original_tau = self.tau.copy()
        original_slip = self.slip.copy()
        for slope, intercept, smooth_w in (
            (3.0, 4.5, 1), (-2.25, -1.7, 1), (1.75, 6.0, 7)
        ):
            with self.subTest(slope=slope, intercept=intercept, smooth=smooth_w):
                tau = slope * self.slip + intercept
                self.tau = tau
                official, _ = self.assert_ols_parity(smooth_w=smooth_w)
                self.assertAlmostEqual(official["k"], slope, places=10)
                np.testing.assert_array_equal(self.tau, tau)
                np.testing.assert_array_equal(self.slip, original_slip)
        self.tau = original_tau

    def test_ols_filter_and_processing_order_parity(self):
        cases = (
            {"smooth_w": 5},
            {"smooth_w": 3, "highpass_freq": 0.2},
            {"smooth_w": 3, "lowpass_freq": 10.0},
            {"smooth_w": 3, "highpass_freq": 0.2, "lowpass_freq": 10.0},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                official, _ = self.assert_ols_parity(**arguments)
                mask = (self.time >= -3.5) & (self.time <= 3.5)
                fs = 1.0 / np.median(np.diff(self.time[mask]))
                expected_tau = POTTER_K_STIFFNESS._process_signal(
                    self.tau[mask], arguments.get("smooth_w", 1),
                    arguments.get("highpass_freq", 0.0),
                    arguments.get("lowpass_freq", 0.0), fs,
                )
                expected_slip = POTTER_K_STIFFNESS._process_signal(
                    self.slip[mask], arguments.get("smooth_w", 1),
                    arguments.get("highpass_freq", 0.0),
                    arguments.get("lowpass_freq", 0.0), fs,
                )
                expected_tau = expected_tau - expected_tau[0]
                expected_slip = expected_slip - expected_slip[0]
                np.testing.assert_allclose(official["processed_tau"], expected_tau)
                np.testing.assert_allclose(official["processed_slip"], expected_slip)
                np.testing.assert_array_equal(official["raw_tau"], self.tau[mask])
                np.testing.assert_array_equal(official["raw_slip"], self.slip[mask])

    def test_ols_inclusive_masks_and_result_arrays_match_reference(self):
        official, _ = self.assert_ols_parity(pre_start=-2.0, pre_end=-1.0)
        mask = (self.time >= -3.5) & (self.time <= 3.5)
        expected_relative = self.time[mask]
        expected_fit = (expected_relative >= -2.0) & (expected_relative <= -1.0)
        np.testing.assert_array_equal(official["relative_time"], expected_relative)
        np.testing.assert_array_equal(official["fit_mask"], expected_fit)
        self.assertEqual(expected_relative[expected_fit][0], -2.0)
        self.assertEqual(expected_relative[expected_fit][-1], -1.0)

    def test_perfect_line_ransac_all_paths_match(self):
        x = np.linspace(-2.0, 2.0, 30)
        y = -3.5 * x + 1.25
        sklearn_result = _potter_sklearn_ransac_seeded(x, y)
        fallback_result = _potter_fallback_ransac(x, y)
        official_result = k_stiffness._fit_ransac(x, y)
        for result in (sklearn_result, fallback_result, official_result):
            np.testing.assert_allclose(result, [-3.5, 1.25], rtol=1e-10, atol=1e-10)

    def test_ransac_outlier_dataset_contracts(self):
        base_x = np.linspace(-3.0, 3.0, 30)
        datasets = {}
        y = 2.0 * base_x + 0.5
        y[10] += 50.0
        datasets["single_large"] = (base_x, y)
        y = 2.0 * base_x + 0.5
        y[[4, 8, 21, 25]] += np.array([20.0, -20.0, 20.0, -20.0])
        datasets["symmetric"] = (base_x, y)
        y = 2.0 * base_x + 0.5
        y[10:16] += 15.0
        datasets["clustered"] = (base_x, y)
        x = base_x.copy()
        x[-1] = 40.0
        y = 2.0 * base_x + 0.5
        datasets["high_leverage"] = (x, y)
        x = np.r_[np.linspace(-3.0, 3.0, 24), [0.0] * 6]
        y = 2.0 * x + 0.5
        y[-3:] += 10.0
        datasets["duplicate_x"] = (x, y)

        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(-3.0, 3.0, 30))
        y = 2.0 * x + 0.5 + rng.normal(0.0, 0.15, 30)
        outliers = rng.choice(30, 8, replace=False)
        y[outliers] += rng.normal(0.0, 5.0, 8)
        datasets["fixed_noisy_outliers"] = (x, y)

        observed_divergence = False
        for name, (x, y) in datasets.items():
            with self.subTest(dataset=name):
                sklearn_result = _potter_sklearn_ransac_seeded(x, y)
                fallback_result = _potter_fallback_ransac(x, y)
                official_result = k_stiffness._fit_ransac(x, y)
                for result in (sklearn_result, fallback_result, official_result):
                    self.assertEqual(np.asarray(result).shape, (2,))
                    self.assertTrue(np.all(np.isfinite(result)))
                np.testing.assert_allclose(
                    fallback_result, official_result, rtol=0.0, atol=0.0
                )
                if name == "fixed_noisy_outliers":
                    self.assertFalse(
                        np.allclose(
                            sklearn_result, official_result, rtol=1e-12, atol=1e-12
                        )
                    )
                if not np.allclose(sklearn_result, official_result, rtol=1e-12, atol=1e-12):
                    observed_divergence = True
        self.assertTrue(observed_divergence)

    def test_sklearn_randomness_contract_and_official_determinism(self):
        x = np.linspace(-3.0, 3.0, 40)
        y = 1.8 * x - 0.4
        y[[2, 5, 8, 12, 28, 32, 36]] += [8, -9, 7, -8, 9, -7, 8]
        potter_results = np.asarray(
            [POTTER_K_STIFFNESS.robust_fit_ransac(x, y) for _ in range(20)]
        )
        official_results = np.asarray(
            [k_stiffness._fit_ransac(x, y) for _ in range(20)]
        )
        self.assertTrue(np.all(np.isfinite(potter_results)))
        np.testing.assert_array_equal(
            official_results, np.repeat(official_results[:1], 20, axis=0)
        )

    def test_small_minimum_and_negative_ransac_characterization(self):
        for count in (5, 6):
            x = np.linspace(-1.0, 1.0, count)
            y = -4.0 * x + 2.0
            with self.subTest(count=count):
                sklearn_result = _potter_sklearn_ransac_seeded(x, y)
                fallback_result = _potter_fallback_ransac(x, y)
                np.testing.assert_allclose(sklearn_result, [-4.0, 2.0], atol=1e-10)
                np.testing.assert_allclose(fallback_result, [-4.0, 2.0], atol=1e-10)
                if count == 5:
                    with self.assertRaises(ValueError):
                        k_stiffness._fit_ransac(x, y)
                else:
                    np.testing.assert_allclose(
                        k_stiffness._fit_ransac(x, y), [-4.0, 2.0], atol=1e-10
                    )

    def test_constant_and_near_constant_failure_contracts(self):
        for label, x in (
            ("constant", np.ones(12)),
            ("near_constant", 1.0 + np.arange(12) * np.finfo(float).eps),
        ):
            y = np.linspace(0.0, 1.0, 12)
            with self.subTest(label=label):
                sklearn_result = _potter_sklearn_ransac_seeded(x, y)
                self.assertEqual(np.asarray(sklearn_result).shape, (2,))
                self.assertTrue(np.all(np.isfinite(sklearn_result)))
                fallback = _potter_fallback_ransac(x, y)
                self.assertEqual(np.asarray(fallback).shape, (2,))
                self.assertTrue(np.all(np.isfinite(fallback)))
                with self.assertRaises(ValueError):
                    k_stiffness._fit_ransac(x, y)
        self.assertEqual(self.official(slip_signal=np.ones_like(self.slip)), {"valid": False})

    def test_regression_failure_contract_is_swallowed_by_potter_and_invalid_official(self):
        with mock.patch.object(
            POTTER_K_STIFFNESS, "robust_fit_ransac", side_effect=ValueError("fit")
        ):
            potter = self.potter(k_use_ransac=True)
        self.assertFalse(potter["skipped"])
        self.assertTrue(np.isnan(potter["k"]["value"]))
        self.assertNotIn("k_coeffs", potter)
        with mock.patch.object(k_stiffness, "_fit_ransac", side_effect=ValueError("fit")):
            self.assertEqual(self.official(use_ransac=True), {"valid": False})

    def test_cutoff_boundaries_characterize_known_difference(self):
        fs = 1.0 / np.median(np.diff(self.time))
        nyquist = fs / 2.0
        raw = self.tau.copy()
        for cutoff in (0.0, nyquist, nyquist + 1e-9):
            with self.subTest(cutoff=cutoff):
                np.testing.assert_array_equal(
                    POTTER_K_STIFFNESS._apply_highpass(raw, cutoff, fs), raw
                )
                np.testing.assert_array_equal(
                    POTTER_K_STIFFNESS._apply_lowpass(raw, cutoff, fs), raw
                )
        for name in ("highpass_freq", "lowpass_freq"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.official(**{name: nyquist})
                with self.assertRaises(ValueError):
                    self.official(**{name: nyquist + 1e-9})

    def test_cutoff_below_nyquist_and_combined_filters_match(self):
        nyquist = 1.0 / np.median(np.diff(self.time)) / 2.0
        for arguments in (
            {"highpass_freq": 0.2},
            {"lowpass_freq": nyquist - 1.0},
            {"highpass_freq": 0.2, "lowpass_freq": 10.0},
            {"highpass_freq": 10.0, "lowpass_freq": 1.0},
        ):
            with self.subTest(arguments=arguments):
                self.assert_ols_parity(**arguments)

    def test_negative_cutoffs_and_short_filter_window_contracts(self):
        raw = np.arange(20.0)
        np.testing.assert_array_equal(
            POTTER_K_STIFFNESS._apply_highpass(raw, -1.0, 10.0), raw
        )
        np.testing.assert_array_equal(
            POTTER_K_STIFFNESS._apply_lowpass(raw, -1.0, 10.0), raw
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.official(highpass_freq=-1.0)

        short_time = np.linspace(-1.0, 1.0, 20)
        short_history = {
            "time": short_time,
            "tau_local": 2.0 * short_time,
            "LP_displacement": short_time,
        }
        short_config = {
            "k_pre_start": -0.9,
            "k_pre_end": -0.1,
            "k_window_sec": 1.0,
            "k_smooth_w": 1,
            "k_highpass_freq": 0.5,
        }
        with mock.patch.object(
            POTTER_K_STIFFNESS, "filtfilt", side_effect=ValueError("padlen")
        ):
            with self.assertRaisesRegex(ValueError, "padlen"):
                POTTER_K_STIFFNESS.analyze_single_k(
                    short_history, [{"event_time": 0.0}], 0, short_config
                )
        with mock.patch.object(k_stiffness, "filtfilt", side_effect=ValueError("padlen")):
            official = calculate_event_loading_stiffness(
                time=short_time, tau_signal=2.0 * short_time,
                slip_signal=short_time, event_time=0.0,
                pre_start=-0.9, pre_end=-0.1, window_sec=1.0,
                smooth_w=1, highpass_freq=0.5,
            )
        self.assertEqual(official, {"valid": False})

    def test_result_contract_characterization(self):
        official = self.official()
        potter = self.potter()
        self.assertEqual(
            set(official),
            {
                "valid", "k", "intercept", "coefficients", "relative_time",
                "raw_tau", "raw_slip", "processed_tau", "processed_slip", "fit_mask",
            },
        )
        self.assertEqual(set(potter), {"event_idx", "skipped", "trigger_time", "k", "k_coeffs"})
        self.assertIsInstance(potter["k"], dict)
        self.assertIsInstance(potter["k_coeffs"], list)
        self.assertIsInstance(official["coefficients"], np.ndarray)


if __name__ == "__main__":
    unittest.main()
