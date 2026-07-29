"""Deterministic tests for the pure event-drop calculation helpers."""

import unittest
from unittest import mock

import numpy as np

from labquake_explorer.analysis.event_drop import (
    calculate_2pt_trend_drop,
    calculate_event_drop_metrics,
    calculate_event_signal_drop,
    calculate_interevent_displacement_metrics,
    calculate_trend_drop,
    compute_half_win,
    moving_average,
)


class CalculateIntereventDisplacementMetricsTests(unittest.TestCase):
    def test_positive_signed_d_push(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=12.0,
            previous_event_time=10.0,
            push_speed=3.5,
        )

        self.assertEqual(
            result,
            {
                "D_Push": {"valid": True, "value": 7.0},
                "D_max": {"valid": False},
                "D_reference": {"valid": False},
            },
        )
        self.assertIsInstance(result["D_Push"]["value"], float)

    def test_negative_and_zero_time_differences_remain_valid(self):
        negative = calculate_interevent_displacement_metrics(
            current_event_time=8.0,
            previous_event_time=10.0,
            push_speed=3.5,
        )
        zero = calculate_interevent_displacement_metrics(
            current_event_time=10.0,
            previous_event_time=10.0,
            push_speed=-3.5,
        )

        self.assertEqual(negative["D_Push"]["value"], -7.0)
        self.assertEqual(zero["D_Push"], {"valid": True, "value": 0.0})

    def test_missing_previous_event_is_unavailable_without_value(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=12.0,
            previous_event_time=None,
            push_speed=3.5,
        )

        self.assertEqual(
            result,
            {
                "D_Push": {"valid": False},
                "D_max": {"valid": False},
                "D_reference": {"valid": False},
            },
        )

    def test_python_integer_and_numpy_scalar_inputs_return_python_float(self):
        integer_result = calculate_interevent_displacement_metrics(
            current_event_time=12,
            previous_event_time=10,
            push_speed=3,
        )
        numpy_result = calculate_interevent_displacement_metrics(
            current_event_time=np.float64(12),
            previous_event_time=np.int64(10),
            push_speed=np.float32(3.5),
        )

        self.assertEqual(integer_result["D_Push"]["value"], 6.0)
        self.assertEqual(numpy_result["D_Push"]["value"], 7.0)
        self.assertIsInstance(integer_result["D_Push"]["value"], float)
        self.assertIsInstance(numpy_result["D_Push"]["value"], float)

    def test_nonfinite_inputs_name_the_invalid_argument(self):
        cases = (
            ("current_event_time", np.nan),
            ("current_event_time", np.inf),
            ("previous_event_time", np.nan),
            ("previous_event_time", -np.inf),
            ("push_speed", np.nan),
            ("push_speed", np.inf),
        )
        for name, value in cases:
            arguments = {
                "current_event_time": 12.0,
                "previous_event_time": 10.0,
                "push_speed": 3.5,
            }
            arguments[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    calculate_interevent_displacement_metrics(**arguments)

    def test_bool_inputs_are_rejected_for_each_argument(self):
        for name in (
            "current_event_time",
            "previous_event_time",
            "push_speed",
        ):
            arguments = {
                "current_event_time": 12.0,
                "previous_event_time": 10.0,
                "push_speed": 3.5,
            }
            arguments[name] = True
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    calculate_interevent_displacement_metrics(**arguments)

    def test_length_one_arrays_are_not_treated_as_scalars(self):
        for name in (
            "current_event_time",
            "previous_event_time",
            "push_speed",
        ):
            arguments = {
                "current_event_time": 12.0,
                "previous_event_time": 10.0,
                "push_speed": 3.5,
            }
            value = np.array([arguments[name]])
            arguments[name] = value
            original = value.copy()
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    calculate_interevent_displacement_metrics(**arguments)
                np.testing.assert_array_equal(value, original)

    def test_each_call_creates_new_result_dictionaries(self):
        first = calculate_interevent_displacement_metrics(
            current_event_time=12.0,
            previous_event_time=10.0,
            push_speed=3.5,
        )
        second = calculate_interevent_displacement_metrics(
            current_event_time=12.0,
            previous_event_time=10.0,
            push_speed=3.5,
        )

        self.assertIsNot(first, second)
        self.assertIsNot(first["D_Push"], second["D_Push"])
        self.assertIsNot(first["D_max"], second["D_max"])
        self.assertIsNot(first["D_reference"], second["D_reference"])

    def test_d_max_smooths_full_run_before_delayed_sample_difference(self):
        time = np.arange(5.0)
        lvdt = np.array([0.0, 0.0, 0.0, 9.0, 9.0])

        result = calculate_interevent_displacement_metrics(
            current_event_time=3.0,
            previous_event_time=1.0,
            push_speed=2.0,
            time=time,
            lvdt_signal=lvdt,
            delay_sec=0.0,
            lvdt_smooth_w=3,
        )

        self.assertEqual(result["D_max"], {"valid": True, "value": 6.0})
        self.assertNotEqual(result["D_max"]["value"], abs(lvdt[3] - lvdt[1]))

    def test_d_max_is_absolute_and_delay_selects_nearest_samples(self):
        time = np.arange(6.0)
        lvdt = np.array([0.0, 8.0, 6.0, 4.0, 2.0, 1.0])

        result = calculate_interevent_displacement_metrics(
            current_event_time=3.2,
            previous_event_time=1.2,
            push_speed=1.0,
            time=time,
            lvdt_signal=lvdt,
            delay_sec=0.6,
            lvdt_smooth_w=1,
        )

        self.assertEqual(result["D_max"], {"valid": True, "value": 4.0})

    def test_nearest_sample_tie_selects_earlier_sample(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=2.5,
            previous_event_time=0.5,
            push_speed=1.0,
            time=np.array([0.0, 1.0, 2.0, 3.0]),
            lvdt_signal=np.array([1.0, 20.0, 5.0, 40.0]),
            delay_sec=0.0,
            lvdt_smooth_w=1,
        )

        self.assertEqual(result["D_max"]["value"], 4.0)

    def test_signal_without_time_and_time_without_signals_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "time must be provided"
        ):
            calculate_interevent_displacement_metrics(
                current_event_time=2.0,
                previous_event_time=1.0,
                push_speed=1.0,
                lvdt_signal=np.array([0.0, 1.0, 2.0]),
            )
        with self.assertRaisesRegex(
            ValueError, "time must be provided"
        ):
            calculate_interevent_displacement_metrics(
                current_event_time=2.0,
                previous_event_time=1.0,
                push_speed=1.0,
                reference_displacement_signal=np.array([0.0, 1.0, 2.0]),
            )
        with self.assertRaisesRegex(
            ValueError, "time must be accompanied"
        ):
            calculate_interevent_displacement_metrics(
                current_event_time=2.0,
                previous_event_time=1.0,
                push_speed=1.0,
                time=np.array([0.0, 1.0, 2.0]),
            )

    def test_out_of_range_target_only_invalidates_d_max(self):
        for current, previous in ((5.0, 1.0), (2.0, -1.0), (5.0, -1.0)):
            with self.subTest(current=current, previous=previous):
                result = calculate_interevent_displacement_metrics(
                    current_event_time=current,
                    previous_event_time=previous,
                    push_speed=2.0,
                    time=np.array([0.0, 1.0, 2.0, 3.0]),
                    lvdt_signal=np.array([0.0, 2.0, 4.0, 6.0]),
                    delay_sec=0.0,
                    lvdt_smooth_w=1,
                )
                self.assertTrue(result["D_Push"]["valid"])
                self.assertEqual(result["D_max"], {"valid": False})

    def test_invalid_run_arrays_are_rejected(self):
        cases = (
            ("empty", np.array([]), np.array([]), "not be empty"),
            ("length", np.array([0.0, 1.0]), np.array([1.0]), "same length"),
            ("time ndim", np.ones((2, 2)), np.ones(4), "time.*one-dimensional"),
            ("lvdt ndim", np.arange(4.0), np.ones((2, 2)), "lvdt_signal.*one-dimensional"),
            ("time finite", np.array([0.0, np.nan]), np.ones(2), "time.*finite"),
            ("lvdt finite", np.arange(2.0), np.array([0.0, np.inf]), "lvdt_signal.*finite"),
            ("descending", np.array([1.0, 0.0]), np.ones(2), "strictly increasing"),
            ("duplicate", np.array([0.0, 0.0]), np.ones(2), "strictly increasing"),
            ("time bool", np.array([False, True]), np.ones(2), "time"),
            ("lvdt bool", np.arange(2.0), np.array([False, True]), "lvdt_signal"),
        )
        for label, time, lvdt, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_interevent_displacement_metrics(
                        current_event_time=1.0,
                        previous_event_time=0.0,
                        push_speed=1.0,
                        time=time,
                        lvdt_signal=lvdt,
                        delay_sec=0.0,
                        lvdt_smooth_w=1,
                    )

    def test_invalid_delay_and_smoothing_window_are_rejected(self):
        base = {
            "current_event_time": 2.0,
            "previous_event_time": 1.0,
            "push_speed": 1.0,
            "time": np.arange(4.0),
            "lvdt_signal": np.arange(4.0),
        }
        for value in (True, np.array([0.0]), np.nan, np.inf):
            with self.subTest(delay=value):
                with self.assertRaisesRegex(ValueError, "delay_sec"):
                    calculate_interevent_displacement_metrics(
                        **base, delay_sec=value, lvdt_smooth_w=1
                    )
        for value in (0, -1, 3.0, True):
            with self.subTest(window=value):
                with self.assertRaisesRegex(ValueError, "lvdt_smooth_w"):
                    calculate_interevent_displacement_metrics(
                        **base, delay_sec=0.0, lvdt_smooth_w=value
                    )

    def test_numpy_parameters_and_input_arrays_are_not_modified(self):
        time = np.arange(5.0)
        lvdt = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        original_time = time.copy()
        original_lvdt = lvdt.copy()

        result = calculate_interevent_displacement_metrics(
            current_event_time=np.float64(3.0),
            previous_event_time=np.float64(1.0),
            push_speed=np.float64(2.0),
            time=time,
            lvdt_signal=lvdt,
            delay_sec=np.float64(0.0),
            lvdt_smooth_w=np.int64(1),
        )

        self.assertTrue(result["D_max"]["valid"])
        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(lvdt, original_lvdt)

    def test_reference_only_uses_raw_delayed_nearest_samples(self):
        time = np.arange(6.0)
        arbitrary_displacement = np.array([0.0, 10.0, 0.0, 4.0, 8.0, 1.0])

        result = calculate_interevent_displacement_metrics(
            current_event_time=3.2,
            previous_event_time=1.2,
            push_speed=2.0,
            time=time,
            reference_displacement_signal=arbitrary_displacement,
            delay_sec=0.6,
            lvdt_smooth_w=0,
        )

        self.assertEqual(result["D_max"], {"valid": False})
        self.assertEqual(
            result["D_reference"], {"valid": True, "value": 8.0}
        )
        self.assertIsInstance(result["D_reference"]["value"], float)

    def test_reference_difference_is_absolute(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=3.0,
            previous_event_time=1.0,
            push_speed=1.0,
            time=np.arange(5.0),
            reference_displacement_signal=np.array(
                [0.0, 9.0, 6.0, 3.0, 1.0]
            ),
            delay_sec=0.0,
        )

        self.assertEqual(
            result["D_reference"], {"valid": True, "value": 6.0}
        )

    def test_both_signals_share_delay_and_nearest_tie(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        lvdt = np.array([1.0, 20.0, 5.0, 40.0])
        reference = np.array([10.0, 100.0, 30.0, 300.0])

        result = calculate_interevent_displacement_metrics(
            current_event_time=2.25,
            previous_event_time=0.25,
            push_speed=1.0,
            time=time,
            lvdt_signal=lvdt,
            reference_displacement_signal=reference,
            delay_sec=0.25,
            lvdt_smooth_w=1,
        )

        self.assertEqual(result["D_max"], {"valid": True, "value": 4.0})
        self.assertEqual(
            result["D_reference"], {"valid": True, "value": 20.0}
        )

    def test_lvdt_only_leaves_reference_unavailable(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=2.0,
            previous_event_time=1.0,
            push_speed=1.0,
            time=np.arange(4.0),
            lvdt_signal=np.arange(4.0),
            delay_sec=0.0,
            lvdt_smooth_w=1,
        )

        self.assertTrue(result["D_max"]["valid"])
        self.assertEqual(result["D_reference"], {"valid": False})

    def test_no_run_context_leaves_both_sample_metrics_unavailable(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=2.0,
            previous_event_time=1.0,
            push_speed=3.0,
            delay_sec=np.nan,
            lvdt_smooth_w=0,
        )

        self.assertEqual(result["D_Push"], {"valid": True, "value": 3.0})
        self.assertEqual(result["D_max"], {"valid": False})
        self.assertEqual(result["D_reference"], {"valid": False})

    def test_missing_previous_event_skips_unused_run_context_validation(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=2.0,
            previous_event_time=None,
            push_speed=1.0,
            time=np.array([np.nan]),
            reference_displacement_signal=np.array([True]),
            delay_sec=np.nan,
        )

        self.assertEqual(
            result,
            {
                "D_Push": {"valid": False},
                "D_max": {"valid": False},
                "D_reference": {"valid": False},
            },
        )

    def test_out_of_range_target_invalidates_both_sample_metrics(self):
        result = calculate_interevent_displacement_metrics(
            current_event_time=5.0,
            previous_event_time=1.0,
            push_speed=2.0,
            time=np.arange(4.0),
            lvdt_signal=np.arange(4.0),
            reference_displacement_signal=np.arange(4.0) * 10.0,
            delay_sec=0.0,
            lvdt_smooth_w=1,
        )

        self.assertTrue(result["D_Push"]["valid"])
        self.assertEqual(result["D_max"], {"valid": False})
        self.assertEqual(result["D_reference"], {"valid": False})

    def test_invalid_reference_arrays_are_rejected_with_argument_name(self):
        cases = (
            ("empty", np.array([])),
            ("length", np.array([1.0, 2.0])),
            ("ndim", np.ones((3, 1))),
            ("nonfinite", np.array([0.0, np.nan, 2.0])),
            ("bool", np.array([False, True, False])),
            ("nonnumeric", np.array(["a", "b", "c"])),
        )
        for label, reference in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, "reference_displacement_signal"
                ):
                    calculate_interevent_displacement_metrics(
                        current_event_time=2.0,
                        previous_event_time=1.0,
                        push_speed=1.0,
                        time=np.arange(3.0),
                        reference_displacement_signal=reference,
                        delay_sec=0.0,
                    )

    def test_reference_input_is_not_modified(self):
        time = np.arange(5.0)
        reference = np.array([0.0, 2.0, 8.0, 3.0, 7.0])
        original_time = time.copy()
        original_reference = reference.copy()

        calculate_interevent_displacement_metrics(
            current_event_time=np.float64(3.0),
            previous_event_time=np.float64(1.0),
            push_speed=np.float64(2.0),
            time=time,
            reference_displacement_signal=reference,
            delay_sec=np.float64(0.0),
        )

        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(reference, original_reference)

    def test_reference_only_validates_delay_but_not_lvdt_smoothing(self):
        base = {
            "current_event_time": 2.0,
            "previous_event_time": 1.0,
            "push_speed": 1.0,
            "time": np.arange(4.0),
            "reference_displacement_signal": np.arange(4.0),
        }
        for window in (0, True):
            with self.subTest(window=window):
                result = calculate_interevent_displacement_metrics(
                    **base, delay_sec=0.0, lvdt_smooth_w=window
                )
                self.assertTrue(result["D_reference"]["valid"])
        with self.assertRaisesRegex(ValueError, "delay_sec"):
            calculate_interevent_displacement_metrics(
                **base, delay_sec=np.nan, lvdt_smooth_w=0
            )

    def test_lvdt_smoothing_does_not_affect_reference_signal(self):
        time = np.arange(5.0)
        lvdt = np.array([0.0, 0.0, 0.0, 9.0, 9.0])
        reference = np.array([0.0, 0.0, 0.0, 9.0, 9.0])

        result = calculate_interevent_displacement_metrics(
            current_event_time=3.0,
            previous_event_time=1.0,
            push_speed=1.0,
            time=time,
            lvdt_signal=lvdt,
            reference_displacement_signal=reference,
            delay_sec=0.0,
            lvdt_smooth_w=3,
        )

        self.assertEqual(result["D_max"]["value"], 6.0)
        self.assertEqual(result["D_reference"]["value"], 9.0)


class CalculateEventDropMetricsTests(unittest.TestCase):
    def setUp(self):
        self.time = np.array([-2.0, -1.0, 1.0, 2.0])
        self.points = (-2.0, -1.0, 1.0, 2.0)

    def test_single_signal(self):
        signals = {
            "tau": np.where(self.time < 0, self.time + 8.0, self.time + 3.0)
        }
        parameters = {
            "tau": {"half_win": 2.0, "points": self.points, "smooth_w": None}
        }

        result = calculate_event_drop_metrics(
            self.time, 0.0, signals, parameters
        )

        self.assertEqual(list(result), ["tau"])
        self.assertTrue(result["tau"]["valid"])
        self.assertAlmostEqual(result["tau"]["delta"], 5.0)

    def test_multiple_signals_keep_input_order(self):
        signals = {
            "eddy_2": np.where(self.time < 0, self.time + 9.0, self.time + 4.0),
            "tau": np.where(self.time < 0, self.time + 8.0, self.time + 3.0),
            "mu": np.where(self.time < 0, -self.time + 2.0, -self.time + 6.0),
        }
        parameters = {
            name: {"half_win": 2.0, "points": self.points}
            for name in signals
        }

        result = calculate_event_drop_metrics(
            self.time, 0.0, signals, parameters
        )

        self.assertEqual(list(result), ["eddy_2", "tau", "mu"])
        self.assertAlmostEqual(result["eddy_2"]["delta"], 5.0)
        self.assertAlmostEqual(result["tau"]["delta"], 5.0)
        self.assertAlmostEqual(result["mu"]["delta"], -4.0)

    def test_one_invalid_signal_does_not_stop_other_signals(self):
        signals = {
            "invalid": np.array([1.0]),
            "valid": np.where(self.time < 0, self.time + 8.0, self.time + 3.0),
        }
        parameters = {
            name: {"half_win": 2.0, "points": self.points}
            for name in signals
        }

        result = calculate_event_drop_metrics(
            self.time, 0.0, signals, parameters
        )

        self.assertEqual(result["invalid"], {"valid": False})
        self.assertTrue(result["valid"]["valid"])
        self.assertAlmostEqual(result["valid"]["delta"], 5.0)

    def test_all_invalid(self):
        signals = {
            "short": np.array([1.0]),
            "missing_parameters": np.array([1.0, 2.0, 3.0, 4.0]),
        }
        parameters = {
            "short": {"half_win": 2.0, "points": self.points},
        }

        result = calculate_event_drop_metrics(
            self.time, 0.0, signals, parameters
        )

        self.assertEqual(
            result,
            {
                "short": {"valid": False},
                "missing_parameters": {"valid": False},
            },
        )

    def test_each_signal_uses_its_own_smoothing_and_fitting_windows(self):
        time = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
        signals = {
            "smoothed": np.array([8.0, 10.0, 9.0, 11.0, 4.0, 6.0, 5.0, 7.0]),
            "windowed": np.where(time < 0, 2.0 * time + 12.0, time + 5.0),
        }
        parameters = {
            "smoothed": {
                "half_win": 3.0,
                "points": (-3.0, -1.0, 1.0, 3.0),
                "smooth_w": 3,
            },
            "windowed": {
                "half_win": 2.0,
                "points": (-2.0, -0.5, 0.5, 2.0),
                "smooth_w": None,
            },
        }

        result = calculate_event_drop_metrics(
            time, 0.0, signals, parameters
        )
        expected_smoothed = calculate_event_signal_drop(
            time, signals["smoothed"], 0.0, **parameters["smoothed"]
        )
        expected_windowed = calculate_event_signal_drop(
            time, signals["windowed"], 0.0, **parameters["windowed"]
        )

        self.assertAlmostEqual(
            result["smoothed"]["delta"], expected_smoothed["delta"]
        )
        self.assertAlmostEqual(
            result["windowed"]["delta"], expected_windowed["delta"]
        )

    def test_calls_single_signal_helper_once_per_configured_signal(self):
        signals = {
            "tau": np.ones(4),
            "mu": np.ones(4),
            "eddy_1": np.ones(4),
        }
        parameters = {
            name: {"half_win": 2.0, "points": self.points}
            for name in signals
        }
        returned_results = [
            {"valid": True, "delta": float(index)}
            for index in range(len(signals))
        ]

        with mock.patch(
            "labquake_explorer.analysis.event_drop.calculate_event_signal_drop",
            side_effect=returned_results,
        ) as helper:
            result = calculate_event_drop_metrics(
                self.time, 0.0, signals, parameters
            )

        self.assertEqual(helper.call_count, len(signals))
        self.assertEqual(
            [result[name]["delta"] for name in signals],
            [0.0, 1.0, 2.0],
        )

    def test_inputs_are_not_modified_and_outputs_do_not_share_objects(self):
        shared_signal = np.where(
            self.time < 0, self.time + 8.0, self.time + 3.0
        )
        signals = {"tau": shared_signal, "mu": shared_signal}
        shared_parameters = {
            "half_win": 2.0,
            "points": list(self.points),
            "smooth_w": None,
        }
        parameters = {"tau": shared_parameters, "mu": shared_parameters}
        original_time = self.time.copy()
        original_signal = shared_signal.copy()
        original_points = shared_parameters["points"].copy()

        result = calculate_event_drop_metrics(
            self.time, 0.0, signals, parameters
        )
        result["tau"]["coeff_pre"][0] = 999.0

        np.testing.assert_array_equal(self.time, original_time)
        np.testing.assert_array_equal(shared_signal, original_signal)
        self.assertEqual(shared_parameters["points"], original_points)
        self.assertNotEqual(result["mu"]["coeff_pre"][0], 999.0)
        self.assertIsNot(result["tau"], result["mu"])
        self.assertIsNot(result["tau"]["coeff_pre"], result["mu"]["coeff_pre"])

    def test_top_level_inputs_must_be_mappings(self):
        with self.assertRaises(ValueError):
            calculate_event_drop_metrics(self.time, 0.0, [], {})
        with self.assertRaises(ValueError):
            calculate_event_drop_metrics(self.time, 0.0, {}, [])


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


class CalculateEventSignalDropTests(unittest.TestCase):
    def test_absolute_time_is_converted_and_baseline_is_subtracted(self):
        time = np.array([99.0, 99.5, 100.5, 101.0])
        relative_time = time - 100.0
        signal = np.where(
            relative_time < 0,
            relative_time + 12.0,
            2.0 * relative_time + 7.0,
        )

        result = calculate_event_signal_drop(
            time,
            signal,
            event_time=100.0,
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["val_pre_0"], 1.0)
        self.assertAlmostEqual(result["val_post_0"], -4.0)
        self.assertAlmostEqual(result["delta"], 5.0)

    def test_event_mask_includes_both_endpoints(self):
        time = np.array([-1.1, -1.0, -0.5, 0.5, 1.0, 1.1])
        signal = np.array([1000.0, 9.0, 9.5, 5.5, 6.0, -1000.0])

        result = calculate_event_signal_drop(
            time,
            signal,
            event_time=0.0,
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], 5.0)

    def test_smoothing_matches_existing_moving_average_pipeline(self):
        time = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        signal = np.array([10.0, 12.0, 11.0, 13.0, 5.0, 7.0, 6.0, 8.0])
        points = (-2.0, -0.5, 0.5, 2.0)
        smoothed = moving_average(signal, 3)
        expected = calculate_trend_drop(time, smoothed - smoothed[0], points)

        result = calculate_event_signal_drop(
            time,
            signal,
            event_time=0.0,
            half_win=2.0,
            points=points,
            smooth_w=3,
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], expected["delta"])
        np.testing.assert_allclose(result["coeff_pre"], expected["coeff_pre"])
        np.testing.assert_allclose(result["coeff_post"], expected["coeff_post"])

    def test_no_smoothing_preserves_negative_signed_drop(self):
        time = np.array([-2.0, -1.0, 1.0, 2.0])
        signal = np.where(time < 0, time + 3.0, -time + 8.0)

        result = calculate_event_signal_drop(
            time,
            signal,
            event_time=0.0,
            half_win=2.0,
            points=(-2.0, -1.0, 1.0, 2.0),
            smooth_w=None,
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], -5.0)

    def test_empty_event_window_returns_invalid(self):
        result = calculate_event_signal_drop(
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            event_time=10.0,
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
        )

        self.assertEqual(result, {"valid": False})

    def test_insufficient_fitting_samples_return_invalid(self):
        result = calculate_event_signal_drop(
            np.array([-1.0, 0.5, 1.0]),
            np.array([2.0, 1.0, 1.5]),
            event_time=0.0,
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
        )

        self.assertEqual(result, {"valid": False})

    def test_mismatched_lengths_raise_value_error(self):
        with self.assertRaises(ValueError):
            calculate_event_signal_drop(
                [0.0, 1.0],
                [1.0],
                event_time=0.0,
                half_win=1.0,
                points=(-1.0, -0.5, 0.5, 1.0),
            )

    def test_non_one_dimensional_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            calculate_event_signal_drop(
                [[-1.0, 1.0]],
                [[2.0, 1.0]],
                event_time=0.0,
                half_win=1.0,
                points=(-1.0, -0.5, 0.5, 1.0),
            )

    def test_nan_and_infinity_follow_trend_fit_validation(self):
        time = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        signal = np.where(time < 0, time + 10.0, -time + 6.0)
        signal[1] = np.nan
        signal[6] = np.inf

        result = calculate_event_signal_drop(
            time,
            signal,
            event_time=0.0,
            half_win=2.0,
            points=(-2.0, -0.5, 0.5, 2.0),
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["delta"], 4.0)

    def test_input_arrays_are_not_modified(self):
        time = np.array([-1.0, -0.5, 0.5, 1.0])
        signal = np.array([4.0, 5.0, 1.0, 2.0])
        original_time = time.copy()
        original_signal = signal.copy()

        calculate_event_signal_drop(
            time,
            signal,
            event_time=0.0,
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
            smooth_w=2,
        )

        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(signal, original_signal)

    def test_two_signals_are_analyzed_by_separate_calls(self):
        time = np.array([-2.0, -1.0, 1.0, 2.0])
        falling_signal = np.where(time < 0, time + 8.0, time + 3.0)
        rising_signal = np.where(time < 0, -time + 2.0, -time + 6.0)
        points = (-2.0, -1.0, 1.0, 2.0)

        falling_result = calculate_event_signal_drop(
            time, falling_signal, 0.0, 2.0, points
        )
        rising_result = calculate_event_signal_drop(
            time, rising_signal, 0.0, 2.0, points
        )

        self.assertAlmostEqual(falling_result["delta"], 5.0)
        self.assertAlmostEqual(rising_result["delta"], -4.0)


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
