"""Headless-safe tests for the tau/mu event-drop preview view."""

import copy
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import EventDropEditorView as ExportedEventDropEditorView
from labquake_explorer.ui.views.event_drop_editor_view import (
    EventDropEditorView,
    _DraggableVerticalLine,
    find_signal_candidates,
    format_preview_result,
    parse_preview_parameters,
    parse_smooth_window,
)


class SignalCandidateTests(unittest.TestCase):
    def test_aligned_numeric_array_and_list_are_candidates(self):
        event = {
            "time": np.array([0.0, 1.0, 2.0]),
            "array_signal": np.array([3.0, 4.0, 5.0]),
            "list_signal": [6, 7, 8],
        }

        candidates = find_signal_candidates(event)

        self.assertEqual(list(candidates), ["array_signal", "list_signal"])
        np.testing.assert_array_equal(candidates["list_signal"], [6, 7, 8])

    def test_time_is_not_a_candidate(self):
        event = {
            "time": np.array([0.0, 1.0]),
            "signal": np.array([2.0, 3.0]),
        }

        self.assertNotIn("time", find_signal_candidates(event))

    def test_two_dimensional_array_is_not_a_candidate(self):
        event = {
            "time": np.array([0.0, 1.0, 2.0]),
            "strain": np.ones((2, 3)),
        }

        self.assertEqual(find_signal_candidates(event), {})

    def test_wrong_length_array_is_not_a_candidate(self):
        event = {
            "time": np.array([0.0, 1.0, 2.0]),
            "short": np.array([1.0, 2.0]),
        }

        self.assertEqual(find_signal_candidates(event), {})

    def test_non_numeric_list_is_not_a_candidate(self):
        event = {
            "time": np.array([0.0, 1.0]),
            "labels": ["a", "b"],
        }

        self.assertEqual(find_signal_candidates(event), {})

    def test_complex_and_boolean_arrays_are_not_candidates(self):
        event = {
            "time": np.array([0.0, 1.0]),
            "complex_signal": np.array([1.0 + 2.0j, 3.0 + 4.0j]),
            "boolean_signal": np.array([True, False]),
        }

        self.assertEqual(find_signal_candidates(event), {})

    def test_candidate_filter_does_not_modify_or_copy_original_array(self):
        signal = np.array([1.0, 2.0])
        event = {
            "time": np.array([0.0, 1.0]),
            "signal": signal,
        }
        original = signal.copy()

        candidates = find_signal_candidates(event)

        self.assertIs(candidates["signal"], signal)
        np.testing.assert_array_equal(signal, original)

    def test_candidate_order_follows_event_without_name_sorting(self):
        event = {
            "time": np.array([0.0, 1.0]),
            "z_signal": np.array([1.0, 2.0]),
            "a_signal": np.array([3.0, 4.0]),
        }

        self.assertEqual(
            list(find_signal_candidates(event)),
            ["z_signal", "a_signal"],
        )


class ParameterParsingTests(unittest.TestCase):
    def test_blank_smooth_window_is_none(self):
        self.assertIsNone(parse_smooth_window(""))
        self.assertIsNone(parse_smooth_window("   "))

    def test_positive_integer_smooth_window_is_parsed(self):
        self.assertEqual(parse_smooth_window("100"), 100)

    def test_invalid_smooth_windows_are_rejected(self):
        for value in ("0", "-1", "1.5", "abc", "True"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_smooth_window(value)

    def test_preview_parameters_preserve_half_window_and_points(self):
        result = parse_preview_parameters(
            "2.5",
            "-1.2",
            "-0.4",
            "0.3",
            "1.1",
            "7",
        )

        self.assertEqual(result["half_win"], 2.5)
        self.assertEqual(result["points"], (-1.2, -0.4, 0.3, 1.1))
        self.assertEqual(result["smooth_w"], 7)

    def test_reversed_window_endpoints_are_accepted(self):
        result = parse_preview_parameters(
            "2.0",
            "-0.5",
            "-1.0",
            "1.0",
            "0.5",
            "",
        )

        self.assertEqual(result["points"], (-0.5, -1.0, 1.0, 0.5))

    def test_pre_window_must_be_at_or_before_event(self):
        with self.assertRaisesRegex(ValueError, "pre fitting window"):
            parse_preview_parameters("2", "-1", "0.1", "0.5", "1", "")

    def test_post_window_must_be_at_or_after_event(self):
        with self.assertRaisesRegex(ValueError, "post fitting window"):
            parse_preview_parameters("2", "-1", "-0.5", "-0.1", "1", "")

    def test_fitting_windows_must_have_nonzero_width(self):
        with self.assertRaisesRegex(ValueError, "pre fitting window.*non-zero"):
            parse_preview_parameters("2", "-1", "-1", "0.5", "1", "")
        with self.assertRaisesRegex(ValueError, "post fitting window.*non-zero"):
            parse_preview_parameters("2", "-1", "-0.5", "1", "1", "")

    def test_all_endpoints_must_be_inside_half_window(self):
        with self.assertRaisesRegex(ValueError, "within.*half_win"):
            parse_preview_parameters("1", "-1.1", "-0.5", "0.5", "1", "")

    def test_half_window_and_endpoints_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "half_win.*finite"):
            parse_preview_parameters("inf", "-1", "-0.5", "0.5", "1", "")
        with self.assertRaisesRegex(ValueError, "values must be finite"):
            parse_preview_parameters("2", "-1", "nan", "0.5", "1", "")


class PreviewResultFormattingTests(unittest.TestCase):
    def test_signed_delta_is_preserved_and_magnitude_is_separate(self):
        display = format_preview_result(
            {
                "valid": True,
                "delta": -4.5,
                "val_pre_0": 1.0,
                "val_post_0": 5.5,
            }
        )

        self.assertEqual(display["delta"], "-4.5")
        self.assertEqual(display["magnitude"], "4.5")

    def test_invalid_result_is_formatted_without_optional_keys(self):
        display = format_preview_result({"valid": False})

        self.assertEqual(display["valid"], "False")
        self.assertEqual(display["delta"], "—")
        self.assertEqual(display["status"], "Invalid fitting windows/data")


class PreviewCallTests(unittest.TestCase):
    @staticmethod
    def _parameter_vars(
        half_win="1.5",
        points=("-1", "-0.5", "0.5", "1"),
        smooth_w="9",
    ):
        return {
            "half_win": FakeVariable(half_win),
            "pre_start": FakeVariable(points[0]),
            "pre_end": FakeVariable(points[1]),
            "post_start": FakeVariable(points[2]),
            "post_end": FakeVariable(points[3]),
            "smooth_w": FakeVariable(smooth_w),
        }

    def make_view(self, event):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.event = event
        view.signal_candidates = find_signal_candidates(event)
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        view.preview_results = {}
        view.active_metric_role = None
        view.preview_parameters = None
        return view

    def test_headless_view_has_tau_mu_and_lvdt_metric_bindings(self):
        view = self.make_view({
            "time": np.array([0.0, 1.0]),
            "event_time": 0.5,
            "signal": np.array([1.0, 2.0]),
        })

        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_orchestrator_receives_tau_and_mu_once_in_role_order(self, calculator):
        event = {
            "time": np.array([9.0, 9.5, 10.5, 11.0]),
            "event_time": 10.0,
            "tau_signal": np.array([5.0, 6.0, 2.0, 3.0]),
            "mu_signal": np.array([0.5, 0.6, 0.2, 0.3]),
        }
        expected_results = {
            "tau": {"valid": True, "delta": 4.0},
            "mu": {"valid": False},
        }
        calculator.return_value = expected_results
        view = self.make_view(event)
        view.metric_bindings = {
            "tau": "tau_signal",
            "mu": "mu_signal",
            "lvdt": None,
        }
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()

        with mock.patch(
            "labquake_explorer.analysis.event_drop.calculate_event_signal_drop"
        ) as direct_calculator:
            results = view.calculate_preview()

        calculator.assert_called_once()
        call = calculator.call_args.kwargs
        self.assertIs(call["time"], event["time"])
        self.assertEqual(call["event_time"], 10.0)
        self.assertEqual(list(call["signals"]), ["tau", "mu"])
        self.assertIs(call["signals"]["tau"], event["tau_signal"])
        self.assertIs(call["signals"]["mu"], event["mu_signal"])
        self.assertEqual(list(call["parameters"]), ["tau", "mu"])
        self.assertIsNot(call["parameters"]["tau"], call["parameters"]["mu"])
        self.assertEqual(
            call["parameters"]["tau"],
            {
                "half_win": 1.5,
                "points": (-1.0, -0.5, 0.5, 1.0),
                "smooth_w": 9,
            },
        )
        self.assertEqual(call["parameters"]["mu"], call["parameters"]["tau"])
        direct_calculator.assert_not_called()
        self.assertIs(results, expected_results)
        self.assertIs(view.preview_results, expected_results)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_each_single_role_uses_domain_role_as_orchestration_key(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "shared": np.array([2.0, 1.0]),
        }
        calculator.return_value = {"mu": {"valid": False}}
        view = self.make_view(event)
        view.metric_bindings["mu"] = "shared"
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()

        view.calculate_preview()

        call = calculator.call_args.kwargs
        self.assertEqual(list(call["signals"]), ["mu"])
        self.assertIs(call["signals"]["mu"], event["shared"])
        self.assertEqual(list(call["parameters"]), ["mu"])

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_stale_tau_binding_is_rejected_before_orchestration(self, calculator):
        event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
            "signal": np.array([4.0, 5.0, 1.0, 2.0]),
        }
        view = self.make_view(event)
        view.metric_bindings = {"tau": "removed", "mu": "signal", "lvdt": None}

        with self.assertRaisesRegex(ValueError, "bound to tau"):
            view.calculate_preview()

        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_stale_mu_binding_is_rejected_without_partial_calculation(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
            "signal": np.array([4.0, 5.0, 1.0, 2.0]),
        }
        view = self.make_view(event)
        view.metric_bindings = {"tau": "signal", "mu": "removed", "lvdt": None}

        with self.assertRaisesRegex(ValueError, "bound to mu"):
            view.calculate_preview()

        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_stale_lvdt_binding_is_rejected_without_partial_calculation(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "signal": np.array([2.0, 1.0]),
        }
        view = self.make_view(event)
        view.metric_bindings = {
            "tau": "signal",
            "mu": None,
            "lvdt": "removed",
        }

        with self.assertRaisesRegex(ValueError, "bound to lvdt"):
            view.calculate_preview()

        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_all_roles_use_independent_parameter_groups(self, calculator):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "tau_source": np.array([3.0, 1.0]),
            "mu_source": np.array([0.3, 0.1]),
            "lvdt_source": np.array([8.0, 5.0]),
        }
        view = self.make_view(event)
        view.metric_bindings = {
            "tau": "tau_source",
            "mu": "mu_source",
            "lvdt": "lvdt_source",
        }
        view.parameter_vars = self._parameter_vars(
            half_win="1.5", smooth_w="3"
        )
        view.lvdt_parameter_vars = self._parameter_vars(
            half_win="2.5",
            points=("-2", "-1", "1", "2"),
            smooth_w="5",
        )
        calculator.return_value = {
            "tau": {"valid": True},
            "mu": {"valid": True},
            "lvdt": {"valid": True},
        }

        view.calculate_preview()

        call = calculator.call_args.kwargs
        self.assertEqual(list(call["signals"]), ["tau", "mu", "lvdt"])
        self.assertIs(call["signals"]["tau"], event["tau_source"])
        self.assertIs(call["signals"]["mu"], event["mu_source"])
        self.assertIs(call["signals"]["lvdt"], event["lvdt_source"])
        self.assertEqual(list(call["parameters"]), ["tau", "mu", "lvdt"])
        self.assertIsNot(call["parameters"]["tau"], call["parameters"]["mu"])
        self.assertIsNot(call["parameters"]["tau"], call["parameters"]["lvdt"])
        self.assertIsNot(call["parameters"]["mu"], call["parameters"]["lvdt"])
        self.assertEqual(call["parameters"]["tau"]["half_win"], 1.5)
        self.assertEqual(call["parameters"]["mu"]["smooth_w"], 3)
        self.assertEqual(call["parameters"]["lvdt"]["half_win"], 2.5)
        self.assertEqual(
            call["parameters"]["lvdt"]["points"], (-2.0, -1.0, 1.0, 2.0)
        )
        self.assertEqual(call["parameters"]["lvdt"]["smooth_w"], 5)
        calculator.assert_called_once()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_unbound_parameter_group_is_not_parsed(self, calculator):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "signal": np.array([2.0, 1.0]),
        }
        view = self.make_view(event)
        view.metric_bindings["tau"] = "signal"
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars(half_win="invalid")
        calculator.return_value = {"tau": {"valid": False}}

        view.calculate_preview()

        calculator.assert_called_once()

        view.metric_bindings = {"tau": None, "mu": None, "lvdt": "signal"}
        view.parameter_vars = self._parameter_vars(half_win="invalid")
        view.lvdt_parameter_vars = self._parameter_vars()
        calculator.reset_mock()
        calculator.return_value = {"lvdt": {"valid": False}}

        view.calculate_preview()

        calculator.assert_called_once()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_invalid_bound_parameter_group_prevents_orchestration(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "signal": np.array([2.0, 1.0]),
        }
        view = self.make_view(event)
        view.metric_bindings = {
            "tau": "signal",
            "mu": None,
            "lvdt": "signal",
        }
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars(half_win="invalid")

        with self.assertRaises(ValueError):
            view.calculate_preview()

        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_preview_does_not_modify_event(self, calculator):
        event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
            "signal": np.array([4.0, 5.0, 1.0, 2.0]),
        }
        original = copy.deepcopy(event)
        calculator.return_value = {"tau": {"valid": False}}
        view = self.make_view(event)
        view.metric_bindings["tau"] = "signal"
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()

        view.calculate_preview()

        self.assertEqual(event.keys(), original.keys())
        np.testing.assert_array_equal(event["time"], original["time"])
        np.testing.assert_array_equal(event["signal"], original["signal"])
        self.assertEqual(event["event_time"], original["event_time"])

    def test_preview_has_no_data_manager_persistence_call(self):
        event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
            "signal": np.array([4.0, 5.0, 1.0, 2.0]),
        }
        view = self.make_view(event)
        view.data_manager = mock.Mock()
        view.metric_bindings["tau"] = "signal"
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()

        view.calculate_preview()

        view.data_manager.set_data.assert_not_called()

    def test_recompute_formats_valid_and_invalid_metrics_independently(self):
        view = self.make_view({
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "tau_signal": np.array([2.0, 1.0]),
            "mu_signal": np.array([0.2, 0.1]),
        })
        view.metric_bindings = {
            "tau": "tau_signal",
            "mu": "mu_signal",
            "lvdt": "mu_signal",
        }
        view.result_vars = {
            role: {
                key: FakeVariable()
                for key in (
                    "valid",
                    "delta",
                    "magnitude",
                    "val_pre_0",
                    "val_post_0",
                )
            }
            for role in ("tau", "mu", "lvdt")
        }
        view.status_var = FakeVariable()
        view.calculate_preview = mock.Mock(return_value={
            "tau": {
                "valid": True,
                "delta": -2.5,
                "val_pre_0": 1.0,
                "val_post_0": 3.5,
            },
            "mu": {"valid": False},
            "lvdt": {
                "valid": True,
                "delta": 0.25,
                "val_pre_0": 0.5,
                "val_post_0": 0.25,
            },
        })
        view._plot_preview = mock.Mock()

        view.recompute_preview()

        view.calculate_preview.assert_called_once_with()
        self.assertEqual(view.result_vars["tau"]["valid"].get(), "True")
        self.assertEqual(view.result_vars["tau"]["delta"].get(), "-2.5")
        self.assertEqual(view.result_vars["tau"]["magnitude"].get(), "2.5")
        self.assertEqual(view.result_vars["mu"]["valid"].get(), "False")
        self.assertEqual(view.result_vars["mu"]["delta"].get(), "—")
        self.assertEqual(view.result_vars["lvdt"]["valid"].get(), "True")
        self.assertEqual(view.result_vars["lvdt"]["delta"].get(), "0.25")
        self.assertEqual(view.result_vars["lvdt"]["magnitude"].get(), "0.25")
        self.assertEqual(view.status_var.get(), "Preview only — not saved")
        view._plot_preview.assert_called_once_with()


class PreviewPlotTests(unittest.TestCase):
    def test_reversed_endpoints_use_normalized_fit_and_shaded_ranges(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = mock.Mock()
        view.event = {
            "time": np.array([-2.0, -1.0, 1.0, 2.0]),
            "event_time": 0.0,
        }
        view.signal_candidates = {
            "signal": np.array([5.0, 6.0, 1.0, 2.0]),
        }
        view.metric_bindings = {"tau": "signal", "mu": None, "lvdt": None}
        view.active_metric_role = "tau"
        view.parameter_vars = {
            "half_win": FakeVariable("2"),
            "pre_start": FakeVariable("-0.5"),
            "pre_end": FakeVariable("-1.5"),
            "post_start": FakeVariable("1.5"),
            "post_end": FakeVariable("0.5"),
            "smooth_w": FakeVariable(""),
        }
        view.lvdt_parameter_vars = {
            key: FakeVariable(variable.get())
            for key, variable in view.parameter_vars.items()
        }
        view.preview_results = {
            "tau": {
                "valid": True,
                "coeff_pre": np.array([1.0, 4.0]),
                "coeff_post": np.array([2.0, 1.0]),
                "delta": 3.0,
                "val_pre_0": 4.0,
                "val_post_0": 1.0,
            }
        }

        view._plot_preview()

        fit_lines = {line.get_label(): line for line in view.fit_ax.lines}
        pre_x = fit_lines["Pre fit"].get_xdata()
        post_x = fit_lines["Post fit"].get_xdata()
        self.assertEqual(pre_x[0], -1.5)
        self.assertEqual(pre_x[-1], 0.0)
        self.assertEqual(post_x[0], 0.0)
        self.assertEqual(post_x[-1], 1.5)

        shaded_ranges = []
        for patch in view.raw_ax.patches:
            start = float(patch.get_x())
            shaded_ranges.append((start, start + float(patch.get_width())))
        self.assertIn((-1.5, -0.5), shaded_ranges)
        self.assertIn((0.5, 1.5), shaded_ranges)

    def test_active_metric_controls_raw_signal_and_fit_result(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = mock.Mock()
        view.event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
        }
        tau = np.array([10.0, 11.0, 7.0, 8.0])
        mu = np.array([1.0, 1.1, 0.7, 0.8])
        view.signal_candidates = {"tau_source": tau, "mu_source": mu}
        view.metric_bindings = {"tau": "tau_source", "mu": "mu_source"}
        view.active_metric_role = "mu"
        view.preview_results = {
            "tau": {
                "valid": True,
                "coeff_pre": np.array([1.0, 10.0]),
                "coeff_post": np.array([1.0, 7.0]),
            },
            "mu": {
                "valid": True,
                "coeff_pre": np.array([2.0, 1.0]),
                "coeff_post": np.array([3.0, 0.7]),
            },
        }
        view.parameter_vars = {
            "half_win": FakeVariable("1"),
            "pre_start": FakeVariable("-1"),
            "pre_end": FakeVariable("-0.5"),
            "post_start": FakeVariable("0.5"),
            "post_end": FakeVariable("1"),
            "smooth_w": FakeVariable(""),
        }

        view._plot_preview()

        np.testing.assert_array_equal(view.raw_ax.lines[0].get_ydata(), mu)
        fit_lines = {line.get_label(): line for line in view.fit_ax.lines}
        self.assertAlmostEqual(fit_lines["Pre fit"].get_ydata()[-1], 1.0)
        self.assertAlmostEqual(fit_lines["Post fit"].get_ydata()[0], 0.7)

    def test_active_lvdt_uses_lvdt_signal_result_and_windows(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = mock.Mock()
        view.event = {
            "time": np.array([-2.0, -1.0, 1.0, 2.0]),
            "event_time": 0.0,
        }
        lvdt = np.array([8.0, 9.0, 4.0, 5.0])
        view.signal_candidates = {"lvdt_source": lvdt}
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": "lvdt_source"}
        view.active_metric_role = "lvdt"
        view.preview_results = {
            "lvdt": {
                "valid": True,
                "coeff_pre": np.array([1.0, 8.0]),
                "coeff_post": np.array([2.0, 4.0]),
            }
        }
        view.parameter_vars = PreviewCallTests._parameter_vars()
        view.lvdt_parameter_vars = PreviewCallTests._parameter_vars(
            half_win="2",
            points=("-1.75", "-0.75", "0.25", "1.5"),
            smooth_w="",
        )

        view._plot_preview()

        np.testing.assert_array_equal(view.raw_ax.lines[0].get_ydata(), lvdt)
        shaded = [
            (
                float(patch.get_x()),
                float(patch.get_x()) + float(patch.get_width()),
            )
            for patch in view.raw_ax.patches
        ]
        self.assertIn((-1.75, -0.75), shaded)
        self.assertIn((0.25, 1.5), shaded)
        fit_lines = {line.get_label(): line for line in view.fit_ax.lines}
        self.assertAlmostEqual(fit_lines["Pre fit"].get_ydata()[-1], 8.0)
        self.assertAlmostEqual(fit_lines["Post fit"].get_ydata()[0], 4.0)


class EventSwitchingTests(unittest.TestCase):
    def test_set_event_reloads_canonical_event_and_candidates(self):
        first_event = {
            "time": np.array([0.0, 1.0]),
            "event_time": 0.5,
            "first": np.array([1.0, 2.0]),
        }
        second_event = {
            "time": np.array([2.0, 3.0, 4.0]),
            "event_time": 3.0,
            "second": [4.0, 5.0, 6.0],
        }
        events = {0: first_event, 1: second_event}

        class FakeDataManager:
            def __init__(self):
                self.paths = []

            def get_data(self, path):
                self.paths.append(path)
                index = int(path.rsplit("[", 1)[1][:-1])
                return events[index]

            def set_data(self, *args, **kwargs):
                raise AssertionError("set_data must not be called")

        view = EventDropEditorView.__new__(EventDropEditorView)
        view.run_idx = 2
        view.data_manager = FakeDataManager()
        view.preview_results = {"tau": {"valid": True}}
        view.preview_parameters = {"half_win": 1.0}
        view.metric_bindings = {"tau": "first", "mu": "first"}
        view.active_metric_role = "tau"

        view._set_event(1)

        self.assertIs(view.event, second_event)
        self.assertEqual(list(view.signal_candidates), ["second"])
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)
        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )
        self.assertIsNone(view.active_metric_role)
        self.assertEqual(
            view.data_manager.paths,
            ["runs/[2]/events/[1]"],
        )


class FakeWidget:
    def __init__(self, value=""):
        self.value = value
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeVariable:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class DraggableWindowTests(unittest.TestCase):
    def make_plot_view(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = mock.Mock()
        view.event = {
            "time": np.array([-2.0, -1.0, 1.0, 2.0]),
            "event_time": 0.0,
        }
        view.signal_candidates = {
            "signal": np.array([5.0, 6.0, 1.0, 2.0]),
        }
        view.metric_bindings = {"tau": "signal", "mu": None, "lvdt": None}
        view.active_metric_role = "tau"
        view.tau_signal_combobox = FakeWidget("signal")
        view.mu_signal_combobox = FakeWidget()
        view.lvdt_signal_combobox = FakeWidget()
        view.parameter_vars = {
            "half_win": FakeVariable("2"),
            "pre_start": FakeVariable("-0.5"),
            "pre_end": FakeVariable("-1.5"),
            "post_start": FakeVariable("1.5"),
            "post_end": FakeVariable("0.5"),
            "smooth_w": FakeVariable(""),
        }
        view.lvdt_parameter_vars = {
            key: FakeVariable(variable.get())
            for key, variable in view.parameter_vars.items()
        }
        view.result_vars = {
            role: {
                key: FakeVariable("old")
                for key in (
                    "valid",
                    "delta",
                    "magnitude",
                    "val_pre_0",
                    "val_post_0",
                )
            }
            for role in ("tau", "mu", "lvdt")
        }
        view.status_var = FakeVariable()
        view.preview_results = {}
        view.preview_parameters = None
        view._endpoint_draggables = {}
        view._active_endpoint = None
        return view

    def test_four_endpoint_lines_follow_controls_without_reordering(self):
        view = self.make_plot_view()

        view._plot_preview()

        self.assertEqual(
            list(view._endpoint_draggables),
            ["pre_start", "pre_end", "post_start", "post_end"],
        )
        positions = {
            key: float(draggable.line.get_xdata()[0])
            for key, draggable in view._endpoint_draggables.items()
        }
        self.assertEqual(
            positions,
            {
                "pre_start": -0.5,
                "pre_end": -1.5,
                "post_start": 1.5,
                "post_end": 0.5,
            },
        )
        view._disconnect_endpoint_lines()

    def test_zero_width_controls_keep_overlapping_lines_available(self):
        view = self.make_plot_view()
        view.parameter_vars["pre_start"].set("-1")
        view.parameter_vars["pre_end"].set("-1")

        view._plot_preview()

        self.assertEqual(
            float(view._endpoint_draggables["pre_start"].line.get_xdata()[0]),
            -1.0,
        )
        self.assertEqual(
            float(view._endpoint_draggables["pre_end"].line.get_xdata()[0]),
            -1.0,
        )
        view._disconnect_endpoint_lines()

    def test_pre_post_and_half_window_constraints(self):
        view = self.make_plot_view()

        self.assertEqual(view._constrain_endpoint("pre_start", 1.0), 0.0)
        self.assertEqual(view._constrain_endpoint("post_end", -1.0), 0.0)
        self.assertEqual(view._constrain_endpoint("pre_end", -3.0), -2.0)
        self.assertEqual(view._constrain_endpoint("post_start", 3.0), 2.0)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_endpoint_change_updates_control_and_invalidates_without_analysis(
        self, calculator
    ):
        view = self.make_plot_view()
        view.preview_results = {"tau": {"valid": True}}
        view.preview_parameters = {"half_win": 2.0}
        view.data_manager = mock.Mock()

        view._on_endpoint_changed("pre_start", -0.83456789)

        self.assertEqual(view.parameter_vars["pre_start"].get(), "-0.834568")
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)
        self.assertEqual(
            view.status_var.get(),
            "Fitting windows changed — recompute preview",
        )
        self.assertTrue(
            all(
                variable.get() == "—"
                for role in ("tau", "mu")
                for variable in view.result_vars[role].values()
            )
        )
        self.assertTrue(
            all(
                variable.get() == "old"
                for variable in view.result_vars["lvdt"].values()
            )
        )
        calculator.assert_not_called()
        view.data_manager.set_data.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_lvdt_endpoint_only_invalidates_lvdt_result(self, calculator):
        view = self.make_plot_view()
        view.metric_bindings["lvdt"] = "signal"
        view.active_metric_role = "lvdt"
        view.preview_results = {
            "tau": {"valid": True},
            "mu": {"valid": True},
            "lvdt": {"valid": True},
        }
        view.data_manager = mock.Mock()

        view._on_endpoint_changed("post_end", 1.234567)

        self.assertEqual(view.lvdt_parameter_vars["post_end"].get(), "1.23457")
        self.assertEqual(
            view.preview_results,
            {"tau": {"valid": True}, "mu": {"valid": True}},
        )
        self.assertEqual(view.metric_bindings["lvdt"], "signal")
        calculator.assert_not_called()
        view.data_manager.set_data.assert_not_called()

    def test_redraw_disconnects_old_callbacks_before_recreating_lines(self):
        view = self.make_plot_view()

        view._plot_preview()
        old_draggables = list(view._endpoint_draggables.values())
        view._plot_preview()

        self.assertTrue(all(not draggable.connected for draggable in old_draggables))
        self.assertEqual(len(view._endpoint_draggables), 4)
        self.assertTrue(
            all(draggable.connected for draggable in view._endpoint_draggables.values())
        )
        view._disconnect_endpoint_lines()

    def test_event_redraw_uses_current_controls(self):
        view = self.make_plot_view()
        view.parameter_vars["pre_start"].set("-0.75")
        view.parameter_vars["post_end"].set("1.25")

        view._plot_preview()

        self.assertEqual(
            float(view._endpoint_draggables["pre_start"].line.get_xdata()[0]),
            -0.75,
        )
        self.assertEqual(
            float(view._endpoint_draggables["post_end"].line.get_xdata()[0]),
            1.25,
        )
        view._disconnect_endpoint_lines()

    def test_event_switch_keeps_controls_and_rebuilds_endpoint_lines(self):
        view = self.make_plot_view()
        view.run_idx = 4
        view.preview_button = FakeWidget()
        view.data_manager = mock.Mock()
        view.data_manager.get_data.return_value = {
            "time": np.array([-3.0, -1.0, 1.0, 3.0]),
            "event_time": 0.0,
            "new_signal": np.array([8.0, 9.0, 2.0, 3.0]),
        }
        original_controls = {
            key: variable.get()
            for key, variable in view.parameter_vars.items()
        }

        view._set_event(2)
        view._refresh_event_widgets()

        self.assertEqual(
            {
                key: variable.get()
                for key, variable in view.parameter_vars.items()
            },
            original_controls,
        )
        self.assertEqual(view.tau_signal_combobox.get(), "")
        self.assertEqual(view.mu_signal_combobox.get(), "")
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )
        self.assertIsNone(view.active_metric_role)
        self.assertEqual(
            float(view._endpoint_draggables["pre_start"].line.get_xdata()[0]),
            -0.5,
        )
        self.assertEqual(
            float(view._endpoint_draggables["post_end"].line.get_xdata()[0]),
            0.5,
        )
        view.data_manager.set_data.assert_not_called()
        view._disconnect_endpoint_lines()

    def test_signal_switch_keeps_fitting_window_controls(self):
        view = self.make_plot_view()
        view.preview_button = FakeWidget()
        view._plot_preview = mock.Mock()
        original = {
            key: variable.get()
            for key, variable in view.parameter_vars.items()
        }

        view.on_tau_signal_changed()

        self.assertEqual(
            {
                key: variable.get()
                for key, variable in view.parameter_vars.items()
            },
            original,
        )

    def test_close_disconnects_callbacks_and_removes_view(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        draggable = mock.Mock()
        view._endpoint_draggables = {"pre_start": draggable}
        view.parent = mock.Mock()
        view.parent.child_windows = [view]
        view.destroy = mock.Mock()

        view.on_close()

        draggable.disconnect.assert_called_once_with()
        self.assertEqual(view.parent.child_windows, [])
        view.destroy.assert_called_once_with()

    def test_mouse_must_be_in_axes_and_near_line_to_start_drag(self):
        figure = Figure()
        axes = figure.add_subplot(111)
        line = axes.axvline(-0.5)
        figure.canvas.draw()
        changed = mock.Mock()
        released = mock.Mock()
        draggable = _DraggableVerticalLine(
            line,
            on_changed=changed,
            on_released=released,
            constrain=lambda value: value,
        )
        line_pixel_x = axes.transData.transform((-0.5, 0.0))[0]

        outside_axes = mock.Mock(
            inaxes=None,
            xdata=-0.5,
            x=line_pixel_x,
            button=1,
        )
        draggable._on_press(outside_axes)
        self.assertFalse(draggable.dragging)

        far_from_line = mock.Mock(
            inaxes=axes,
            xdata=0.5,
            x=line_pixel_x + 100,
            button=1,
        )
        draggable._on_press(far_from_line)
        self.assertFalse(draggable.dragging)

        near_line = mock.Mock(
            inaxes=axes,
            xdata=-0.5,
            x=line_pixel_x + 3,
            button=1,
        )
        draggable._on_press(near_line)
        self.assertTrue(draggable.dragging)
        draggable.disconnect()

    def test_motion_updates_line_and_release_calls_once(self):
        figure = Figure()
        axes = figure.add_subplot(111)
        line = axes.axvline(-0.5)
        changed = mock.Mock()
        released = mock.Mock()
        draggable = _DraggableVerticalLine(
            line,
            on_changed=changed,
            on_released=released,
            constrain=lambda value: min(value, 0.0),
        )
        draggable.dragging = True
        motion = mock.Mock(inaxes=axes, xdata=0.75)

        draggable._on_motion(motion)
        draggable._on_release(mock.Mock())
        draggable._on_release(mock.Mock())

        self.assertEqual(float(line.get_xdata()[0]), 0.0)
        changed.assert_called_once_with(0.0)
        released.assert_called_once_with()
        draggable.disconnect()

    def test_only_one_overlapping_line_can_own_a_drag(self):
        view = self.make_plot_view()

        self.assertTrue(view._begin_endpoint_drag("pre_start"))
        self.assertFalse(view._begin_endpoint_drag("pre_end"))
        view._plot_preview = mock.Mock()
        view._on_endpoint_released("pre_start")
        self.assertIsNone(view._active_endpoint)
        view._plot_preview.assert_called_once_with()


class SignalSelectionStateTests(unittest.TestCase):
    def make_view(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.signal_candidates = {
            "first": np.array([1.0, 2.0]),
            "second": np.array([3.0, 4.0]),
        }
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        view.active_metric_role = None
        view.tau_signal_combobox = FakeWidget()
        view.mu_signal_combobox = FakeWidget()
        view.lvdt_signal_combobox = FakeWidget()
        view.preview_button = FakeWidget()
        view.status_var = FakeVariable()
        view.result_vars = {
            role: {
                "valid": FakeVariable(),
                "delta": FakeVariable(),
            }
            for role in ("tau", "mu", "lvdt")
        }
        view.preview_results = {
            "tau": {"valid": True},
            "mu": {"valid": True},
        }
        view.preview_parameters = {"half_win": 1.0}
        view._plot_preview = mock.Mock()
        return view

    def test_candidates_do_not_cause_automatic_signal_selection(self):
        view = self.make_view()

        view._refresh_event_widgets()

        self.assertEqual(
            view.tau_signal_combobox.options["values"], ["first", "second"]
        )
        self.assertEqual(
            view.mu_signal_combobox.options["values"], ["first", "second"]
        )
        self.assertEqual(view.tau_signal_combobox.get(), "")
        self.assertEqual(view.mu_signal_combobox.get(), "")
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )

    def test_tau_selection_only_updates_tau_and_does_not_analyze(self):
        view = self.make_view()
        view.tau_signal_combobox.set("second")

        view.on_tau_signal_changed()

        self.assertEqual(view.preview_button.options["state"], "normal")
        self.assertEqual(
            view.metric_bindings, {"tau": "second", "mu": None, "lvdt": None}
        )
        self.assertEqual(view.preview_results, {"mu": {"valid": True}})
        self.assertEqual(view.active_metric_role, "tau")
        self.assertIsNone(view.preview_parameters)

    def test_mu_selection_only_updates_mu_and_same_signal_is_allowed(self):
        view = self.make_view()
        view.metric_bindings["tau"] = "first"
        view.tau_signal_combobox.set("first")
        view.mu_signal_combobox.set("first")

        view.on_mu_signal_changed()

        self.assertEqual(
            view.metric_bindings,
            {"tau": "first", "mu": "first", "lvdt": None},
        )
        self.assertEqual(view.active_metric_role, "mu")
        self.assertEqual(view.preview_button.options["state"], "normal")

    def test_clearing_one_role_preserves_other_binding_and_result(self):
        view = self.make_view()
        view.metric_bindings = {
            "tau": "first",
            "mu": "second",
            "lvdt": None,
        }
        view.tau_signal_combobox.set("first")
        view.mu_signal_combobox.set("missing")
        view.active_metric_role = "mu"

        view.on_mu_signal_changed()

        self.assertEqual(
            view.metric_bindings, {"tau": "first", "mu": None, "lvdt": None}
        )
        self.assertEqual(view.preview_results, {"tau": {"valid": True}})
        self.assertEqual(view.active_metric_role, "tau")
        self.assertEqual(view.preview_button.options["state"], "normal")

    def test_clearing_both_roles_disables_preview(self):
        view = self.make_view()
        view.metric_bindings = {"tau": "first", "mu": None, "lvdt": None}
        view.preview_results = {"tau": {"valid": True}}
        view.tau_signal_combobox.set("")

        view.on_tau_signal_changed()

        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)

    def test_lvdt_selection_only_updates_lvdt_and_preserves_other_results(self):
        view = self.make_view()
        view.metric_bindings["tau"] = "first"
        view.tau_signal_combobox.set("first")
        view.lvdt_signal_combobox.set("first")

        view.on_lvdt_signal_changed()

        self.assertEqual(
            view.metric_bindings,
            {"tau": "first", "mu": None, "lvdt": "first"},
        )
        self.assertEqual(
            view.preview_results,
            {"tau": {"valid": True}, "mu": {"valid": True}},
        )
        self.assertEqual(view.active_metric_role, "lvdt")
        self.assertEqual(view.preview_button.options["state"], "normal")


class WiringTests(unittest.TestCase):
    def test_view_is_exported_from_views_package(self):
        self.assertIs(ExportedEventDropEditorView, EventDropEditorView)

    def test_labquake_explorer_exposes_analyze_event_drop(self):
        self.assertTrue(callable(LabquakeExplorer.analyze_event_drop))

    def test_run_event_parser_accepts_forward_slashes(self):
        self.assertEqual(
            LabquakeExplorer._extract_run_event_indices(
                "runs/[3]/events/[7]"
            ),
            (3, 7),
        )

    def test_run_event_parser_accepts_windows_backslashes(self):
        self.assertEqual(
            LabquakeExplorer._extract_run_event_indices(
                r"runs\[3]\events\[7]"
            ),
            (3, 7),
        )

    def test_run_event_parser_accepts_multi_digit_indices(self):
        self.assertEqual(
            LabquakeExplorer._extract_run_event_indices(
                "runs/[12]/events/[345]"
            ),
            (12, 345),
        )

    def test_run_event_parser_rejects_invalid_path_clearly(self):
        with self.assertRaisesRegex(ValueError, "Expected a tree path"):
            LabquakeExplorer._extract_run_event_indices(
                "runs/not-an-index/events/[7]"
            )

    @mock.patch("labquake_explorer.ui.labquake_explorer.EventDropEditorView")
    def test_analyze_event_drop_accepts_windows_path_without_analysis_or_save(
        self, view_class
    ):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(
            return_value=(r"runs\[3]\events\[7]", "[7]")
        )
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        explorer.data_manager = mock.Mock()
        view = mock.Mock()
        view_class.return_value = view

        explorer.analyze_event_drop()

        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])
        explorer.data_manager.set_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
