"""Headless-safe tests for scalar and ordered slip event-drop previews."""

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
    format_interevent_result,
    format_preview_result,
    parse_interevent_parameters,
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
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)

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
    def test_scalar_and_slip_mappings_keep_order_identity_and_parameters(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "tau_source": np.array([3.0, 1.0]),
            "slip_a": np.array([8.0, 5.0]),
            "slip_b": np.array([7.0, 4.0]),
        }
        view = self.make_view(event)
        view.metric_bindings["tau"] = "tau_source"
        view.slip_bindings = ["slip_a", None, "slip_b"]
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()
        view.slip_parameter_vars = self._parameter_vars(
            half_win="2", points=("-2", "-1", "1", "2"), smooth_w=""
        )
        calculator.return_value = {
            "tau": {"valid": True},
            "slip_1": {"valid": True},
            "slip_3": {"valid": False},
        }

        view.calculate_preview()

        call = calculator.call_args.kwargs
        self.assertEqual(list(call["signals"]), ["tau", "slip_1", "slip_3"])
        self.assertIs(call["signals"]["slip_1"], event["slip_a"])
        self.assertIs(call["signals"]["slip_3"], event["slip_b"])
        self.assertEqual(list(call["parameters"]), ["tau", "slip_1", "slip_3"])
        self.assertIsNot(
            call["parameters"]["slip_1"], call["parameters"]["slip_3"]
        )
        self.assertIsNone(call["parameters"]["slip_1"]["smooth_w"])
        self.assertEqual(call["parameters"]["slip_3"]["half_win"], 2.0)
        calculator.assert_called_once()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_stale_slip_and_invalid_bound_slip_parameters_stop_all_analysis(
        self, calculator
    ):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "tau_source": np.array([3.0, 1.0]),
        }
        view = self.make_view(event)
        view.metric_bindings["tau"] = "tau_source"
        view.slip_bindings = [None, "missing"]

        with self.assertRaisesRegex(ValueError, "bound to slip_2"):
            view.calculate_preview()
        calculator.assert_not_called()

        view.slip_bindings = ["tau_source"]
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()
        view.slip_parameter_vars = self._parameter_vars(half_win="invalid")
        with self.assertRaises(ValueError):
            view.calculate_preview()
        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_empty_slip_rows_do_not_parse_slip_parameters(self, calculator):
        event = {
            "time": np.array([-1.0, 1.0]),
            "event_time": 0.0,
            "tau_source": np.array([3.0, 1.0]),
        }
        view = self.make_view(event)
        view.metric_bindings["tau"] = "tau_source"
        view.slip_bindings = [None]
        view.parameter_vars = self._parameter_vars()
        view.lvdt_parameter_vars = self._parameter_vars()
        view.slip_parameter_vars = self._parameter_vars(half_win="invalid")
        calculator.return_value = {"tau": {"valid": False}}

        view.calculate_preview()

        calculator.assert_called_once()

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
        view.slip_bindings = ["tau_signal", None]
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
        view.slip_rows = [
            {
                "result_vars": {
                    key: FakeVariable()
                    for key in (
                        "valid",
                        "delta",
                        "magnitude",
                        "val_pre_0",
                        "val_post_0",
                    )
                }
            },
            {
                "result_vars": {
                    key: FakeVariable()
                    for key in (
                        "valid",
                        "delta",
                        "magnitude",
                        "val_pre_0",
                        "val_post_0",
                    )
                }
            },
        ]
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
            "slip_1": {
                "valid": True,
                "delta": -1.25,
                "val_pre_0": 2.0,
                "val_post_0": 3.25,
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
        self.assertEqual(view.slip_rows[0]["result_vars"]["delta"].get(), "-1.25")
        self.assertEqual(
            view.slip_rows[0]["result_vars"]["magnitude"].get(), "1.25"
        )
        self.assertEqual(view.slip_rows[1]["result_vars"]["valid"].get(), "—")
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

    def test_active_slip_uses_selected_signal_result_and_slip_windows(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = mock.Mock()
        view.event = {
            "time": np.array([-2.0, -1.0, 1.0, 2.0]),
            "event_time": 0.0,
        }
        slip = np.array([9.0, 10.0, 5.0, 6.0])
        view.signal_candidates = {"chosen": slip}
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        view.slip_bindings = ["chosen"]
        view.active_metric_role = "slip_1"
        view.preview_results = {
            "slip_1": {
                "valid": True,
                "coeff_pre": np.array([1.0, 9.0]),
                "coeff_post": np.array([2.0, 5.0]),
            }
        }
        view.parameter_vars = PreviewCallTests._parameter_vars()
        view.lvdt_parameter_vars = PreviewCallTests._parameter_vars()
        view.slip_parameter_vars = PreviewCallTests._parameter_vars(
            half_win="2",
            points=("-1.8", "-0.8", "0.4", "1.6"),
            smooth_w="",
        )

        view._plot_preview()

        np.testing.assert_array_equal(view.raw_ax.lines[0].get_ydata(), slip)
        shaded = [
            (
                float(patch.get_x()),
                float(patch.get_x()) + float(patch.get_width()),
            )
            for patch in view.raw_ax.patches
        ]
        self.assertIn((-1.8, -0.8), shaded)
        self.assertIn((0.4, 1.6), shaded)
        fit_lines = {line.get_label(): line for line in view.fit_ax.lines}
        self.assertAlmostEqual(fit_lines["Pre fit"].get_ydata()[-1], 9.0)
        self.assertAlmostEqual(fit_lines["Post fit"].get_ydata()[0], 5.0)


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
        events = [first_event, second_event]
        run_data = {
            "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "events": events,
            "second": np.arange(5.0),
        }

        class FakeDataManager:
            def __init__(self):
                self.paths = []

            def get_data(self, path):
                self.paths.append(path)
                if path == "runs/[2]":
                    return run_data
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
        view.slip_bindings = ["first"]
        view.active_metric_role = "tau"

        view._set_event(1)

        self.assertIs(view.event, second_event)
        self.assertEqual(list(view.signal_candidates), ["second"])
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)
        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )
        self.assertEqual(view.slip_bindings, [])
        self.assertIsNone(view.active_metric_role)
        self.assertEqual(
            view.data_manager.paths,
            ["runs/[2]/events/[1]", "runs/[2]"],
        )
        self.assertIs(view.run_data, run_data)
        self.assertEqual(view.current_event_time, 3.0)
        self.assertEqual(view.previous_event_time, 0.5)
        self.assertEqual(view.full_run_signal_candidates, ["second"])


class IntereventContextResolutionTests(unittest.TestCase):
    @staticmethod
    def make_view(
        *,
        event=None,
        events=None,
        event_idx=0,
        run_data=None,
    ):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.run_idx = 3
        view.event_idx = event_idx
        view.event = event if event is not None else {}
        view.events = events
        view.run_data = run_data
        view.metric_bindings = {"tau": "kept", "mu": None, "lvdt": None}
        view.slip_bindings = ["also_kept"]
        view.preview_results = {"tau": {"valid": True}}
        view.preview_parameters = {"tau": {"half_win": 1.5}}
        return view

    def test_event_time_coercion_accepts_real_scalars(self):
        for value, expected in (
            (1.25, 1.25),
            (2, 2.0),
            (np.float32(3.5), 3.5),
            (np.int64(4), 4.0),
        ):
            with self.subTest(value=value):
                result = EventDropEditorView._coerce_finite_event_time(value)
                self.assertEqual(result, expected)
                self.assertIsInstance(result, float)

    def test_event_time_coercion_rejects_invalid_values(self):
        invalid = (
            True,
            np.bool_(False),
            np.array(1.0),
            np.array([1.0]),
            np.nan,
            np.inf,
            -np.inf,
            "1.0",
            None,
        )
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(
                    EventDropEditorView._coerce_finite_event_time(value)
                )

    def test_current_event_time_uses_only_top_level_event_time(self):
        view = self.make_view(
            event={"event_time": np.float64(7.0), "time": np.array([99.0])}
        )
        self.assertEqual(view._get_current_event_time(), 7.0)

        for event in (
            {"time": np.array([99.0])},
            {"event_time": np.array([7.0]), "time": np.array([99.0])},
            "not an event",
        ):
            with self.subTest(event=event):
                view.event = event
                self.assertIsNone(view._get_current_event_time())

    def test_previous_event_time_searches_list_positions_without_sorting(self):
        events = [
            {"event_time": 100.0},
            {"event_time": 300.0, "skip": True},
            {"event_time": 200.0},
        ]
        view = self.make_view(events=events, event_idx=2)

        self.assertEqual(view._find_previous_event_time(), 300.0)
        self.assertIsInstance(view._find_previous_event_time(), float)

    def test_previous_event_time_skips_only_invalid_event_times(self):
        events = [
            {"event_time": np.int64(5)},
            {"event_time": np.nan},
            {"missing": 1},
            "not an event",
            {"event_time": np.array([8.0])},
            {"event_time": np.bool_(True)},
            {"event_time": "9.0"},
            {"event_time": 10.0},
        ]
        view = self.make_view(events=events, event_idx=len(events))
        self.assertIsNone(view._find_previous_event_time())

        view.event_idx = 7
        self.assertEqual(view._find_previous_event_time(), 5.0)

    def test_previous_event_time_handles_no_usable_position(self):
        cases = (
            ([], 0),
            ([{"event_time": 1.0}], 0),
            ([{"event_time": 1.0}], 2),
            ([{"event_time": 1.0}], True),
            ([{"event_time": 1.0}], 0.5),
            ("events", 1),
            ({"0": {"event_time": 1.0}}, 1),
            (None, 1),
        )
        for events, event_idx in cases:
            with self.subTest(events=events, event_idx=event_idx):
                view = self.make_view(events=events, event_idx=event_idx)
                self.assertIsNone(view._find_previous_event_time())

    def test_previous_event_search_does_not_modify_events(self):
        events = [
            {"event_time": 1.0, "payload": [1, 2]},
            {"event_time": np.nan},
            {"event_time": 3.0},
        ]
        original = copy.deepcopy(events)
        view = self.make_view(events=events, event_idx=2)

        self.assertEqual(view._find_previous_event_time(), 1.0)
        self.assertEqual(events, original)

    def test_full_run_time_returns_same_object_without_event_fallback(self):
        full_time = np.arange(5.0)
        view = self.make_view(
            event={"time": np.array([99.0])},
            run_data={"time": full_time},
        )
        self.assertIs(view._get_full_run_time(), full_time)

        view.run_data = {}
        self.assertIsNone(view._get_full_run_time())
        view.run_data = None
        self.assertIsNone(view._get_full_run_time())

    def test_full_run_signal_resolution_is_exact_and_identity_preserving(self):
        signal = np.arange(4.0)
        fallback = np.arange(4.0) + 10
        nested = {"inside": signal}
        view = self.make_view(
            event={"chosen": np.array([99.0])},
            run_data={
                "chosen": signal,
                "LP_displacement": fallback,
                "nested": nested,
            },
        )

        self.assertIs(view._resolve_full_run_signal("chosen"), signal)
        for name in (None, "", 3, "missing", "inside"):
            with self.subTest(name=name):
                self.assertIsNone(view._resolve_full_run_signal(name))

    def test_candidate_discovery_filters_shape_dtype_length_and_finiteness(self):
        full_time = np.array([0.0, 1.0, 2.0])
        good_int = np.array([1, 2, 3])
        good_float = np.array([1.5, 2.5, 3.5])
        run_data = {
            "metadata": 4.0,
            "good_int": good_int,
            "time": full_time,
            "good_float": good_float,
            "zero_dim": np.array(1.0),
            "two_dim": np.ones((3, 1)),
            "length_mismatch": np.ones(2),
            "empty": np.array([]),
            "bool": np.array([True, False, True]),
            "string": np.array(["1", "2", "3"]),
            "nan": np.array([1.0, np.nan, 3.0]),
            "inf": np.array([1.0, np.inf, 3.0]),
            "nested": {"signal": np.ones(3)},
        }
        view = self.make_view(run_data=run_data)
        originals = {
            key: value.copy()
            for key, value in run_data.items()
            if isinstance(value, np.ndarray)
        }

        self.assertEqual(
            view._find_full_run_signal_candidates(),
            ["good_int", "good_float"],
        )
        self.assertIs(run_data["time"], full_time)
        self.assertIs(run_data["good_int"], good_int)
        self.assertIs(run_data["good_float"], good_float)
        for key, original in originals.items():
            np.testing.assert_array_equal(run_data[key], original)

    def test_invalid_full_run_time_produces_no_candidates(self):
        invalid_times = (
            None,
            1.0,
            np.array([]),
            np.ones((2, 2)),
            np.array([True, False]),
            np.array(["0", "1"]),
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
        )
        for invalid_time in invalid_times:
            with self.subTest(time=invalid_time):
                run_data = {"signal": np.ones(2)}
                if invalid_time is not None:
                    run_data["time"] = invalid_time
                view = self.make_view(run_data=run_data)
                self.assertEqual(view._find_full_run_signal_candidates(), [])

    def test_candidate_discovery_allows_non_increasing_finite_time(self):
        for full_time in (
            np.array([0.0, 0.0, 1.0]),
            np.array([2.0, 1.0, 0.0]),
        ):
            with self.subTest(time=full_time):
                view = self.make_view(
                    run_data={"time": full_time, "signal": np.ones(3)}
                )
                self.assertEqual(
                    view._find_full_run_signal_candidates(), ["signal"]
                )

    def test_refresh_updates_context_without_touching_event_local_state(self):
        events = [
            {"event_time": 2.0},
            {"event_time": np.float64(4.0)},
        ]
        full_time = np.arange(5.0)
        run_data = {
            "time": full_time,
            "events": events,
            "candidate": np.arange(5.0),
        }

        class FakeDataManager:
            def __init__(self):
                self.calls = []

            def get_data(self, path):
                self.calls.append(path)
                return run_data

            def set_data(self, *args, **kwargs):
                raise AssertionError("write API must not be called")

            def save_file(self, *args, **kwargs):
                raise AssertionError("save API must not be called")

        view = self.make_view(event=events[1], events=None, event_idx=1)
        view.data_manager = FakeDataManager()
        bindings = view.metric_bindings.copy()
        slips = view.slip_bindings.copy()
        results = view.preview_results.copy()
        parameters = view.preview_parameters.copy()

        view._refresh_interevent_context()

        self.assertIs(view.run_data, run_data)
        self.assertIs(view.events, events)
        self.assertEqual(view.current_event_time, 4.0)
        self.assertEqual(view.previous_event_time, 2.0)
        self.assertEqual(view.full_run_signal_candidates, ["candidate"])
        self.assertEqual(view.metric_bindings, bindings)
        self.assertEqual(view.slip_bindings, slips)
        self.assertEqual(view.preview_results, results)
        self.assertEqual(view.preview_parameters, parameters)
        self.assertEqual(view.data_manager.calls, ["runs/[3]"])

    def test_refresh_failure_sets_run_context_unavailable(self):
        class FailingDataManager:
            def get_data(self, path):
                raise KeyError(path)

        view = self.make_view(event={"event_time": 8.0}, event_idx=0)
        view.data_manager = FailingDataManager()
        view.run_data = {"stale": True}
        view.full_run_signal_candidates = ["stale"]

        view._refresh_interevent_context()

        self.assertIsNone(view.run_data)
        self.assertIsNone(view.events)
        self.assertEqual(view.current_event_time, 8.0)
        self.assertIsNone(view.previous_event_time)
        self.assertEqual(view.full_run_signal_candidates, [])

    def test_set_event_continues_when_run_context_is_not_a_mapping(self):
        event = {"event_time": 2.0, "time": np.array([1.0, 2.0])}

        class FakeDataManager:
            def __init__(self):
                self.calls = []

            def get_data(self, path):
                self.calls.append(path)
                if path.endswith("/events/[0]"):
                    return event
                return ["not", "a", "run mapping"]

        view = self.make_view()
        view.data_manager = FakeDataManager()
        view._set_event(0)

        self.assertIs(view.event, event)
        self.assertIsNone(view.run_data)
        self.assertEqual(view.current_event_time, 2.0)
        self.assertIsNone(view.previous_event_time)
        self.assertEqual(view.full_run_signal_candidates, [])
        self.assertEqual(
            view.data_manager.calls,
            ["runs/[3]/events/[0]", "runs/[3]"],
        )

    def test_each_event_switch_reloads_all_derived_context(self):
        first = {"event_time": 1.0, "time": np.array([0.0, 1.0])}
        second = {"event_time": 3.0, "time": np.array([2.0, 3.0])}
        events = [first, second]
        runs = [
            {
                "time": np.arange(4.0),
                "events": events,
                "first_full": np.arange(4.0),
            },
            {
                "time": np.arange(5.0),
                "events": events,
                "second_full": np.arange(5.0),
            },
        ]

        class FakeDataManager:
            def __init__(self):
                self.run_calls = 0

            def get_data(self, path):
                if "/events/" in path:
                    index = int(path.rsplit("[", 1)[1][:-1])
                    return events[index]
                run = runs[self.run_calls]
                self.run_calls += 1
                return run

        view = self.make_view()
        view.data_manager = FakeDataManager()

        view._set_event(0)
        self.assertIs(view.run_data, runs[0])
        self.assertEqual(view.current_event_time, 1.0)
        self.assertIsNone(view.previous_event_time)
        self.assertEqual(
            view.full_run_signal_candidates, ["first_full"]
        )

        view._set_event(1)
        self.assertIs(view.run_data, runs[1])
        self.assertEqual(view.current_event_time, 3.0)
        self.assertEqual(view.previous_event_time, 1.0)
        self.assertEqual(
            view.full_run_signal_candidates, ["second_full"]
        )
        self.assertEqual(
            view.metric_bindings, {"tau": None, "mu": None, "lvdt": None}
        )
        self.assertEqual(view.slip_bindings, [])
        self.assertEqual(view.preview_results, {})

    @mock.patch(
        "labquake_explorer.analysis.event_drop."
        "calculate_interevent_displacement_metrics"
    )
    def test_context_resolution_does_not_call_analysis(self, calculator):
        run_data = {
            "time": np.arange(3.0),
            "events": [{"event_time": 1.0}],
            "signal": np.arange(3.0),
        }

        class FakeDataManager:
            def get_data(self, path):
                return run_data

        view = self.make_view(event=run_data["events"][0])
        view.data_manager = FakeDataManager()
        view._refresh_interevent_context()

        calculator.assert_not_called()


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


class IntereventParameterAndFormattingTests(unittest.TestCase):
    def test_interevent_parameters_accept_finite_signed_values(self):
        for speed, delay in (("3.508", "0.05"), ("-2", "-0.5"), ("0", "0")):
            with self.subTest(speed=speed, delay=delay):
                result = parse_interevent_parameters(speed, delay, "100")
                self.assertIsInstance(result["push_speed"], float)
                self.assertIsInstance(result["delay_sec"], float)
                self.assertIsInstance(result["lvdt_smooth_w"], int)
                self.assertEqual(result["lvdt_smooth_w"], 100)

    def test_interevent_parameters_reject_invalid_numeric_text(self):
        for name, values in (
            ("push_speed", ("", "text", "nan", "inf", "-inf")),
            ("delay_sec", ("", "text", "nan", "inf", "-inf")),
        ):
            for value in values:
                arguments = {
                    "push_speed": "3.508",
                    "delay_sec": "0.05",
                    "lvdt_smooth_w": "100",
                }
                arguments[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        parse_interevent_parameters(**arguments)

    def test_interevent_smoothing_requires_positive_integer_text(self):
        for value in ("", "0", "-1", "1.0", "text", "True", "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "lvdt_smooth_w"):
                    parse_interevent_parameters("1", "0", value)

    def test_interevent_result_format_preserves_signed_value(self):
        self.assertEqual(
            format_interevent_result({"valid": True, "value": -2.5}),
            {"valid": "True", "value": "-2.5"},
        )
        self.assertEqual(
            format_interevent_result({"valid": False}),
            {"valid": "False", "value": "—"},
        )


class IntereventPreviewTests(unittest.TestCase):
    @staticmethod
    def make_view():
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.current_event_time = 4.0
        view.previous_event_time = 2.0
        view.run_data = None
        view.full_run_signal_candidates = []
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        view.interevent_bindings = {"reference": None}
        view.interevent_preview_results = {}
        view.interevent_preview_parameters = None
        view.interevent_parameter_vars = {
            "push_speed": FakeVariable("3.508"),
            "delay_sec": FakeVariable("0.05"),
            "lvdt_smooth_w": FakeVariable("100"),
        }
        view.interevent_result_vars = {
            metric: {
                "valid": FakeVariable("—"),
                "value": FakeVariable("—"),
            }
            for metric in ("D_Push", "D_max", "D_reference")
        }
        view.interevent_status_var = FakeVariable("Preview only — not saved")
        view.reference_signal_combobox = FakeWidget("")
        view.interevent_preview_button = FakeWidget()
        view.preview_results = {"tau": {"valid": True}}
        view.preview_parameters = {"tau": {"half_win": 1.5}}
        view.result_vars = {"lvdt": {"valid": FakeVariable("True")}}
        view.slip_rows = []
        return view

    def test_constructor_initializes_separate_interevent_state(self):
        parent = mock.Mock()
        parent.root = mock.Mock()
        parent.data_manager = mock.Mock()
        with (
            mock.patch.object(
                EventDropEditorView.__bases__[0], "__init__", return_value=None
            ),
            mock.patch.object(EventDropEditorView, "title"),
            mock.patch.object(EventDropEditorView, "protocol"),
            mock.patch.object(EventDropEditorView, "_set_event"),
            mock.patch.object(EventDropEditorView, "_create_controls"),
            mock.patch.object(EventDropEditorView, "_create_figure"),
            mock.patch.object(EventDropEditorView, "_initialize_event_selector"),
            mock.patch.object(EventDropEditorView, "_refresh_event_widgets"),
        ):
            view = EventDropEditorView(parent, 2, 3)

        self.assertEqual(view.interevent_bindings, {"reference": None})
        self.assertEqual(view.interevent_preview_results, {})
        self.assertIsNone(view.interevent_preview_parameters)
        self.assertNotIn("reference", view.metric_bindings)
        self.assertEqual(view.slip_bindings, [])

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_d_push_only_passes_no_run_context_and_calls_once(self, calculator):
        calculator.return_value = {
            "D_Push": {"valid": True, "value": 7.016},
            "D_max": {"valid": False},
            "D_reference": {"valid": False},
        }
        view = self.make_view()

        result = view.calculate_interevent_preview()

        self.assertIs(result, calculator.return_value)
        calculator.assert_called_once_with(
            current_event_time=4.0,
            previous_event_time=2.0,
            push_speed=3.508,
            time=None,
            lvdt_signal=None,
            reference_displacement_signal=None,
            delay_sec=0.05,
            lvdt_smooth_w=100,
        )
        self.assertIs(view.interevent_preview_results, result)
        self.assertEqual(
            view.interevent_preview_parameters,
            {
                "push_speed": 3.508,
                "delay_sec": 0.05,
                "lvdt_smooth_w": 100,
            },
        )

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_lvdt_only_uses_full_run_objects_not_event_array(self, calculator):
        full_time = np.arange(5.0)
        full_lvdt = np.arange(5.0) * 2
        event_lvdt = np.array([99.0, 100.0])
        view = self.make_view()
        view.run_data = {"time": full_time, "chosen": full_lvdt}
        view.metric_bindings["lvdt"] = "chosen"
        view.signal_candidates = {"chosen": event_lvdt}
        original_run = {key: value.copy() for key, value in view.run_data.items()}
        original_event_signal = event_lvdt.copy()
        calculator.return_value = {
            "D_Push": {"valid": True, "value": 1.0},
            "D_max": {"valid": True, "value": 2.0},
            "D_reference": {"valid": False},
        }

        view.calculate_interevent_preview()

        arguments = calculator.call_args.kwargs
        self.assertIs(arguments["time"], full_time)
        self.assertIs(arguments["lvdt_signal"], full_lvdt)
        self.assertIsNot(arguments["lvdt_signal"], event_lvdt)
        self.assertIsNone(arguments["reference_displacement_signal"])
        calculator.assert_called_once()
        for key, original in original_run.items():
            np.testing.assert_array_equal(view.run_data[key], original)
        np.testing.assert_array_equal(event_lvdt, original_event_signal)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_reference_only_and_dual_signal_preserve_identity(self, calculator):
        full_time = np.arange(5.0)
        lvdt = np.arange(5.0)
        reference = np.arange(5.0) * 3
        view = self.make_view()
        view.run_data = {
            "time": full_time,
            "lvdt_key": lvdt,
            "reference_key": reference,
        }
        view.interevent_bindings["reference"] = "reference_key"
        calculator.return_value = {
            "D_Push": {"valid": True, "value": 1.0},
            "D_max": {"valid": False},
            "D_reference": {"valid": True, "value": 3.0},
        }

        view.calculate_interevent_preview()
        first_arguments = calculator.call_args.kwargs
        self.assertIs(first_arguments["time"], full_time)
        self.assertIsNone(first_arguments["lvdt_signal"])
        self.assertIs(
            first_arguments["reference_displacement_signal"], reference
        )

        calculator.reset_mock()
        view.metric_bindings["lvdt"] = "lvdt_key"
        view.calculate_interevent_preview()
        second_arguments = calculator.call_args.kwargs
        self.assertIs(second_arguments["time"], full_time)
        self.assertIs(second_arguments["lvdt_signal"], lvdt)
        self.assertIs(
            second_arguments["reference_displacement_signal"], reference
        )
        calculator.assert_called_once()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_missing_full_run_bindings_do_not_fallback(self, calculator):
        fallback = np.arange(4.0)
        reference = np.arange(4.0) + 10
        event_lvdt = np.array([1.0, 2.0])
        view = self.make_view()
        view.run_data = {
            "time": np.arange(4.0),
            "LP_displacement": fallback,
            "reference": reference,
        }
        view.metric_bindings["lvdt"] = "event_only"
        view.interevent_bindings["reference"] = "reference"
        view.signal_candidates = {"event_only": event_lvdt}
        calculator.return_value = {
            "D_Push": {"valid": True, "value": 1.0},
            "D_max": {"valid": False},
            "D_reference": {"valid": True, "value": 2.0},
        }

        view.calculate_interevent_preview()

        arguments = calculator.call_args.kwargs
        self.assertIsNone(arguments["lvdt_signal"])
        self.assertIs(
            arguments["reference_displacement_signal"], reference
        )
        self.assertIsNot(arguments["lvdt_signal"], fallback)
        self.assertIsNot(arguments["lvdt_signal"], event_lvdt)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_context_errors_prevent_analysis_without_touching_event_preview(
        self, calculator
    ):
        view = self.make_view()
        event_results = view.preview_results
        view.current_event_time = None
        with self.assertRaisesRegex(ValueError, "Current event"):
            view.calculate_interevent_preview()
        calculator.assert_not_called()
        self.assertIs(view.preview_results, event_results)

        view.current_event_time = 4.0
        view.run_data = {"selected": np.arange(3.0)}
        view.interevent_bindings["reference"] = "selected"
        with self.assertRaisesRegex(ValueError, "Full-run time"):
            view.calculate_interevent_preview()
        calculator.assert_not_called()

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_first_event_calls_analysis_and_displays_unavailable(
        self, calculator
    ):
        calculator.return_value = {
            "D_Push": {"valid": False},
            "D_max": {"valid": False},
            "D_reference": {"valid": False},
        }
        view = self.make_view()
        view.previous_event_time = None

        with mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view."
            "messagebox.showerror"
        ) as showerror:
            view.recompute_interevent_preview()

        calculator.assert_called_once()
        showerror.assert_not_called()
        for metric in ("D_Push", "D_max", "D_reference"):
            self.assertEqual(
                view.interevent_result_vars[metric]["valid"].get(), "False"
            )
            self.assertEqual(
                view.interevent_result_vars[metric]["value"].get(), "—"
            )
        self.assertEqual(
            view.interevent_status_var.get(),
            "No previous event with a finite event_time",
        )

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_interevent_displacement_metrics"
    )
    def test_recompute_formats_mixed_results_and_handles_value_error(
        self, calculator
    ):
        view = self.make_view()
        calculator.return_value = {
            "D_Push": {"valid": True, "value": -2.0},
            "D_max": {"valid": False},
            "D_reference": {"valid": True, "value": 4.5},
        }
        view.recompute_interevent_preview()
        self.assertEqual(
            view.interevent_result_vars["D_Push"]["value"].get(), "-2"
        )
        self.assertEqual(
            view.interevent_result_vars["D_max"]["value"].get(), "—"
        )
        self.assertEqual(
            view.interevent_result_vars["D_reference"]["value"].get(), "4.5"
        )

        event_results = view.preview_results
        calculator.side_effect = ValueError("bad run array")
        with mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view."
            "messagebox.showerror"
        ) as showerror:
            view.recompute_interevent_preview()
        self.assertEqual(view.interevent_preview_results, {})
        self.assertIsNone(view.interevent_preview_parameters)
        self.assertIs(view.preview_results, event_results)
        showerror.assert_called_once_with(
            "Inter-event displacement", "bad run array"
        )

    def test_clear_and_metric_invalidation_are_isolated(self):
        view = self.make_view()
        view.interevent_preview_results = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }
        for metric in view.interevent_result_vars:
            view.interevent_result_vars[metric]["valid"].set("True")
            view.interevent_result_vars[metric]["value"].set("1")
        event_results = view.preview_results

        view._invalidate_interevent_preview(("D_max",), "changed")

        self.assertIn("D_Push", view.interevent_preview_results)
        self.assertNotIn("D_max", view.interevent_preview_results)
        self.assertIn("D_reference", view.interevent_preview_results)
        self.assertEqual(
            view.interevent_result_vars["D_max"]["value"].get(), "—"
        )
        self.assertEqual(
            view.interevent_result_vars["D_Push"]["value"].get(), "1"
        )
        self.assertIs(view.preview_results, event_results)

        view._clear_interevent_result_display()
        for metric in view.interevent_result_vars:
            self.assertEqual(
                view.interevent_result_vars[metric]["valid"].get(), "—"
            )

    def test_parameter_invalidation_matrix_does_not_analyze(self):
        expected = {
            "push_speed": {"D_max", "D_reference"},
            "delay_sec": {"D_Push"},
            "lvdt_smooth_w": {"D_Push", "D_reference"},
        }
        for parameter, remaining in expected.items():
            with self.subTest(parameter=parameter):
                view = self.make_view()
                view.interevent_preview_results = {
                    metric: {"valid": True}
                    for metric in ("D_Push", "D_max", "D_reference")
                }
                with mock.patch(
                    "labquake_explorer.ui.views.event_drop_editor_view."
                    "calculate_interevent_displacement_metrics"
                ) as calculator:
                    view._on_interevent_parameter_changed(parameter)
                self.assertEqual(
                    set(view.interevent_preview_results), remaining
                )
                calculator.assert_not_called()
                self.assertEqual(
                    view.preview_results, {"tau": {"valid": True}}
                )

    def test_event_local_endpoint_invalidation_does_not_clear_d_results(self):
        view = self.make_view()
        view.active_metric_role = "tau"
        view.preview_results = {
            "tau": {"valid": True},
            "mu": {"valid": True},
        }
        view.preview_parameters = {"tau": {}}
        view.status_var = FakeVariable()
        view._clear_result_display = mock.Mock()
        view.interevent_preview_results = {
            metric: {"valid": True}
            for metric in ("D_Push", "D_max", "D_reference")
        }
        original_d_results = view.interevent_preview_results.copy()

        view._invalidate_preview()

        self.assertEqual(
            view.interevent_preview_results, original_d_results
        )

    def test_reference_binding_changes_only_reference_result(self):
        view = self.make_view()
        view.full_run_signal_candidates = ["same"]
        view.metric_bindings["lvdt"] = "same"
        view.reference_signal_combobox.set("same")
        view.interevent_preview_results = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }

        view.on_reference_signal_changed()

        self.assertEqual(view.interevent_bindings, {"reference": "same"})
        self.assertEqual(
            set(view.interevent_preview_results), {"D_Push", "D_max"}
        )
        self.assertEqual(view.preview_results, {"tau": {"valid": True}})

        view.reference_signal_combobox.set("stale")
        view.on_reference_signal_changed()
        self.assertIsNone(view.interevent_bindings["reference"])

    def test_lvdt_binding_invalidates_d_max_but_tau_does_not(self):
        view = self.make_view()
        view.signal_candidates = {"signal": np.arange(2.0)}
        view.lvdt_signal_combobox = FakeWidget("signal")
        view.tau_signal_combobox = FakeWidget("signal")
        view.preview_button = FakeWidget()
        view.status_var = FakeVariable()
        view.raw_ax = mock.Mock()
        view.fit_ax = mock.Mock()
        view.canvas = mock.Mock()
        view._plot_preview = mock.Mock()
        view._clear_result_display = mock.Mock()
        view.interevent_preview_results = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }

        view.on_lvdt_signal_changed()
        self.assertNotIn("D_max", view.interevent_preview_results)
        self.assertIn("D_Push", view.interevent_preview_results)
        self.assertIn("D_reference", view.interevent_preview_results)

        view.interevent_preview_results["D_max"] = {"valid": True}
        view.on_tau_signal_changed()
        self.assertIn("D_max", view.interevent_preview_results)

    def test_refresh_widgets_preserves_valid_reference_and_clears_stale(self):
        view = self.make_view()
        view.current_event_time = 3.0
        view.full_run_signal_candidates = ["kept", "other"]
        view.interevent_bindings["reference"] = "kept"
        view._refresh_interevent_widgets()
        self.assertEqual(view.reference_signal_combobox.get(), "kept")
        self.assertEqual(
            view.reference_signal_combobox.options["values"],
            ["kept", "other"],
        )
        self.assertEqual(
            view.interevent_preview_button.options["state"], "normal"
        )

        view.full_run_signal_candidates = ["new"]
        view._refresh_interevent_widgets()
        self.assertIsNone(view.interevent_bindings["reference"])
        self.assertEqual(view.reference_signal_combobox.get(), "")

        view.current_event_time = None
        view._refresh_interevent_widgets()
        self.assertEqual(
            view.interevent_preview_button.options["state"], "disabled"
        )
        self.assertEqual(
            view.interevent_status_var.get(),
            "Current event has no finite event_time",
        )


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

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_drop_metrics"
    )
    def test_slip_endpoint_invalidates_all_slip_results_only(self, calculator):
        view = self.make_plot_view()
        view.slip_bindings = ["signal", "signal"]
        view.active_metric_role = "slip_2"
        view.slip_parameter_vars = {
            key: FakeVariable(variable.get())
            for key, variable in view.parameter_vars.items()
        }
        view.preview_results = {
            "tau": {"valid": True},
            "mu": {"valid": True},
            "lvdt": {"valid": True},
            "slip_1": {"valid": True},
            "slip_2": {"valid": False},
        }
        view.slip_rows = [
            {"result_vars": {key: FakeVariable("old") for key in view.result_vars["tau"]}},
            {"result_vars": {key: FakeVariable("old") for key in view.result_vars["tau"]}},
        ]
        view.data_manager = mock.Mock()

        view._on_endpoint_changed("pre_start", -0.9)

        self.assertEqual(
            view.preview_results,
            {
                "tau": {"valid": True},
                "mu": {"valid": True},
                "lvdt": {"valid": True},
            },
        )
        self.assertEqual(view.slip_parameter_vars["pre_start"].get(), "-0.9")
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
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)
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
        self.assertEqual(view.preview_results, {})
        self.assertIsNone(view.preview_parameters)

    def test_tau_selection_only_updates_tau_and_does_not_analyze(self):
        view = self.make_view()
        view.tau_signal_combobox.set("second")

        view.on_tau_signal_changed()

        self.assertEqual(view.preview_button.options["state"], "normal")


class SlipBindingStateTests(unittest.TestCase):
    def make_view(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.signal_candidates = {
            "z_signal": np.array([1.0, 2.0]),
            "a_signal": np.array([3.0, 4.0]),
            "first": np.array([5.0, 6.0]),
            "second": np.array([7.0, 8.0]),
        }
        view.metric_bindings = {"tau": None, "mu": None, "lvdt": None}
        view.slip_bindings = []
        view.slip_rows = []
        view.preview_results = {}
        view.preview_parameters = None
        view.active_metric_role = None
        view.tau_signal_combobox = FakeWidget()
        view.mu_signal_combobox = FakeWidget()
        view.lvdt_signal_combobox = FakeWidget()
        view.preview_button = FakeWidget()
        view.status_var = FakeVariable()
        view.result_vars = {
            role: {"valid": FakeVariable(), "delta": FakeVariable()}
            for role in ("tau", "mu", "lvdt")
        }
        view.preview_results = {
            "tau": {"valid": True},
            "mu": {"valid": True},
        }
        view.preview_parameters = {"half_win": 1.0}
        view._plot_preview = mock.Mock()
        view._rebuild_slip_rows = mock.Mock()
        return view

    def test_initial_state_and_empty_added_row_do_not_enable_preview(self):
        view = self.make_view()

        view.add_slip_sensor()

        self.assertEqual(view.slip_bindings, [None])
        self.assertEqual(view.preview_button.options["state"], "disabled")
        view._rebuild_slip_rows.assert_called_once_with()

    def test_selection_updates_only_slot_and_allows_duplicate_signal(self):
        view = self.make_view()
        view.slip_bindings = [None, None]
        view.slip_rows = [
            {"selector": FakeWidget("z_signal"), "result_vars": {}},
            {"selector": FakeWidget("z_signal"), "result_vars": {}},
        ]
        view.preview_results = {
            "tau": {"valid": True},
            "slip_1": {"valid": True},
            "slip_2": {"valid": True},
        }
        view._clear_result_display = mock.Mock()

        view.on_slip_signal_changed(1)

        self.assertEqual(view.slip_bindings, [None, "z_signal"])
        self.assertEqual(view.active_metric_role, "slip_2")
        self.assertEqual(
            view.preview_results,
            {"tau": {"valid": True}, "slip_1": {"valid": True}},
        )
        self.assertEqual(view.preview_button.options["state"], "normal")

        view.on_slip_signal_changed(0)
        self.assertEqual(view.slip_bindings, ["z_signal", "z_signal"])

    def test_remove_compacts_rows_clears_slip_results_and_falls_back(self):
        view = self.make_view()
        view.metric_bindings["tau"] = "z_signal"
        view.slip_bindings = ["z_signal", "a_signal", "z_signal"]
        view.preview_results = {
            "tau": {"valid": True},
            "slip_1": {"valid": True},
            "slip_2": {"valid": True},
            "slip_3": {"valid": True},
        }
        view.active_metric_role = "slip_2"

        view.remove_slip_sensor(1)

        self.assertEqual(view.slip_bindings, ["z_signal", "z_signal"])
        self.assertEqual(view.preview_results, {"tau": {"valid": True}})
        self.assertEqual(view.active_metric_role, "tau")
        view._rebuild_slip_rows.assert_called_once_with()
        view._plot_preview.assert_called_once_with()

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
