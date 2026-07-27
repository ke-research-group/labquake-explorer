"""Headless-safe tests for the single-signal event-drop preview view."""

import copy
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import EventDropEditorView as ExportedEventDropEditorView
from labquake_explorer.ui.views.event_drop_editor_view import (
    EventDropEditorView,
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
    def make_view(self, event):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.event = event
        view.signal_candidates = find_signal_candidates(event)
        view.preview_result = None
        view.preview_parameters = None
        return view

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_signal_drop"
    )
    def test_helper_receives_explicit_preview_arguments(self, calculator):
        event = {
            "time": np.array([9.0, 9.5, 10.5, 11.0]),
            "event_time": 10.0,
            "selected": np.array([5.0, 6.0, 2.0, 3.0]),
        }
        expected_result = {
            "valid": True,
            "delta": 4.0,
            "val_pre_0": 5.0,
            "val_post_0": 1.0,
        }
        calculator.return_value = expected_result
        view = self.make_view(event)

        result = view.calculate_preview(
            "selected",
            half_win=1.5,
            points=(-1.0, -0.5, 0.5, 1.0),
            smooth_w=9,
        )

        calculator.assert_called_once()
        call = calculator.call_args.kwargs
        self.assertIs(call["time"], event["time"])
        self.assertIs(call["signal"], view.signal_candidates["selected"])
        self.assertEqual(call["event_time"], 10.0)
        self.assertEqual(call["half_win"], 1.5)
        self.assertEqual(call["points"], (-1.0, -0.5, 0.5, 1.0))
        self.assertEqual(call["smooth_w"], 9)
        self.assertIs(result, expected_result)
        self.assertIs(view.preview_result, expected_result)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view."
        "calculate_event_signal_drop"
    )
    def test_preview_does_not_modify_event(self, calculator):
        event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
            "signal": np.array([4.0, 5.0, 1.0, 2.0]),
        }
        original = copy.deepcopy(event)
        calculator.return_value = {"valid": False}
        view = self.make_view(event)

        view.calculate_preview(
            "signal",
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
            smooth_w=None,
        )

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

        view.calculate_preview(
            "signal",
            half_win=1.0,
            points=(-1.0, -0.5, 0.5, 1.0),
            smooth_w=None,
        )

        view.data_manager.set_data.assert_not_called()


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
        view.signal_combobox = FakeWidget("signal")
        view.parameter_vars = {
            "half_win": FakeVariable("2"),
            "pre_start": FakeVariable("-0.5"),
            "pre_end": FakeVariable("-1.5"),
            "post_start": FakeVariable("1.5"),
            "post_end": FakeVariable("0.5"),
            "smooth_w": FakeVariable(""),
        }
        view.preview_result = {
            "valid": True,
            "coeff_pre": np.array([1.0, 4.0]),
            "coeff_post": np.array([2.0, 1.0]),
            "delta": 3.0,
            "val_pre_0": 4.0,
            "val_post_0": 1.0,
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
        view.preview_result = {"valid": True}
        view.preview_parameters = {"half_win": 1.0}

        view._set_event(1)

        self.assertIs(view.event, second_event)
        self.assertEqual(list(view.signal_candidates), ["second"])
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.preview_parameters)
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


class SignalSelectionStateTests(unittest.TestCase):
    def make_view(self):
        view = EventDropEditorView.__new__(EventDropEditorView)
        view.signal_candidates = {
            "first": np.array([1.0, 2.0]),
            "second": np.array([3.0, 4.0]),
        }
        view.signal_combobox = FakeWidget()
        view.preview_button = FakeWidget()
        view.status_var = FakeVariable()
        view.result_vars = {
            "valid": FakeVariable(),
            "delta": FakeVariable(),
        }
        view.preview_result = {"valid": True}
        view.preview_parameters = {"half_win": 1.0}
        view._plot_preview = mock.Mock()
        return view

    def test_candidates_do_not_cause_automatic_signal_selection(self):
        view = self.make_view()

        view._refresh_event_widgets()

        self.assertEqual(view.signal_combobox.options["values"], ["first", "second"])
        self.assertEqual(view.signal_combobox.get(), "")
        self.assertEqual(view.preview_button.options["state"], "disabled")

    def test_explicit_valid_selection_enables_preview_and_clears_old_result(self):
        view = self.make_view()
        view.signal_combobox.set("second")

        view.on_signal_changed()

        self.assertEqual(view.preview_button.options["state"], "normal")
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.preview_parameters)


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
