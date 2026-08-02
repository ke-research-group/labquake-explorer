"""Headless-safe tests for the loading-stiffness preview view."""

import copy
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import EventKEditorView as ExportedEventKEditorView
from labquake_explorer.ui.views.event_k_editor_view import (
    EventKEditorView,
    _DraggablePreEndpoint,
    find_full_run_signal_candidates,
    parse_k_preview_parameters,
)


class FakeVariable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeWidget(FakeVariable):
    def __init__(self, value=""):
        super().__init__(value)
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


class FakeCanvas:
    def draw_idle(self):
        pass


def parameter_vars(**overrides):
    values = {
        "pre_start": "-3.0",
        "pre_end": "-0.5",
        "window_sec": "3.5",
        "smooth_w": "100",
        "highpass_freq": "0.0",
        "lowpass_freq": "0.0",
    }
    values.update(overrides)
    return {key: FakeVariable(value) for key, value in values.items()}


def make_headless_view(run_data=None, event=None):
    time = np.linspace(0.0, 10.0, 101)
    run_data = run_data or {
        "time": time,
        "tau source": time * 2.0,
        "slip source": time * 0.5,
        "events": [event or {"event_time": 5.0}],
    }
    event = event or run_data["events"][0]
    view = EventKEditorView.__new__(EventKEditorView)
    view.run_idx = 2
    view.event_idx = 0
    view.run_data = run_data
    view.events = run_data.get("events")
    view.event = event
    view.current_event_time = EventKEditorView._coerce_event_time(
        event.get("event_time")
    )
    view.full_run_signal_candidates = find_full_run_signal_candidates(run_data)
    view.metric_bindings = {"tau": None, "slip": None}
    view.preview_result = None
    view.preview_parameters = None
    view.parameter_vars = parameter_vars()
    view.use_ransac_var = FakeVariable(False)
    view.signal_comboboxes = {"tau": FakeWidget(), "slip": FakeWidget()}
    view.preview_button = FakeWidget()
    view.result_vars = {
        "valid": FakeVariable("—"),
        "k": FakeVariable("—"),
        "intercept": FakeVariable("—"),
    }
    view.status_var = FakeVariable()
    view._endpoint_draggables = []
    view._initializing_parameters = False
    view.figure = Figure()
    view.slip_ax = view.figure.add_subplot(311)
    view.tau_ax = view.figure.add_subplot(312)
    view.fit_ax = view.figure.add_subplot(313)
    view.canvas = FakeCanvas()
    return view


class ParameterParsingTests(unittest.TestCase):
    def test_defaults_and_python_types(self):
        result = parse_k_preview_parameters(
            "-3.0", "-0.5", "3.5", "100", "0.0", "0.0", False
        )
        self.assertEqual(
            result,
            {
                "pre_start": -3.0,
                "pre_end": -0.5,
                "window_sec": 3.5,
                "highpass_freq": 0.0,
                "lowpass_freq": 0.0,
                "smooth_w": 100,
                "use_ransac": False,
            },
        )
        self.assertIsInstance(result["smooth_w"], int)
        self.assertIsInstance(result["use_ransac"], bool)

    def test_finite_numeric_fields_are_required(self):
        fields = ["nan", "inf", "text", ""]
        for bad in fields:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                parse_k_preview_parameters(bad, "-0.5", "3.5", "1", "0", "0", False)

    def test_pre_window_rules(self):
        cases = [
            (("-0.5", "-1"), "less than pre_end"),
            (("0", "0.5"), "less than zero"),
            (("-1", "0.1"), "at or before zero"),
        ]
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                parse_k_preview_parameters(*values, "3.5", "1", "0", "0", False)

    def test_window_smoothing_and_cutoff_rules(self):
        with self.assertRaisesRegex(ValueError, "window_sec"):
            parse_k_preview_parameters("-1", "0", "0", "1", "0", "0", False)
        for bad in ("0", "-1", "1.5", "abc"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "smooth_w"):
                parse_k_preview_parameters("-1", "0", "1", bad, "0", "0", False)
        for high, low in (("-1", "0"), ("0", "-1")):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                parse_k_preview_parameters("-1", "0", "1", "1", high, low, False)


class CandidateAndContextTests(unittest.TestCase):
    def test_candidates_are_exact_top_level_aligned_finite_real_arrays(self):
        time = np.arange(3.0)
        run = {
            "time": time,
            "z arbitrary": np.ones(3),
            "tau_local": np.ones((3, 1)),
            "bool": np.array([True, False, True]),
            "complex": np.ones(3, dtype=complex),
            "nan": np.array([1.0, np.nan, 2.0]),
            "short": np.ones(2),
            "a arbitrary": [1, 2, 3],
            "nested": {"signal": np.ones(3)},
        }
        self.assertEqual(
            find_full_run_signal_candidates(run),
            ["z arbitrary", "a arbitrary"],
        )

    def test_candidate_discovery_does_not_copy_or_modify_arrays(self):
        time = np.arange(3.0)
        signal = np.arange(3.0)
        run = {"time": time, "signal": signal}
        before = signal.copy()
        self.assertEqual(find_full_run_signal_candidates(run), ["signal"])
        np.testing.assert_array_equal(signal, before)

    def test_invalid_time_yields_no_candidates(self):
        for time in (np.ones((2, 2)), np.array([0.0, np.inf]), [True, False]):
            with self.subTest(time=time):
                self.assertEqual(
                    find_full_run_signal_candidates({"time": time, "x": [1, 2]}),
                    [],
                )

    def test_event_time_accepts_only_finite_real_scalar(self):
        self.assertEqual(EventKEditorView._coerce_event_time(np.float64(2.5)), 2.5)
        for value in (True, np.array(2.0), np.nan, np.inf, "2"):
            with self.subTest(value=value):
                self.assertIsNone(EventKEditorView._coerce_event_time(value))

    def test_set_event_uses_exact_paths_and_identity_without_mutation(self):
        run = {"time": np.arange(3.0), "signal": np.arange(3.0), "events": [{"event_time": 1.0}]}
        manager = mock.Mock()
        manager.get_data.side_effect = [run, run["events"][0]]
        view = EventKEditorView.__new__(EventKEditorView)
        view.data_manager = manager
        view.run_idx = 4
        view.metric_bindings = {"tau": "old", "slip": "old"}
        before = copy.deepcopy(run)
        view._set_event(0)
        self.assertIs(view.run_data, run)
        self.assertIs(view.event, run["events"][0])
        self.assertEqual(manager.get_data.call_args_list, [mock.call("runs/[4]"), mock.call("runs/[4]/events/[0]")])
        self.assertEqual(view.metric_bindings, {"tau": None, "slip": None})
        np.testing.assert_array_equal(run["time"], before["time"])
        self.assertEqual(run["events"], before["events"])


class BindingAndAnalysisTests(unittest.TestCase):
    def test_refresh_and_selection_never_auto_bind(self):
        view = make_headless_view()
        view._refresh_event_widgets()
        self.assertEqual(view.metric_bindings, {"tau": None, "slip": None})
        self.assertEqual(view.signal_comboboxes["tau"].get(), "")
        self.assertEqual(view.preview_button.options["state"], "disabled")

    def test_two_explicit_bindings_enable_preview_and_duplicates_are_allowed(self):
        view = make_headless_view()
        view.signal_comboboxes["tau"].set("tau source")
        view.on_signal_changed("tau")
        self.assertEqual(view.preview_button.options["state"], "disabled")
        view.signal_comboboxes["slip"].set("tau source")
        view.on_signal_changed("slip")
        self.assertEqual(view.metric_bindings, {"tau": "tau source", "slip": "tau source"})
        self.assertEqual(view.preview_button.options["state"], "normal")

    def test_stale_binding_is_cleared_and_invalidates_without_analysis(self):
        view = make_headless_view()
        view.preview_result = {"valid": True}
        view.preview_parameters = {"old": True}
        view.signal_comboboxes["tau"].set("missing")
        with mock.patch("labquake_explorer.ui.views.event_k_editor_view.calculate_event_loading_stiffness") as calculator:
            view.on_signal_changed("tau")
        self.assertIsNone(view.metric_bindings["tau"])
        self.assertIsNone(view.preview_result)
        calculator.assert_not_called()

    @mock.patch("labquake_explorer.ui.views.event_k_editor_view.calculate_event_loading_stiffness")
    def test_calculate_calls_helper_exactly_once_with_full_run_identity(self, calculator):
        view = make_headless_view()
        view.metric_bindings = {"tau": "tau source", "slip": "slip source"}
        expected = {"valid": False}
        calculator.return_value = expected
        result = view.calculate_preview()
        self.assertIs(result, expected)
        self.assertIs(view.preview_result, expected)
        calculator.assert_called_once()
        kwargs = calculator.call_args.kwargs
        self.assertIs(kwargs["time"], view.run_data["time"])
        self.assertIs(kwargs["tau_signal"], view.run_data["tau source"])
        self.assertIs(kwargs["slip_signal"], view.run_data["slip source"])
        self.assertEqual(kwargs["event_time"], 5.0)
        self.assertNotIn("event", kwargs)

    @mock.patch("labquake_explorer.ui.views.event_k_editor_view.calculate_event_loading_stiffness")
    def test_missing_event_time_or_stale_signal_prevents_analysis(self, calculator):
        view = make_headless_view()
        view.metric_bindings = {"tau": "missing", "slip": "slip source"}
        with self.assertRaisesRegex(ValueError, "tau"):
            view.calculate_preview()
        view.current_event_time = None
        with self.assertRaisesRegex(ValueError, "event_time"):
            view.calculate_preview()
        calculator.assert_not_called()

    def test_parameter_change_clears_result_without_analysis_or_bindings(self):
        view = make_headless_view()
        view.metric_bindings = {"tau": "tau source", "slip": "slip source"}
        view.preview_result = {"valid": True}
        view.preview_parameters = {"old": True}
        with mock.patch("labquake_explorer.ui.views.event_k_editor_view.calculate_event_loading_stiffness") as calculator:
            view._on_parameter_changed()
        self.assertIsNone(view.preview_result)
        self.assertEqual(view.metric_bindings, {"tau": "tau source", "slip": "slip source"})
        calculator.assert_not_called()

    def test_event_switch_keeps_parameters_and_clears_bindings(self):
        view = make_headless_view()
        old_parameters = view.parameter_vars
        view.event_combobox = FakeWidget("0")
        view.title = mock.Mock()
        view._set_event = mock.Mock(side_effect=lambda index: setattr(view, "metric_bindings", {"tau": None, "slip": None}))
        view._refresh_event_widgets = mock.Mock()
        view.on_event_changed()
        self.assertIs(view.parameter_vars, old_parameters)
        view._set_event.assert_called_once_with(0)
        view._refresh_event_widgets.assert_called_once()


class ResultAndPlotTests(unittest.TestCase):
    def valid_result(self, k=-2.5):
        relative = np.linspace(-1.0, 1.0, 9)
        slip = relative + 2.0
        tau = k * slip + 1.25
        mask = (relative >= -1.0) & (relative <= -0.5)
        return {
            "valid": True,
            "k": k,
            "intercept": 1.25,
            "coefficients": np.array([k, 1.25]),
            "relative_time": relative,
            "raw_tau": tau + 1.0,
            "raw_slip": slip + 1.0,
            "processed_tau": tau,
            "processed_slip": slip,
            "fit_mask": mask,
        }

    def test_recompute_displays_signed_k_and_intercept(self):
        view = make_headless_view()
        view.calculate_preview = mock.Mock(return_value=self.valid_result(-2.5))
        view.calculate_preview.side_effect = lambda: setattr(view, "preview_result", self.valid_result(-2.5)) or view.preview_result
        view.recompute_preview()
        self.assertEqual(view.result_vars["valid"].get(), "True")
        self.assertEqual(view.result_vars["k"].get(), "-2.5")
        self.assertEqual(view.result_vars["intercept"].get(), "1.25")
        self.assertEqual(view.status_var.get(), "Preview only — not saved")

    def test_invalid_result_is_normal_not_error_dialog(self):
        view = make_headless_view()
        view.calculate_preview = mock.Mock(side_effect=lambda: setattr(view, "preview_result", {"valid": False}) or view.preview_result)
        with mock.patch("labquake_explorer.ui.views.event_k_editor_view.messagebox.showerror") as showerror:
            view.recompute_preview()
        self.assertEqual(view.result_vars["valid"].get(), "False")
        self.assertEqual(view.result_vars["k"].get(), "—")
        showerror.assert_not_called()

    def test_value_error_clears_state_and_shows_specific_error(self):
        view = make_headless_view()
        view.calculate_preview = mock.Mock(side_effect=ValueError("bad cutoff"))
        with mock.patch("labquake_explorer.ui.views.event_k_editor_view.messagebox.showerror") as showerror:
            view.recompute_preview()
        self.assertIsNone(view.preview_result)
        self.assertEqual(view.status_var.get(), "bad cutoff")
        showerror.assert_called_once_with("Loading stiffness", "bad cutoff")

    def test_plot_uses_only_returned_arrays_mask_and_coefficients(self):
        view = make_headless_view()
        result = self.valid_result(3.0)
        view.preview_result = result
        view._plot_preview()
        np.testing.assert_array_equal(view.slip_ax.lines[0].get_ydata(), result["raw_slip"])
        np.testing.assert_array_equal(view.slip_ax.lines[1].get_ydata(), result["processed_slip"])
        np.testing.assert_array_equal(view.tau_ax.lines[0].get_ydata(), result["raw_tau"])
        np.testing.assert_array_equal(view.tau_ax.lines[1].get_ydata(), result["processed_tau"])
        self.assertEqual(len(view.fit_ax.collections), 1)
        self.assertEqual(len(view.fit_ax.lines), 1)

    def test_invalid_plot_has_no_fit_line(self):
        view = make_headless_view()
        view.preview_result = {"valid": False}
        view._plot_preview()
        self.assertEqual(len(view.fit_ax.lines), 0)


class LifecycleAndWiringTests(unittest.TestCase):
    def test_view_is_exported_and_controller_method_exists(self):
        self.assertIs(ExportedEventKEditorView, EventKEditorView)
        self.assertTrue(callable(LabquakeExplorer.analyze_event_loading_stiffness))

    @mock.patch("labquake_explorer.ui.labquake_explorer.EventKEditorView")
    def test_controller_opens_view_without_analysis_or_persistence(self, view_class):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", "[7]"))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        view = view_class.return_value
        with mock.patch("labquake_explorer.data.data_manager.DataManager.set_data") as set_data:
            explorer.analyze_event_loading_stiffness()
        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])
        set_data.assert_not_called()

    def test_context_menu_contains_loading_stiffness_entry(self):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.root = object()
        menus = []

        class FakeMenu:
            def __init__(self, *args, **kwargs):
                self.commands = []
                menus.append(self)

            def add_command(self, **kwargs):
                self.commands.append(kwargs)

        with mock.patch("labquake_explorer.ui.labquake_explorer.tk.Menu", FakeMenu):
            explorer.create_context_menus()
        labels = [command["label"] for command in explorer.event_menu.commands]
        self.assertIn("Analyze Loading Stiffness", labels)

    def test_endpoint_drag_updates_control_and_disconnects(self):
        figure = Figure()
        axis = figure.add_subplot()
        line = axis.axvline(-1.0)
        changed = mock.Mock()
        released = mock.Mock()
        draggable = _DraggablePreEndpoint(line, changed, released, lambda value: min(value, 0.0))
        draggable.dragging = True
        motion = mock.Mock(inaxes=axis, xdata=0.5)
        draggable._on_motion(motion)
        changed.assert_called_once_with(0.0)
        draggable._on_release(mock.Mock())
        released.assert_called_once()
        draggable.disconnect()
        self.assertEqual(draggable._connection_ids, [])

    def test_close_disconnects_and_removes_child_without_saving(self):
        view = EventKEditorView.__new__(EventKEditorView)
        view.parent = mock.Mock()
        view.parent.child_windows = [view]
        view._disconnect_endpoint_lines = mock.Mock()
        view.destroy = mock.Mock()
        view.on_close()
        view._disconnect_endpoint_lines.assert_called_once()
        self.assertNotIn(view, view.parent.child_windows)
        view.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
