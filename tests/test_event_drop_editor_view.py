import copy
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.figure import Figure

from labquake_explorer.ui import labquake_explorer as explorer_module
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
)


class FakeVariable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class TraceVariable(FakeVariable):
    def __init__(self, value, callback):
        super().__init__(value)
        self.callback = callback

    def set(self, value):
        super().set(value)
        self.callback()


class RaisingVariable(FakeVariable):
    def set(self, value):
        raise RuntimeError("control update failed")


class FakeWidget:
    def __init__(self, value=""):
        self.value = value
        self.options = {}

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def configure(self, **kwargs):
        self.options.update(kwargs)


def parameter_vars(**overrides):
    values = {
        "half_win": "1.5",
        "pre_start": "-1",
        "pre_end": "-0.5",
        "post_start": "0.5",
        "post_end": "1",
        "smooth_w": "",
    }
    values.update(overrides)
    return {key: FakeVariable(value) for key, value in values.items()}


def result_vars():
    return {
        key: FakeVariable("—")
        for key in (
            "signal_name",
            "valid",
            "delta",
            "magnitude",
            "val_pre_0",
            "val_post_0",
        )
    }


def headless_view():
    view = EventDropEditorView.__new__(EventDropEditorView)
    view.signal_candidates = {}
    view.selected_signal_name = None
    view.preview_result = None
    view.preview_parameters = None
    view._updating_endpoint_control = False
    view._active_endpoint = None
    view.parameter_vars = parameter_vars()
    view.result_vars = result_vars()
    view.status_var = FakeVariable()
    view.preview_button = FakeWidget()
    view.signal_combobox = FakeWidget()
    view.interevent_bindings = {"dmax": None, "reference": None}
    view.interevent_preview_results = {}
    view.interevent_preview_parameters = None
    view.interevent_result_vars = {
        metric: {"valid": FakeVariable("—"), "value": FakeVariable("—")}
        for metric in ("D_Push", "D_max", "D_reference")
    }
    view.interevent_status_var = FakeVariable()
    view.interevent_preview_button = FakeWidget()
    view.dmax_signal_combobox = FakeWidget()
    view.reference_signal_combobox = FakeWidget()
    view.interevent_parameter_vars = {
        "push_speed": FakeVariable("3.508"),
        "delay_sec": FakeVariable("0.05"),
        "dmax_smooth_w": FakeVariable("100"),
    }
    view.full_run_signal_candidates = []
    view.current_event_time = 2.0
    view.previous_event_time = 1.0
    view.run_data = None
    view.event = {}
    view._plot_preview = mock.Mock()
    return view


class SignalCandidateTests(unittest.TestCase):
    def test_candidates_are_explicit_aligned_real_numeric_signals(self):
        source = np.array([1.0, 2.0])
        event = {
            "time": np.array([0.0, 1.0]),
            "z": source,
            "list": [3, 4],
            "bool": np.array([True, False]),
            "complex": np.array([1 + 2j, 2 + 3j]),
            "matrix": np.ones((2, 1)),
            "short": np.array([1.0]),
            "text": ["a", "b"],
        }
        candidates = find_signal_candidates(event)
        self.assertEqual(list(candidates), ["z", "list"])
        self.assertIs(candidates["z"], source)

    def test_candidate_discovery_does_not_modify_inputs(self):
        signal = np.array([2.0, 1.0])
        before = signal.copy()
        find_signal_candidates({"time": np.array([0.0, 1.0]), "signal": signal})
        np.testing.assert_array_equal(signal, before)


class ParameterAndFormattingTests(unittest.TestCase):
    def test_preview_parameters_support_blank_and_integer_smoothing(self):
        blank = parse_preview_parameters("1", "-1", "-.5", ".5", "1", "")
        smooth = parse_preview_parameters("1", "-1", "-.5", ".5", "1", "5")
        self.assertIsNone(blank["smooth_w"])
        self.assertEqual(smooth["smooth_w"], 5)

    def test_preview_parameters_validate_windows(self):
        invalid = (
            ("-1", "-1", "-.5", ".5", "1", ""),
            ("1", ".1", "-.5", ".5", "1", ""),
            ("1", "-1", "-.5", "-.1", "1", ""),
            ("1", "-.5", "-.5", ".5", "1", ""),
            ("1", "-1", "-.5", ".5", ".5", ""),
            ("1", "-1.1", "-.5", ".5", "1", ""),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                parse_preview_parameters(*arguments)

    def test_signed_delta_and_magnitude_remain_separate(self):
        display = format_preview_result({
            "valid": True,
            "delta": -2.5,
            "val_pre_0": 1.0,
            "val_post_0": 3.5,
        })
        self.assertEqual(display["delta"], "-2.5")
        self.assertEqual(display["magnitude"], "2.5")
        self.assertEqual(format_preview_result({"valid": False})["delta"], "—")

    def test_interevent_parameters_use_dmax_name(self):
        result = parse_interevent_parameters("3.508", "0.05", "100")
        self.assertEqual(
            result,
            {"push_speed": 3.508, "delay_sec": 0.05, "dmax_smooth_w": 100},
        )
        for value in ("", "0", "-1", "1.0", "text"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "dmax_smooth_w"
            ):
                parse_interevent_parameters("1", "0", value)

    def test_interevent_format_preserves_sign(self):
        result = format_interevent_result({"valid": True, "value": -2.0})
        self.assertEqual(result["value"], "-2")


class SingleSignalWorkflowTests(unittest.TestCase):
    def test_initial_state_is_single_and_unselected(self):
        view = headless_view()
        self.assertIsNone(view.selected_signal_name)
        self.assertIsNone(view.preview_result)
        self.assertFalse(view._updating_endpoint_control)
        self.assertFalse(hasattr(view, "metric_bindings"))
        self.assertFalse(hasattr(view, "slip_bindings"))

    def test_production_view_has_no_legacy_event_local_roles(self):
        source = inspect.getsource(EventDropEditorView)
        source = source.replace("lvdt_signal=", "").replace("lvdt_smooth_w=", "")
        for legacy_name in ("tau", "mu", "lvdt", "slip"):
            with self.subTest(legacy_name=legacy_name):
                self.assertNotRegex(source.lower(), rf"\b{legacy_name}\b")
        for method_name in (
            "on_tau_signal_changed",
            "on_mu_signal_changed",
            "on_lvdt_signal_changed",
            "add_slip_sensor",
        ):
            self.assertFalse(hasattr(EventDropEditorView, method_name))

    def test_refresh_does_not_auto_select_first_candidate(self):
        view = headless_view()
        view.signal_candidates = {"first": np.ones(2), "second": np.zeros(2)}
        view._refresh_interevent_widgets = mock.Mock()
        view._refresh_event_widgets()
        self.assertEqual(
            view.signal_combobox.options["values"], ["first", "second"]
        )
        self.assertEqual(view.signal_combobox.get(), "")
        self.assertIsNone(view.selected_signal_name)
        self.assertEqual(view.preview_button.options["state"], "disabled")

    def test_valid_selection_enables_preview_and_clears_old_result(self):
        view = headless_view()
        view.signal_candidates = {"arbitrary": np.array([1.0, 2.0])}
        view.preview_result = {"valid": True}
        view.signal_combobox.set("arbitrary")
        view.on_signal_changed()
        self.assertEqual(view.selected_signal_name, "arbitrary")
        self.assertIsNone(view.preview_result)
        self.assertEqual(view.preview_button.options["state"], "normal")
        view._plot_preview.assert_called_once_with()

    def test_blank_or_stale_selection_disables_preview(self):
        for selection in ("", "missing"):
            with self.subTest(selection=selection):
                view = headless_view()
                view.signal_candidates = {"available": np.ones(2)}
                view.signal_combobox.set(selection)
                view.on_signal_changed()
                self.assertIsNone(view.selected_signal_name)
                self.assertEqual(view.preview_button.options["state"], "disabled")

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
    )
    def test_preview_uses_one_entry_orchestration_mapping_once(self, calculator):
        view = headless_view()
        signal = np.array([4.0, 2.0])
        event = {
            "time": np.array([0.0, 1.0]),
            "event_time": 0.5,
            "chosen": signal,
        }
        view.event = event
        view.signal_candidates = {"chosen": signal, "unused": np.ones(2)}
        view.selected_signal_name = "chosen"
        expected = {"valid": True, "delta": -2.0}
        calculator.return_value = {"signal": expected}
        before = copy.deepcopy(event)
        result = view.calculate_preview()
        calculator.assert_called_once()
        arguments = calculator.call_args.kwargs
        self.assertEqual(list(arguments["signals"]), ["signal"])
        self.assertIs(arguments["signals"]["signal"], signal)
        self.assertEqual(list(arguments["parameters"]), ["signal"])
        self.assertIs(result, expected)
        self.assertIs(view.preview_result, expected)
        np.testing.assert_array_equal(event["chosen"], before["chosen"])

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
    )
    def test_stale_selection_prevents_analysis(self, calculator):
        view = headless_view()
        view.selected_signal_name = "removed"
        with self.assertRaisesRegex(ValueError, "selected"):
            view.calculate_preview()
        calculator.assert_not_called()

    def test_recompute_displays_single_result(self):
        view = headless_view()
        view.selected_signal_name = "chosen"
        view.calculate_preview = mock.Mock(
            return_value={
                "valid": True,
                "delta": -3.0,
                "val_pre_0": 1.0,
                "val_post_0": 4.0,
            }
        )
        view.recompute_preview()
        self.assertEqual(view.result_vars["signal_name"].get(), "chosen")
        self.assertEqual(view.result_vars["delta"].get(), "-3")
        self.assertEqual(view.result_vars["magnitude"].get(), "3")

    def test_signal_and_parameter_changes_do_not_touch_d_results(self):
        view = headless_view()
        d_results = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }
        view.interevent_preview_results = d_results.copy()
        view.signal_candidates = {"new": np.ones(2)}
        view.signal_combobox.set("new")
        view.on_signal_changed()
        view._invalidate_preview()
        self.assertEqual(view.interevent_preview_results, d_results)


class PlotAndDraggingTests(unittest.TestCase):
    def make_plot_view(self):
        view = headless_view()
        view.figure = Figure()
        view.raw_ax = view.figure.add_subplot(211)
        view.fit_ax = view.figure.add_subplot(212, sharex=view.raw_ax)
        view.canvas = SimpleNamespace(draw_idle=mock.Mock())
        view._endpoint_draggables = {}
        view._active_endpoint = None
        view._plot_preview = EventDropEditorView._plot_preview.__get__(view)
        return view

    def test_plot_uses_selected_raw_signal_and_single_result(self):
        view = self.make_plot_view()
        signal = np.array([10.0, 11.0, 7.0, 8.0])
        view.event = {
            "time": np.array([-1.0, -0.5, 0.5, 1.0]),
            "event_time": 0.0,
        }
        view.signal_candidates = {"chosen": signal}
        view.selected_signal_name = "chosen"
        view.preview_result = {
            "valid": True,
            "coeff_pre": np.array([1.0, 2.0]),
            "coeff_post": np.array([-1.0, 1.0]),
        }
        view._plot_preview()
        np.testing.assert_array_equal(view.raw_ax.lines[0].get_ydata(), signal)
        self.assertEqual(view.raw_ax.get_ylabel(), "chosen")
        self.assertGreaterEqual(len(view.fit_ax.lines), 3)

    def test_parameter_change_keeps_raw_and_removes_fit(self):
        view = self.make_plot_view()
        view.event = {"time": np.array([-1.0, 1.0]), "event_time": 0.0}
        view.signal_candidates = {"chosen": np.array([1.0, 2.0])}
        view.selected_signal_name = "chosen"
        view.preview_result = {"valid": True}
        view._on_parameter_changed()
        self.assertIsNone(view.preview_result)
        np.testing.assert_array_equal(
            view.raw_ax.lines[0].get_ydata(), [1.0, 2.0]
        )

    def test_endpoint_change_updates_single_parameter(self):
        view = headless_view()
        view.preview_result = {"valid": True}
        view.preview_parameters = {"old": True}
        view.result_vars["valid"].set("True")
        view._plot_preview = mock.Mock()
        view._on_endpoint_changed("pre_start", -0.75)
        self.assertEqual(view.parameter_vars["pre_start"].get(), "-0.75")
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.preview_parameters)
        self.assertEqual(view.result_vars["valid"].get(), "—")
        self.assertEqual(
            view.status_var.get(),
            "Fitting windows changed — recompute preview",
        )
        view._plot_preview.assert_not_called()
        self.assertFalse(view._updating_endpoint_control)

    def test_endpoint_control_trace_is_suppressed_during_motion(self):
        view = headless_view()
        view._plot_preview = mock.Mock()
        view._on_parameter_changed = mock.Mock(wraps=view._on_parameter_changed)
        view.parameter_vars["pre_start"] = TraceVariable(
            "-1", view._on_parameter_changed
        )
        with (
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_event_drop_metrics"
            ) as event_calculator,
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_interevent_displacement_metrics"
            ) as interevent_calculator,
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "messagebox.showerror"
            ) as showerror,
        ):
            view._on_endpoint_changed("pre_start", -0.625)
        view._on_parameter_changed.assert_called_once_with()
        view._plot_preview.assert_not_called()
        event_calculator.assert_not_called()
        interevent_calculator.assert_not_called()
        showerror.assert_not_called()
        self.assertFalse(view._updating_endpoint_control)

    def test_endpoint_control_flag_is_restored_when_update_fails(self):
        view = headless_view()
        view.parameter_vars["pre_start"] = RaisingVariable("-1")
        with self.assertRaisesRegex(RuntimeError, "control update failed"):
            view._on_endpoint_changed("pre_start", -0.5)
        self.assertFalse(view._updating_endpoint_control)

    def test_manual_parameter_edit_redraws_once_without_analysis(self):
        view = headless_view()
        view.preview_result = {"valid": True}
        view.preview_parameters = {"old": True}
        view.raw_ax = mock.Mock()
        view._plot_preview = mock.Mock()
        with (
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_event_drop_metrics"
            ) as event_calculator,
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_interevent_displacement_metrics"
            ) as interevent_calculator,
        ):
            view._on_parameter_changed()
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.preview_parameters)
        view._plot_preview.assert_called_once_with()
        event_calculator.assert_not_called()
        interevent_calculator.assert_not_called()

    def test_repeated_motion_keeps_connections_and_release_redraws_once(self):
        view = headless_view()
        view.parameter_vars["pre_start"] = TraceVariable(
            "-1", view._on_parameter_changed
        )
        view._plot_preview = mock.Mock()
        figure = Figure()
        axis = figure.add_subplot(111)
        line = axis.axvline(-1.0)
        draggable = _DraggableVerticalLine(
            line=line,
            on_changed=lambda value: view._on_endpoint_changed(
                "pre_start", value
            ),
            on_released=lambda: view._on_endpoint_released("pre_start"),
            constrain=lambda value: value,
            on_started=lambda: view._begin_endpoint_drag("pre_start"),
        )
        original_connections = list(draggable._connection_ids)
        draggable.dragging = True
        view._active_endpoint = "pre_start"

        with (
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_event_drop_metrics"
            ) as event_calculator,
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "calculate_interevent_displacement_metrics"
            ) as interevent_calculator,
            mock.patch(
                "labquake_explorer.ui.views.event_drop_editor_view."
                "messagebox.showerror"
            ) as showerror,
        ):
            for position in (-0.9, -0.7, -0.4, -0.25):
                draggable._on_motion(mock.Mock(inaxes=axis, xdata=position))
                self.assertTrue(draggable.dragging)
                self.assertEqual(
                    draggable._connection_ids, original_connections
                )
                view._plot_preview.assert_not_called()
            draggable._on_release(mock.Mock())

        self.assertEqual(view.parameter_vars["pre_start"].get(), "-0.25")
        np.testing.assert_allclose(line.get_xdata(), [-0.25, -0.25])
        self.assertFalse(draggable.dragging)
        self.assertIsNone(view._active_endpoint)
        view._plot_preview.assert_called_once_with()
        event_calculator.assert_not_called()
        interevent_calculator.assert_not_called()
        showerror.assert_not_called()

    def test_press_motion_motion_release_lifecycle(self):
        view = headless_view()
        view.parameter_vars["pre_start"] = TraceVariable(
            "-1", view._on_parameter_changed
        )
        view._plot_preview = mock.Mock()
        figure = Figure()
        axis = figure.add_subplot(111)
        axis.set_xlim(-2.0, 1.0)
        line = axis.axvline(-1.0)
        figure.canvas.draw()
        draggable = _DraggableVerticalLine(
            line=line,
            on_changed=lambda value: view._on_endpoint_changed(
                "pre_start", value
            ),
            on_released=lambda: view._on_endpoint_released("pre_start"),
            constrain=lambda value: value,
            on_started=lambda: view._begin_endpoint_drag("pre_start"),
        )
        line_pixel_x = axis.transData.transform((-1.0, 0.0))[0]

        draggable._on_press(
            mock.Mock(
                inaxes=axis,
                xdata=-1.0,
                x=line_pixel_x,
                button=1,
            )
        )
        self.assertTrue(draggable.dragging)
        self.assertEqual(view._active_endpoint, "pre_start")
        for position in (-0.8, -0.4):
            draggable._on_motion(mock.Mock(inaxes=axis, xdata=position))
        view._plot_preview.assert_not_called()
        draggable._on_release(mock.Mock())

        self.assertFalse(draggable.dragging)
        self.assertIsNone(view._active_endpoint)
        self.assertEqual(view.parameter_vars["pre_start"].get(), "-0.4")
        view._plot_preview.assert_called_once_with()

    def test_release_rebuilds_four_draggables_without_accumulation(self):
        view = self.make_plot_view()
        view.event = {"time": np.array([-1.0, 1.0]), "event_time": 0.0}
        view.signal_candidates = {"chosen": np.array([1.0, 2.0])}
        view.selected_signal_name = "chosen"
        view._plot_preview()
        first_generation = dict(view._endpoint_draggables)
        self.assertEqual(len(first_generation), 4)
        view._active_endpoint = "pre_start"

        view._on_endpoint_released("pre_start")

        self.assertIsNone(view._active_endpoint)
        self.assertEqual(len(view._endpoint_draggables), 4)
        for draggable in first_generation.values():
            self.assertFalse(draggable.connected)
        self.assertTrue(
            all(
                draggable.connected
                for draggable in view._endpoint_draggables.values()
            )
        )

    def test_draggable_line_disconnects(self):
        figure = Figure()
        axis = figure.add_subplot(111)
        line = axis.axvline(0.0)
        draggable = _DraggableVerticalLine(
            line, mock.Mock(), mock.Mock(), lambda value: value
        )
        self.assertTrue(draggable.connected)
        draggable.disconnect()
        self.assertFalse(draggable.connected)

    def test_close_disconnects_all_endpoint_callbacks(self):
        view = headless_view()
        view.parent = SimpleNamespace(child_windows=[view])
        view._disconnect_endpoint_lines = mock.Mock()
        view.destroy = mock.Mock()
        view.on_close()
        view._disconnect_endpoint_lines.assert_called_once_with()
        self.assertNotIn(view, view.parent.child_windows)
        view.destroy.assert_called_once_with()


class IntereventBindingTests(unittest.TestCase):
    def test_dmax_and_reference_bindings_are_independent_and_explicit(self):
        view = headless_view()
        view.full_run_signal_candidates = ["run_signal"]
        view.dmax_signal_combobox.set("run_signal")
        view.reference_signal_combobox.set("run_signal")
        view.interevent_preview_results = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }
        view.on_dmax_signal_changed()
        self.assertEqual(view.interevent_bindings["dmax"], "run_signal")
        self.assertNotIn("D_max", view.interevent_preview_results)
        view.on_reference_signal_changed()
        self.assertEqual(view.interevent_bindings["reference"], "run_signal")
        self.assertNotIn("D_reference", view.interevent_preview_results)
        self.assertIn("D_Push", view.interevent_preview_results)

    def test_blank_and_stale_dmax_binding_clear_only_dmax(self):
        for value in ("", "missing"):
            with self.subTest(value=value):
                view = headless_view()
                view.full_run_signal_candidates = ["valid"]
                view.dmax_signal_combobox.set(value)
                view.on_dmax_signal_changed()
                self.assertIsNone(view.interevent_bindings["dmax"])

    def test_refresh_uses_full_run_candidates_and_preserves_valid_bindings(self):
        view = headless_view()
        view.full_run_signal_candidates = ["kept", "other"]
        view.interevent_bindings = {"dmax": "kept", "reference": "kept"}
        view._refresh_interevent_widgets()
        self.assertEqual(
            view.dmax_signal_combobox.options["values"], ["kept", "other"]
        )
        self.assertEqual(view.dmax_signal_combobox.get(), "kept")
        self.assertEqual(view.reference_signal_combobox.get(), "kept")

    def test_refresh_clears_stale_run_bindings(self):
        view = headless_view()
        view.full_run_signal_candidates = ["available"]
        view.interevent_bindings = {"dmax": "gone", "reference": "gone"}
        view._refresh_interevent_widgets()
        self.assertEqual(
            view.interevent_bindings, {"dmax": None, "reference": None}
        )

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view.calculate_interevent_displacement_metrics"
    )
    def test_interevent_call_uses_run_arrays_and_existing_api_keywords(
        self, calculator
    ):
        view = headless_view()
        time = np.arange(5.0)
        dmax = np.arange(5.0) * 2
        reference = np.arange(5.0) * 3
        view.run_data = {"time": time, "d": dmax, "r": reference}
        view.interevent_bindings = {"dmax": "d", "reference": "r"}
        calculator.return_value = {
            "D_Push": {"valid": True},
            "D_max": {"valid": True},
            "D_reference": {"valid": True},
        }
        view.calculate_interevent_preview()
        calculator.assert_called_once()
        arguments = calculator.call_args.kwargs
        self.assertIs(arguments["time"], time)
        self.assertIs(arguments["lvdt_signal"], dmax)
        self.assertIs(arguments["reference_displacement_signal"], reference)
        self.assertEqual(arguments["lvdt_smooth_w"], 100)

    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view.calculate_interevent_displacement_metrics"
    )
    def test_dpush_only_passes_no_run_arrays(self, calculator):
        view = headless_view()
        calculator.return_value = {
            "D_Push": {"valid": True},
            "D_max": {"valid": False},
            "D_reference": {"valid": False},
        }
        view.calculate_interevent_preview()
        arguments = calculator.call_args.kwargs
        self.assertIsNone(arguments["time"])
        self.assertIsNone(arguments["lvdt_signal"])
        self.assertIsNone(arguments["reference_displacement_signal"])

    def test_invalidation_dependency_matrix(self):
        expected = {
            "push_speed": {"D_max", "D_reference"},
            "delay_sec": {"D_Push"},
            "dmax_smooth_w": {"D_Push", "D_reference"},
        }
        for parameter, remaining in expected.items():
            with self.subTest(parameter=parameter):
                view = headless_view()
                view.interevent_preview_results = {
                    metric: {"valid": True}
                    for metric in ("D_Push", "D_max", "D_reference")
                }
                view._on_interevent_parameter_changed(parameter)
                self.assertEqual(set(view.interevent_preview_results), remaining)


class EventSwitchingTests(unittest.TestCase):
    def test_set_event_clears_event_local_state_and_preserves_run_bindings(self):
        view = headless_view()
        view.run_idx = 2
        view.interevent_bindings = {"dmax": "kept", "reference": "kept"}
        view.selected_signal_name = "old"
        view.preview_result = {"valid": True}
        event = {
            "time": np.array([0.0, 1.0]),
            "event_time": 0.5,
            "new": np.ones(2),
        }
        view.data_manager = SimpleNamespace(get_data=mock.Mock(return_value=event))
        view._refresh_interevent_context = mock.Mock()
        view._set_event(4)
        self.assertIsNone(view.selected_signal_name)
        self.assertIsNone(view.preview_result)
        self.assertEqual(
            view.interevent_bindings, {"dmax": "kept", "reference": "kept"}
        )
        self.assertEqual(list(view.signal_candidates), ["new"])


class CanonicalEventContextTests(unittest.TestCase):
    def test_event_time_coercion_accepts_only_finite_nonboolean_scalars(self):
        accepted = (1.25, 2, np.float64(3.5), np.int64(4), 0, -2.5)
        for value in accepted:
            with self.subTest(value=value):
                result = EventDropEditorView._coerce_finite_event_time(value)
                self.assertIsInstance(result, float)
                self.assertEqual(result, float(value))

        rejected = (
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
        for value in rejected:
            with self.subTest(value=repr(value)):
                self.assertIsNone(
                    EventDropEditorView._coerce_finite_event_time(value)
                )

    def test_current_event_time_uses_only_top_level_key_without_mutation(self):
        view = headless_view()
        event = {
            "event_time": np.float64(-1.5),
            "time": np.array([99.0, 100.0]),
        }
        original_time = event["time"].copy()
        view.event = event
        self.assertEqual(view._get_current_event_time(), -1.5)
        np.testing.assert_array_equal(event["time"], original_time)
        self.assertEqual(event["event_time"], np.float64(-1.5))

        for invalid_event in ({"time": np.array([7.0])}, [], "event"):
            with self.subTest(event=repr(invalid_event)):
                view.event = invalid_event
                self.assertIsNone(view._get_current_event_time())

    def test_previous_event_search_follows_list_position_and_skips_invalid(self):
        invalid_values = (
            {"event_time": np.nan},
            "not an event",
            {"event_time": True},
            {"event_time": np.array(3.0)},
            {"missing": 1},
        )
        events = [
            {"event_time": 10.0},
            *invalid_values,
            {"event_time": -5.0, "skipped": True, "deleted": True},
            {"event_time": 100.0},
        ]
        original = copy.deepcopy(events)
        view = headless_view()
        view.events = events
        view.event_idx = len(events) - 1
        self.assertEqual(view._find_previous_event_time(), -5.0)
        self.assertEqual(events[0]["event_time"], original[0]["event_time"])
        self.assertEqual(events[-2], original[-2])

    def test_previous_event_search_boundaries_and_supported_containers(self):
        view = headless_view()
        events = [{"event_time": 3.0}, {"event_time": 1.0}]
        cases = (
            (events, 0, None),
            (events, 1, 3.0),
            (events, 2, None),
            (events, True, None),
            ({0: events[0]}, 1, None),
            ("events", 1, None),
            (np.array(events, dtype=object).reshape(1, 2), 1, None),
            ([{"event_time": np.nan}, {}], 1, None),
        )
        for container, index, expected in cases:
            with self.subTest(container=type(container), index=index):
                view.events = container
                view.event_idx = index
                self.assertEqual(view._find_previous_event_time(), expected)

        view.events = np.array(events, dtype=object)
        view.event_idx = 1
        self.assertEqual(view._find_previous_event_time(), 3.0)


class FullRunResolutionTests(unittest.TestCase):
    def test_full_run_time_is_returned_by_identity_without_validation(self):
        view = headless_view()
        for time in (
            [2.0, 1.0, 1.0],
            np.array([2.0, 1.0, 1.0]),
        ):
            with self.subTest(kind=type(time)):
                view.run_data = {"time": time}
                self.assertIs(view._get_full_run_time(), time)

        view.event = {"time": np.array([1.0])}
        for run_data in ({}, None, []):
            view.run_data = run_data
            self.assertIsNone(view._get_full_run_time())

    def test_exact_full_run_signal_resolution_has_no_alias_or_inference(self):
        view = headless_view()
        array = np.array([1.0, 2.0])
        values = [3.0, 4.0]
        nested = {"target": np.array([5.0])}
        run = {
            "Exact": array,
            "list": values,
            "nested": nested,
            "LP_displacement": np.array([6.0, 7.0]),
        }
        view.run_data = run
        self.assertIs(view._resolve_full_run_signal("Exact"), array)
        self.assertIs(view._resolve_full_run_signal("list"), values)
        for key in ("", None, 1, "exact", "LVDT", "Eddy", "target"):
            with self.subTest(key=key):
                self.assertIsNone(view._resolve_full_run_signal(key))
        self.assertIs(run["Exact"], array)
        self.assertIs(run["nested"], nested)
        view.run_data = None
        self.assertIsNone(view._resolve_full_run_signal("Exact"))

    def test_run_array_validator_covers_shape_dtype_finiteness_and_conversion(self):
        class BadArray:
            def __array__(self, dtype=None):
                raise TypeError("cannot convert")

        valid = ([1, 2], np.array([1.0, 2.0]), np.array([1, 2]))
        for value in valid:
            with self.subTest(valid=repr(value)):
                self.assertEqual(
                    EventDropEditorView._is_finite_real_run_array(value),
                    (True, 2),
                )

        invalid = (
            1.0,
            np.array(1.0),
            np.ones((1, 2)),
            [],
            np.array([], dtype=float),
            np.array([True, False]),
            np.array([1 + 2j]),
            np.array(["1", "2"]),
            np.array([object()], dtype=object),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
            {"nested": [1.0]},
            BadArray(),
        )
        for value in invalid:
            with self.subTest(invalid=repr(value)):
                self.assertEqual(
                    EventDropEditorView._is_finite_real_run_array(value),
                    (False, 0),
                )

    def test_candidate_filtering_preserves_order_identity_and_inputs(self):
        time = np.array([2.0, 1.0, 1.0])
        zulu = np.array([1.0, 2.0, 3.0])
        alpha = np.array([4, 5, 6])
        run = {
            "time": time,
            "zulu": zulu,
            "alpha": alpha,
            "bool": np.array([True, False, True]),
            "complex": np.array([1 + 0j] * 3),
            "object": np.array([1, 2, 3], dtype=object),
            "scalar": 1.0,
            "two_d": np.ones((3, 1)),
            "empty": np.array([]),
            "short": np.ones(2),
            "nan": np.array([1.0, np.nan, 3.0]),
            "inf": np.array([1.0, np.inf, 3.0]),
            "nested": {"signal": np.ones(3)},
            99: np.ones(3),
        }
        originals = {key: value.copy() for key, value in run.items() if isinstance(value, np.ndarray)}
        view = headless_view()
        view.run_data = run
        self.assertEqual(view._find_full_run_signal_candidates(), ["zulu", "alpha"])
        self.assertIs(run["zulu"], zulu)
        self.assertIs(run["alpha"], alpha)
        for key, original in originals.items():
            np.testing.assert_array_equal(run[key], original)

        integer_time = np.array([2, 1, 1])
        integer_signal = [7, 8, 9]
        view.run_data = {"time": integer_time, "signal": integer_signal}
        self.assertEqual(view._find_full_run_signal_candidates(), ["signal"])
        self.assertIs(view.run_data["time"], integer_time)
        self.assertIs(view.run_data["signal"], integer_signal)

    def test_invalid_run_time_prevents_all_candidates(self):
        invalid_times = (
            None,
            1.0,
            np.array(1.0),
            np.ones((2, 2)),
            [],
            np.array([True]),
            np.array(["1"]),
            np.array([object()], dtype=object),
            np.array([np.nan]),
            np.array([np.inf]),
        )
        view = headless_view()
        for time in invalid_times:
            with self.subTest(time=repr(time)):
                view.run_data = {"time": time, "signal": np.ones(1)}
                self.assertEqual(view._find_full_run_signal_candidates(), [])
        view.run_data = None
        self.assertEqual(view._find_full_run_signal_candidates(), [])


class IntereventContextRefreshTests(unittest.TestCase):
    def test_refresh_reads_exact_run_once_and_preserves_unrelated_state(self):
        view = headless_view()
        view.run_idx = 3
        view.event_idx = 2
        view.event = {"event_time": 5.0, "time": np.array([4.0, 5.0])}
        events = [{"event_time": 1.0}, {"event_time": 4.0}, view.event]
        time = np.arange(6.0)
        signal = np.arange(6.0) * 2
        run = {"events": events, "time": time, "signal": signal}
        view.data_manager = mock.Mock()
        view.data_manager.get_data.return_value = run
        view.selected_signal_name = "event signal"
        view.preview_result = {"valid": True}
        view.preview_parameters = {"half_win": 1.0}
        view.interevent_bindings = {"dmax": "signal", "reference": "signal"}
        controls = view.parameter_vars

        with mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
        ) as event_calc, mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_interevent_displacement_metrics"
        ) as d_calc:
            view._refresh_interevent_context()

        view.data_manager.get_data.assert_called_once_with("runs/[3]")
        self.assertIs(view.run_data, run)
        self.assertIs(view.events, events)
        self.assertEqual(view.current_event_time, 5.0)
        self.assertEqual(view.previous_event_time, 4.0)
        self.assertEqual(view.full_run_signal_candidates, ["signal"])
        self.assertEqual(view.selected_signal_name, "event signal")
        self.assertEqual(view.preview_result, {"valid": True})
        self.assertEqual(view.preview_parameters, {"half_win": 1.0})
        self.assertEqual(view.interevent_bindings, {"dmax": "signal", "reference": "signal"})
        self.assertIs(view.parameter_vars, controls)
        event_calc.assert_not_called()
        d_calc.assert_not_called()
        self.assertFalse(any(call[0] == "set_data" for call in view.data_manager.method_calls))
        self.assertIs(run["events"], events)
        self.assertIs(run["time"], time)
        self.assertIs(run["signal"], signal)

    def test_unavailable_run_context_handles_only_expected_read_errors(self):
        expected_failures = (None, [], KeyError("run"), IndexError("run"), TypeError("run"), ValueError("run"))
        for outcome in expected_failures:
            with self.subTest(outcome=type(outcome).__name__):
                view = headless_view()
                view.run_idx = 1
                view.event_idx = 0
                view.event = {"event_time": 2.0}
                get_data = mock.Mock()
                if isinstance(outcome, Exception):
                    get_data.side_effect = outcome
                else:
                    get_data.return_value = outcome
                view.data_manager = SimpleNamespace(get_data=get_data)
                view._refresh_interevent_context()
                self.assertIsNone(view.run_data)
                self.assertIsNone(view.events)
                self.assertEqual(view.current_event_time, 2.0)
                self.assertIsNone(view.previous_event_time)
                self.assertEqual(view.full_run_signal_candidates, [])

        view = headless_view()
        view.run_idx = 1
        view.data_manager = SimpleNamespace(
            get_data=mock.Mock(side_effect=RuntimeError("programming error"))
        )
        with self.assertRaisesRegex(RuntimeError, "programming error"):
            view._refresh_interevent_context()


class FocusedEventSwitchingTests(unittest.TestCase):
    def _switching_view(self):
        view = headless_view()
        view.run_idx = 2
        view.event_idx = 0
        view.event_combobox = FakeWidget("1")
        view.title = mock.Mock()
        view.interevent_bindings = {"dmax": "kept", "reference": "stale"}
        view.interevent_preview_results = {
            metric: {"valid": True}
            for metric in ("D_Push", "D_max", "D_reference")
        }
        view.interevent_preview_parameters = {"delay_sec": 0.1}
        return view

    def test_event_switch_refreshes_canonical_context_without_analysis_or_write(self):
        view = self._switching_view()
        new_event = {
            "event_time": 3.0,
            "time": np.array([2.0, 3.0]),
            "new signal": np.ones(2),
        }
        events = [{"event_time": 1.0}, new_event]
        run = {"events": events, "time": np.arange(4.0), "kept": np.arange(4.0)}

        def get_data(path):
            if path == "runs/[2]/events/[1]":
                return new_event
            if path == "runs/[2]":
                return run
            raise AssertionError(path)

        view.data_manager = mock.Mock()
        view.data_manager.get_data.side_effect = get_data
        fitting_controls = view.parameter_vars
        with mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
        ) as event_calc, mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_interevent_displacement_metrics"
        ) as d_calc:
            view.on_event_changed()

        self.assertEqual(view.event_idx, 1)
        self.assertIs(view.event, new_event)
        self.assertEqual(list(view.signal_candidates), ["new signal"])
        self.assertIsNone(view.selected_signal_name)
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.preview_parameters)
        self.assertEqual(view.signal_combobox.get(), "")
        self.assertIs(view.parameter_vars, fitting_controls)
        self.assertEqual(view.interevent_bindings, {"dmax": "kept", "reference": None})
        self.assertEqual(view.interevent_preview_results, {})
        self.assertIsNone(view.interevent_preview_parameters)
        self.assertEqual(view.current_event_time, 3.0)
        self.assertEqual(view.previous_event_time, 1.0)
        self.assertEqual(view.dmax_signal_combobox.get(), "kept")
        self.assertEqual(view.reference_signal_combobox.get(), "")
        view.title.assert_called_once_with("Event Drop Preview - Event 1")
        event_calc.assert_not_called()
        d_calc.assert_not_called()
        self.assertFalse(any(call[0] == "set_data" for call in view.data_manager.method_calls))

    def test_valid_run_bindings_survive_run_object_identity_change(self):
        view = self._switching_view()
        view.interevent_bindings = {"dmax": "same", "reference": "same"}
        first = {"time": np.arange(2.0), "same": np.ones(2)}
        second = {"time": np.arange(3.0), "same": np.ones(3)}
        view.run_data = first
        view.full_run_signal_candidates = ["same"]
        view._refresh_interevent_widgets()
        view.run_data = second
        view.full_run_signal_candidates = ["same"]
        view._refresh_interevent_widgets()
        self.assertEqual(view.interevent_bindings, {"dmax": "same", "reference": "same"})

    def test_event_local_key_does_not_rescue_stale_run_binding(self):
        view = self._switching_view()
        view.event = {"local only": np.ones(2), "time": np.arange(2.0)}
        view.interevent_bindings = {"dmax": "local only", "reference": "LOCAL ONLY"}
        view.full_run_signal_candidates = []
        view._refresh_interevent_widgets()
        self.assertEqual(view.interevent_bindings, {"dmax": None, "reference": None})

    @mock.patch("labquake_explorer.ui.views.event_drop_editor_view.messagebox.showerror")
    def test_on_event_changed_broadly_reports_expected_and_programming_errors(self, showerror):
        for error in (ValueError("bad selector"), KeyError("event"), RuntimeError("bug"), AttributeError("bug")):
            with self.subTest(error=type(error).__name__):
                view = self._switching_view()
                view._set_event = mock.Mock(side_effect=error)
                view.on_event_changed()
        self.assertEqual(showerror.call_count, 4)

    def test_event_selector_uses_zero_based_canonical_positions(self):
        view = headless_view()
        events = [{}, {}, {}]
        view.run_idx = 4
        view.event_idx = 2
        view.event_combobox = FakeWidget()
        view.data_manager = SimpleNamespace(get_data=mock.Mock(return_value=events))
        view._initialize_event_selector()
        view.data_manager.get_data.assert_called_once_with("runs/[4]/events")
        self.assertEqual(view.event_combobox.options["values"], ["0", "1", "2"])
        self.assertEqual(view.event_combobox.get(), "2")


class EventDropControllerWiringTests(unittest.TestCase):
    def test_view_export_and_controller_callback_exist(self):
        self.assertIs(ExportedEventDropEditorView, EventDropEditorView)
        self.assertTrue(hasattr(LabquakeExplorer, "analyze_event_drop"))

    def test_context_menu_registers_event_drop_callback(self):
        menus = []

        class FakeMenu:
            def __init__(self, *args, **kwargs):
                self.commands = []
                menus.append(self)

            def add_command(self, **kwargs):
                self.commands.append(kwargs)

        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.root = object()
        with mock.patch.object(explorer_module.tk, "Menu", FakeMenu):
            explorer.create_context_menus()
        event_commands = {item["label"]: item["command"] for item in menus[1].commands}
        self.assertEqual(event_commands["Analyze Event Drop"], explorer.analyze_event_drop)

    @mock.patch.object(explorer_module, "EventDropEditorView")
    def test_windows_path_wiring_uses_shared_parser_without_analysis_or_write(self, view_class):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", object()))
        explorer._extract_run_event_indices = mock.Mock(return_value=(3, 7))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        explorer.data_manager = mock.Mock()
        view = object()
        view_class.return_value = view

        with mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
        ) as event_calc, mock.patch(
            "labquake_explorer.ui.views.event_drop_editor_view.calculate_interevent_displacement_metrics"
        ) as d_calc:
            explorer.analyze_event_drop()

        explorer._extract_run_event_indices.assert_called_once_with(r"runs\[3]\events\[7]")
        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])
        event_calc.assert_not_called()
        d_calc.assert_not_called()
        self.assertFalse(explorer.data_manager.method_calls)


class NoPersistenceTests(unittest.TestCase):
    @mock.patch(
        "labquake_explorer.ui.views.event_drop_editor_view.calculate_event_drop_metrics"
    )
    def test_preview_never_calls_data_manager_write_api(self, calculator):
        view = headless_view()
        signal = np.array([1.0, 2.0])
        view.event = {"time": np.array([0.0, 1.0]), "event_time": 0.5}
        view.signal_candidates = {"chosen": signal}
        view.selected_signal_name = "chosen"
        view.data_manager = mock.Mock()
        calculator.return_value = {"signal": {"valid": False}}
        view.calculate_preview()
        self.assertFalse(view.data_manager.method_calls)

    def test_signal_and_interevent_selector_changes_do_not_write(self):
        view = headless_view()
        view.data_manager = mock.Mock()
        view.signal_candidates = {"event signal": np.ones(2)}
        view.signal_combobox.set("event signal")
        view.on_signal_changed()
        view.full_run_signal_candidates = ["run signal"]
        view.dmax_signal_combobox.set("run signal")
        view.reference_signal_combobox.set("run signal")
        view.on_dmax_signal_changed()
        view.on_reference_signal_changed()
        self.assertFalse(view.data_manager.method_calls)


if __name__ == "__main__":
    unittest.main()
