"""Headless-safe tests for the explicit PZT time-domain preview."""

import copy
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.analysis.pzt_analysis_seismology import BlockTrace, TimeWindowResult
from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import PZTTimeDomainView as ExportedView
from labquake_explorer.ui.views.pzt_time_domain_view import (
    PZTTimeDomainView,
    parse_time_domain_parameters,
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
    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


def make_event(time=None, raw=None, event_time=10.0):
    time = np.array([9.0, 10.0, 11.0]) if time is None else time
    raw = np.array([[1.0, 2.0, 1.0], [0.0, 3.0, 0.0]]) if raw is None else raw
    return {
        "event_time": event_time,
        "strain": {"original": {"time": time, "raw": raw}},
    }


def make_view(event=None, event_idx=0):
    event = make_event() if event is None else event
    view = PZTTimeDomainView.__new__(PZTTimeDomainView)
    view.parent = mock.Mock()
    view.run_idx = 3
    view.event_idx = event_idx
    view.events = [event]
    view.event = event
    (
        view.strain_time,
        view.strain_raw,
        view.channel_candidates,
        view.context_error,
    ) = view._resolve_strain_context(event)
    view.selected_channel_index = None
    view.preview_result = None
    view.preview_trace = None
    view.preview_parameters = None
    view._initializing_parameters = False
    view.channel_combobox = FakeWidget()
    view.preview_button = FakeWidget()
    view.event_combobox = FakeWidget(str(event_idx))
    view.pre_sec_var = FakeVariable("0.1")
    view.post_sec_var = FakeVariable("0.1")
    view.status_var = FakeVariable()
    view.figure = Figure()
    view.ax = view.figure.add_subplot(111)
    view.canvas = FakeCanvas()
    return view


def fake_result(trace):
    return TimeWindowResult(
        x_ms=np.array([0.0, 1000.0]),
        voltage_windowed=np.array([2.0, 1.0]),
        displacement_windowed=np.array([2.0, 1.0]),
        window_scaled_voltage=np.array([1.0, 0.5]),
        window_scaled_displacement=np.array([1.0, 0.5]),
        window=np.array([1.0, 0.5]),
        raw_voltage_segment=np.array([2.0, 1.0]),
        displacement_segment=np.array([2.0, 1.0]),
        full_time=np.array([0.0, 1.0, 2.0]),
        full_voltage_corrected=np.array([1.0, 2.0, 1.0]),
        i0=1,
        i1=2,
        noise_start=0,
        noise_end=1,
        dt=1.0,
        coherent_gain=0.75,
        peak_time=1.0,
        block=trace.block,
        sensor=trace.sensor,
    )


class ParameterTests(unittest.TestCase):
    def test_parameters_preserve_values(self):
        self.assertEqual(
            parse_time_domain_parameters("0.2", "0.3"),
            {"pre_sec": 0.2, "post_sec": 0.3},
        )

    def test_parameters_require_finite_nonnegative_values(self):
        for values in (("x", "1"), ("nan", "1"), ("-1", "1"), ("1", "-1")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_time_domain_parameters(*values)


class CanonicalContextTests(unittest.TestCase):
    def test_context_preserves_event_array_identity_and_channel_first_shape(self):
        event = make_event()
        time, raw, channels, error = PZTTimeDomainView._resolve_strain_context(event)
        self.assertIs(time, event["strain"]["original"]["time"])
        self.assertIs(raw, event["strain"]["original"]["raw"])
        self.assertEqual(channels, [0, 1])
        self.assertIsNone(error)

    def test_one_dimensional_raw_is_one_explicit_channel(self):
        event = make_event(raw=np.array([1.0, 2.0, 3.0]))
        _, raw, channels, error = PZTTimeDomainView._resolve_strain_context(event)
        self.assertEqual(raw.ndim, 1)
        self.assertEqual(channels, [0])
        self.assertIsNone(error)

    def test_missing_or_invalid_strain_is_unavailable_without_fallback(self):
        invalid = (
            {},
            {"strain": {}},
            make_event(raw=np.ones((1, 1, 3))),
            make_event(raw=np.ones((2, 2))),
            make_event(raw=np.array([[1.0, np.nan, 2.0]])),
        )
        for event in invalid:
            with self.subTest(event=event):
                time, raw, channels, error = PZTTimeDomainView._resolve_strain_context(event)
                self.assertIsNone(time)
                self.assertIsNone(raw)
                self.assertEqual(channels, [])
                self.assertTrue(error)

    def test_bool_complex_and_nonfinite_time_are_rejected(self):
        for time, raw in (
            (np.array([True, False, True]), np.ones(3)),
            (np.array([0.0, 1.0, np.inf]), np.ones(3)),
            (np.arange(3.0), np.ones(3, dtype=complex)),
            (np.arange(3.0), np.array([True, False, True])),
        ):
            with self.subTest(time=time, raw=raw):
                self.assertTrue(PZTTimeDomainView._resolve_strain_context(make_event(time, raw))[3])

    def test_build_trace_uses_absolute_event_time_and_nearest_sample(self):
        time = np.array([9.8, 9.95, 10.2])
        raw = np.vstack((np.arange(3.0), np.arange(3.0) + 4))
        view = make_view(make_event(time, raw, event_time=np.float64(10.0)), event_idx=7)
        view.selected_channel_index = 1
        trace = view._build_selected_trace()
        self.assertIs(trace.time, time)
        self.assertTrue(np.shares_memory(trace.voltage, raw))
        self.assertEqual(trace.trigger_time, 10.0)
        self.assertEqual(trace.trigger_sample, 1)
        self.assertAlmostEqual(trace.sampling_rate, 1 / 0.2)
        self.assertEqual(trace.block, 7)
        self.assertEqual(trace.sensor, "strain[1]")

    def test_build_trace_rejects_invalid_event_time_and_nonincreasing_time(self):
        for event in (make_event(event_time=np.nan), make_event(time=np.array([0.0, 0.0, 1.0]))):
            view = make_view(event)
            view.selected_channel_index = 0
            with self.subTest(event=event), self.assertRaises(ValueError):
                view._build_selected_trace()


class BindingAndSwitchingTests(unittest.TestCase):
    def test_refresh_does_not_auto_select_and_disables_preview(self):
        view = make_view()
        view._refresh_event_widgets()
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual(view.channel_combobox.get(), "")
        self.assertEqual(view.channel_combobox.options["values"], ["strain[0]", "strain[1]"])
        self.assertEqual(view.preview_button.options["state"], "disabled")

    def test_explicit_channel_selection_and_stale_selection(self):
        view = make_view()
        view.channel_combobox.set("strain[1]")
        view.preview_result = object()
        view.on_channel_changed()
        self.assertEqual(view.selected_channel_index, 1)
        self.assertEqual(view.preview_button.options["state"], "normal")
        self.assertIsNone(view.preview_result)
        view.channel_combobox.set("AS01")
        view.on_channel_changed()
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual(view.preview_button.options["state"], "disabled")

    def test_event_switch_uses_zero_based_position_and_preserves_controls(self):
        first = make_event(event_time=10.0)
        second = make_event(event_time=20.0)
        view = make_view(first)
        view.events = [first, second]
        view.data_manager = mock.Mock()
        view.data_manager.get_data.side_effect = [[first, second], second]
        view.event_combobox.set("1")
        view.pre_sec_var.set("0.8")
        view.selected_channel_index = 1
        view.on_event_changed()
        self.assertEqual(view.event_idx, 1)
        self.assertIs(view.event, second)
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual(view.pre_sec_var.get(), "0.8")
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(
            view.data_manager.get_data.call_args_list,
            [mock.call("runs/[3]/events"), mock.call("runs/[3]/events/[1]")],
        )

    def test_parameter_change_invalidates_without_analysis(self):
        view = make_view()
        view.preview_result = object()
        with mock.patch("labquake_explorer.ui.views.pzt_time_domain_view.compute_time_window_at_trigger") as compute:
            view.on_parameters_changed()
        self.assertIsNone(view.preview_result)
        compute.assert_not_called()


class AnalysisAndPlotTests(unittest.TestCase):
    def test_real_preview_window_stays_at_canonical_event_with_larger_other_pulse(self):
        time = np.linspace(12.0, 13.0, 1001)
        raw = np.zeros(time.shape)
        raw[200] = 2.0
        raw[700] = 0.5
        event = make_event(time=time, raw=raw, event_time=12.7)
        view = make_view(event)
        view.selected_channel_index = 0
        view.pre_sec_var.set("0.01")
        view.post_sec_var.set("0.02")
        view.recompute_preview()
        self.assertAlmostEqual(view.preview_result.peak_time, 0.7)
        self.assertEqual((view.preview_result.i0, view.preview_result.i1), (690, 719))
        self.assertEqual(view.ax.lines[-1].get_xdata()[0], 0.0)
        view.parent.data_manager.set_data.assert_not_called()

    @mock.patch("labquake_explorer.ui.views.pzt_time_domain_view.compute_time_window_at_trigger")
    def test_preview_calls_analysis_once_and_preserves_input(self, compute):
        event = make_event()
        original = copy.deepcopy(event)
        view = make_view(event)
        view.selected_channel_index = 1
        trace_holder = {}

        def calculate(trace, **kwargs):
            trace_holder["trace"] = trace
            return fake_result(trace)

        compute.side_effect = calculate
        view.recompute_preview()
        compute.assert_called_once()
        self.assertEqual(compute.call_args.kwargs, {"pre_sec": 0.1, "post_sec": 0.1})
        self.assertEqual(trace_holder["trace"].trigger_time, event["event_time"])
        self.assertIs(view.preview_trace, trace_holder["trace"])
        self.assertIsInstance(view.preview_result, TimeWindowResult)
        np.testing.assert_array_equal(event["strain"]["original"]["raw"], original["strain"]["original"]["raw"])
        self.assertEqual(len(view.ax.lines), 4)
        np.testing.assert_array_equal(view.ax.lines[0].get_ydata(), trace_holder["trace"].voltage)
        np.testing.assert_array_equal(view.ax.lines[1].get_ydata(), view.preview_result.voltage_windowed)
        np.testing.assert_array_equal(view.ax.lines[2].get_ydata(), view.preview_result.window_scaled_voltage)
        self.assertEqual(view.ax.lines[3].get_xdata()[0], 0.0)
        view.parent.data_manager.set_data.assert_not_called()

    def test_plot_uses_result_windows_without_recomputing_analysis(self):
        view = make_view()
        view.selected_channel_index = 0
        trace = view._build_selected_trace()
        result = fake_result(trace)
        with mock.patch("labquake_explorer.ui.views.pzt_time_domain_view.compute_time_window_at_trigger") as compute:
            view._draw_preview(trace, result)
        compute.assert_not_called()
        self.assertEqual(len(view.ax.patches), 2)
        self.assertIn("Signal window", [text.get_text() for text in view.ax.get_legend().get_texts()])
        self.assertIn("Noise window", [text.get_text() for text in view.ax.get_legend().get_texts()])

    @mock.patch("labquake_explorer.ui.views.pzt_time_domain_view.messagebox.showerror")
    def test_analysis_error_clears_preview_and_reports_status(self, showerror):
        view = make_view()
        view.selected_channel_index = 0
        view.pre_sec_var.set("bad")
        view.preview_result = object()
        view.recompute_preview()
        self.assertIsNone(view.preview_result)
        self.assertIn("pre_sec", view.status_var.get())
        showerror.assert_called_once()


class WiringAndLifecycleTests(unittest.TestCase):
    def test_view_is_exported_and_controller_method_exists(self):
        self.assertIs(ExportedView, PZTTimeDomainView)
        self.assertTrue(callable(LabquakeExplorer.preview_pzt_time_domain))

    @mock.patch("labquake_explorer.ui.labquake_explorer.PZTTimeDomainView")
    def test_controller_uses_shared_parser_without_analysis_or_persistence(self, view_class):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", "[7]"))
        explorer._extract_run_event_indices = mock.Mock(return_value=(3, 7))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        view = view_class.return_value
        explorer.preview_pzt_time_domain()
        explorer._extract_run_event_indices.assert_called_once_with(r"runs\[3]\events\[7]")
        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])

    def test_context_menu_has_pzt_entry(self):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.root = object()

        class FakeMenu:
            def __init__(self, *args, **kwargs):
                self.commands = []

            def add_command(self, **kwargs):
                self.commands.append(kwargs)

        with mock.patch("labquake_explorer.ui.labquake_explorer.tk.Menu", FakeMenu):
            explorer.create_context_menus()
        labels = [item["label"] for item in explorer.event_menu.commands]
        self.assertIn("Preview PZT Time Domain", labels)

    def test_close_removes_view_without_persistence(self):
        view = PZTTimeDomainView.__new__(PZTTimeDomainView)
        view.parent = mock.Mock()
        view.parent.child_windows = [view]
        view.destroy = mock.Mock()
        view.on_close()
        self.assertNotIn(view, view.parent.child_windows)
        view.destroy.assert_called_once()
        view.parent.data_manager.set_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
