"""Headless-safe tests for the explicit canonical-trigger PZT spectrum view."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.figure import Figure
from pandas.errors import EmptyDataError, ParserError

from labquake_explorer.analysis.pzt_analysis_seismology import (
    BlockTrace,
    SpectrumResult,
    TimeWindowResult,
)
from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import PZTSpectrumView as ExportedView
from labquake_explorer.ui.views.pzt_spectrum_view import (
    _EXPECTED_PREVIEW_EXCEPTIONS,
    PZTSpectrumView,
    parse_spectrum_parameters,
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


def make_event(time=None, raw=None, event_time=12.7):
    time = np.linspace(12.0, 13.0, 1001) if time is None else time
    if raw is None:
        relative = time - time[0]
        raw = np.vstack(
            (
                0.002 * np.sin(2 * np.pi * 127 * relative),
                0.003 * np.cos(2 * np.pi * 89 * relative),
            )
        )
    return {"event_time": event_time, "strain": {"original": {"time": time, "raw": raw}}}


def make_view(event=None, event_idx=0):
    event = make_event() if event is None else event
    view = PZTSpectrumView.__new__(PZTSpectrumView)
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
    view.preview_trace = None
    view.preview_result = None
    view.preview_parameters = None
    view._initializing_parameters = False
    view.event_combobox = FakeWidget(str(event_idx))
    view.channel_combobox = FakeWidget()
    view.preview_button = FakeWidget()
    view.pre_sec_var = FakeVariable("0.1")
    view.post_sec_var = FakeVariable("0.1")
    view.nfft_var = FakeVariable("10000")
    view.calibration_path_var = FakeVariable("")
    view.q_path_var = FakeVariable("")
    view.status_var = FakeVariable()
    view.figure = Figure()
    view.time_ax, view.raw_ax, view.resampled_ax = view.figure.subplots(3, 1)
    view.canvas = FakeCanvas()
    return view


def make_result(trace):
    count = len(trace.time)
    window = TimeWindowResult(
        x_ms=np.arange(3.0),
        voltage_windowed=np.array([0.1, 1.0, 0.1]),
        displacement_windowed=np.array([0.1, 1.0, 0.1]),
        window_scaled_voltage=np.array([0.1, 1.0, 0.1]),
        window_scaled_displacement=np.array([0.1, 1.0, 0.1]),
        window=np.array([0.1, 1.0, 0.1]),
        raw_voltage_segment=np.array([0.1, 1.0, 0.1]),
        displacement_segment=np.array([0.1, 1.0, 0.1]),
        full_time=trace.time - trace.time[0],
        full_voltage_corrected=np.asarray(trace.voltage, dtype=float).copy(),
        i0=699,
        i1=701,
        noise_start=696,
        noise_end=699,
        dt=1.0 / trace.sampling_rate,
        coherent_gain=0.4,
        peak_time=trace.trigger_time - trace.time[0],
        block=trace.block,
        sensor=trace.sensor,
    )
    frequency = np.array([0.0, 100.0, 200.0, 300.0])
    signal = np.array([0.0, 1.0, np.nan, 3.0])
    noise = np.array([0.0, 0.5, 0.2, np.inf])
    return SpectrumResult(
        time_window=window,
        freq_hz=frequency,
        amp_raw=signal,
        amp_noise_raw=noise,
        amp_voltage_raw=signal.copy(),
        amp_noise_voltage_raw=noise.copy(),
        amp_cal=np.array([np.nan, 2.0, 0.0, 4.0]),
        amp_noise_cal=np.array([np.nan, 1.0, 0.5, np.nan]),
        valid_cal_mask=np.array([False, True, False, True]),
        valid_noise_cal_mask=np.array([False, True, True, False]),
        f7_hz=np.array([100.0, 200.0, 300.0]),
        y7_cal=np.array([2.0, 3.0, 4.0]),
        y7_noise_cal=np.array([1.0, np.nan, 2.0]),
        y7_qcorr=np.array([2.1, 3.1, 4.1]),
        calibration_path=Path("cal.csv"),
        q_path=Path("q.csv"),
    )


class ParameterTests(unittest.TestCase):
    def test_parser_returns_typed_defaults_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "cal.data"
            q_path = Path(directory) / "q.any"
            calibration.write_text("x", encoding="utf-8")
            q_path.write_text("y", encoding="utf-8")
            result = parse_spectrum_parameters("0.1", "0.2", "10000", str(calibration), str(q_path))
        self.assertEqual(result["pre_sec"], 0.1)
        self.assertEqual(result["post_sec"], 0.2)
        self.assertEqual(result["nfft"], 10000)
        self.assertIsInstance(result["calibration_csv"], Path)
        self.assertIsInstance(result["q_csv"], Path)

    def test_parser_rejects_invalid_numeric_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("x", encoding="utf-8")
            cases = (("nan", "0.1", "10"), ("-1", "0.1", "10"), ("0.1", "-1", "10"), ("0.1", "0.1", "10000.0"), ("0.1", "0.1", "0"))
            for pre, post, nfft in cases:
                with self.subTest(values=(pre, post, nfft)), self.assertRaises(ValueError):
                    parse_spectrum_parameters(pre, post, nfft, str(path), str(path))

    def test_parser_rejects_empty_missing_and_directory_paths_by_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good"
            good.write_text("x", encoding="utf-8")
            cases = (("", str(good), "calibration_csv"), (str(root / "missing"), str(good), "calibration_csv"), (str(root), str(good), "calibration_csv"), (str(good), "", "q_csv"))
            for calibration, q_path, field in cases:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    parse_spectrum_parameters("0.1", "0.1", "10", calibration, q_path)


class ContextAndBindingTests(unittest.TestCase):
    def test_context_preserves_array_identity_and_channel_first_contract(self):
        event = make_event()
        time, raw, channels, error = PZTSpectrumView._resolve_strain_context(event)
        self.assertIs(time, event["strain"]["original"]["time"])
        self.assertIs(raw, event["strain"]["original"]["raw"])
        self.assertEqual(channels, [0, 1])
        self.assertIsNone(error)

    def test_one_dimensional_raw_is_one_channel(self):
        event = make_event(raw=np.ones(1001))
        self.assertEqual(PZTSpectrumView._resolve_strain_context(event)[2], [0])

    def test_invalid_context_is_rejected_without_mechanical_fallback(self):
        invalid = (
            {"time": np.arange(3.0), "displacement": np.ones(3)},
            make_event(raw=np.ones((1, 1, 1001))),
            make_event(raw=np.ones((2, 5))),
            make_event(raw=np.ones(1001, dtype=bool)),
            make_event(raw=np.ones(1001, dtype=complex)),
            make_event(raw=np.r_[np.ones(1000), np.nan]),
            make_event(time=np.r_[np.arange(1000.0), 999.0], raw=np.ones(1001)),
        )
        for event in invalid:
            with self.subTest(event=event):
                time, raw, channels, error = PZTSpectrumView._resolve_strain_context(event)
                self.assertIsNone(time)
                self.assertIsNone(raw)
                self.assertEqual(channels, [])
                self.assertTrue(error)

    def test_refresh_starts_blank_and_button_requires_channel_and_paths(self):
        view = make_view()
        view._refresh_event_widgets()
        self.assertEqual(view.channel_combobox.get(), "")
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual(view.preview_button.options["state"], "disabled")
        view.channel_combobox.set("strain[1]")
        view.on_channel_changed()
        self.assertEqual(view.selected_channel_index, 1)
        self.assertEqual(view.preview_button.options["state"], "disabled")
        view.calibration_path_var.set("cal.csv")
        view.q_path_var.set("q.csv")
        view._update_preview_button()
        self.assertEqual(view.preview_button.options["state"], "normal")

    def test_stale_or_inferred_name_clears_binding_and_preview(self):
        view = make_view()
        view.selected_channel_index = 0
        view.preview_result = object()
        view.channel_combobox.set("AS01")
        view.on_channel_changed()
        self.assertIsNone(view.selected_channel_index)
        self.assertIsNone(view.preview_result)

    def test_block_trace_uses_absolute_event_time_identity_and_channel_view(self):
        event = make_event(event_time=np.float64(12.7))
        original = copy.deepcopy(event)
        view = make_view(event, event_idx=7)
        view.selected_channel_index = 1
        trace = view._build_selected_trace()
        self.assertIs(trace.time, event["strain"]["original"]["time"])
        self.assertTrue(np.shares_memory(trace.voltage, event["strain"]["original"]["raw"]))
        self.assertEqual(trace.trigger_time, 12.7)
        self.assertEqual(trace.trigger_sample, 700)
        self.assertAlmostEqual(trace.sampling_rate, 1000.0)
        self.assertEqual((trace.block, trace.sensor), (7, "strain[1]"))
        np.testing.assert_array_equal(event["strain"]["original"]["raw"], original["strain"]["original"]["raw"])


class BrowseAndSwitchTests(unittest.TestCase):
    @mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.filedialog.askopenfilename")
    def test_browse_updates_only_selected_path_and_invalidates(self, ask):
        view = make_view()
        view.preview_result = object()
        ask.return_value = "C:/chosen/cal.csv"
        view.browse_calibration()
        self.assertEqual(view.calibration_path_var.get(), "C:/chosen/cal.csv")
        self.assertEqual(view.q_path_var.get(), "")
        self.assertIsNone(view.preview_result)

    @mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.filedialog.askopenfilename")
    def test_browse_cancel_preserves_value_without_analysis(self, ask):
        view = make_view()
        view.q_path_var.set("old.csv")
        ask.return_value = ""
        with mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger") as compute:
            view.browse_q()
        self.assertEqual(view.q_path_var.get(), "old.csv")
        compute.assert_not_called()

    def test_event_switch_uses_zero_based_position_preserves_controls_and_paths(self):
        first, second = make_event(event_time=12.7), make_event(event_time=22.7)
        view = make_view(first)
        view.data_manager = mock.Mock()
        view.data_manager.get_data.side_effect = [[first, second], second]
        view.event_combobox.set("1")
        view.pre_sec_var.set("0.3")
        view.nfft_var.set("2048")
        view.calibration_path_var.set("cal.csv")
        view.q_path_var.set("q.csv")
        view.selected_channel_index = 1
        view.on_event_changed()
        self.assertEqual(view.event_idx, 1)
        self.assertIs(view.event, second)
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual((view.pre_sec_var.get(), view.nfft_var.get()), ("0.3", "2048"))
        self.assertEqual((view.calibration_path_var.get(), view.q_path_var.get()), ("cal.csv", "q.csv"))
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(view.data_manager.get_data.call_args_list, [mock.call("runs/[3]/events"), mock.call("runs/[3]/events/[1]")])


class AnalysisAndPlotTests(unittest.TestCase):
    def _configure_existing_paths(self, view, directory):
        calibration = Path(directory) / "cal.csv"
        q_path = Path(directory) / "q.csv"
        calibration.write_text("x", encoding="utf-8")
        q_path.write_text("y", encoding="utf-8")
        view.calibration_path_var.set(str(calibration))
        view.q_path_var.set(str(q_path))

    @mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger")
    def test_preview_calls_only_explicit_analysis_once(self, compute):
        view = make_view()
        view.selected_channel_index = 1
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "cal.csv"
            q_path = Path(directory) / "q.csv"
            calibration.write_text("x", encoding="utf-8")
            q_path.write_text("y", encoding="utf-8")
            view.calibration_path_var.set(str(calibration))
            view.q_path_var.set(str(q_path))
            holder = {}

            def calculate(trace, **kwargs):
                holder["trace"] = trace
                return make_result(trace)

            compute.side_effect = calculate
            view.recompute_preview()
        compute.assert_called_once()
        self.assertEqual(compute.call_args.kwargs["nfft"], 10000)
        self.assertEqual(compute.call_args.kwargs["pre_sec"], 0.1)
        self.assertEqual(compute.call_args.kwargs["post_sec"], 0.1)
        self.assertEqual(holder["trace"].trigger_time, view.event["event_time"])
        self.assertIs(view.preview_trace, holder["trace"])
        self.assertIsInstance(view.preview_result, SpectrumResult)
        self.assertIn("calibrated", view.status_var.get())
        view.parent.data_manager.set_data.assert_not_called()

    def test_plot_uses_result_fields_and_safe_masks_without_mutation(self):
        view = make_view()
        view.selected_channel_index = 0
        trace = view._build_selected_trace()
        result = make_result(trace)
        original = copy.deepcopy(result)
        view._draw_preview(trace, result)
        self.assertEqual(len(view.time_ax.lines), 3)
        self.assertEqual(view.time_ax.lines[-1].get_xdata()[0], 0.0)
        self.assertGreaterEqual(len(view.raw_ax.lines), 3)
        self.assertEqual(len(view.resampled_ax.lines), 3)
        np.testing.assert_array_equal(result.amp_raw, original.amp_raw)
        np.testing.assert_array_equal(result.y7_noise_cal, original.y7_noise_cal)
        labels = [line.get_label() for line in view.resampled_ax.lines]
        self.assertNotIn("fit", " ".join(labels).lower())

    def test_plot_positive_skips_empty_curve(self):
        figure = Figure()
        axes = figure.add_subplot(111)
        self.assertFalse(PZTSpectrumView._plot_positive(axes, [0, 1], [1, np.nan], label="empty"))
        self.assertEqual(len(axes.lines), 0)

    def test_parameter_change_invalidates_without_analysis_or_dialog(self):
        view = make_view()
        view.preview_result = object()
        with mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger") as compute, mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.messagebox.showerror") as showerror:
            view.on_parameters_changed()
        self.assertIsNone(view.preview_result)
        compute.assert_not_called()
        showerror.assert_not_called()

    @mock.patch("labquake_explorer.ui.views.pzt_spectrum_view.messagebox.showerror")
    def test_preview_error_clears_state_and_reports(self, showerror):
        view = make_view()
        view.selected_channel_index = 0
        view.recompute_preview()
        self.assertIsNone(view.preview_result)
        self.assertIn("calibration_csv", view.status_var.get())
        showerror.assert_called_once()

    def test_expected_analysis_and_file_errors_are_presented(self):
        expected_errors = (
            ValueError("invalid spectrum"),
            FileNotFoundError("missing calibration"),
            OSError("cannot read Q file"),
            ParserError("malformed calibration CSV"),
            EmptyDataError("empty calibration CSV"),
        )
        for error in expected_errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                view = make_view()
                view.selected_channel_index = 0
                view.preview_result = object()
                self._configure_existing_paths(view, directory)
                with mock.patch(
                    "labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger",
                    side_effect=error,
                ), mock.patch(
                    "labquake_explorer.ui.views.pzt_spectrum_view.messagebox.showerror"
                ) as showerror:
                    view.recompute_preview()
                self.assertIsNone(view.preview_result)
                self.assertEqual(view.status_var.get(), str(error))
                showerror.assert_called_once_with("PZT spectrum", str(error), parent=view)

    def test_programming_errors_are_not_caught(self):
        for error in (AttributeError("missing implementation attribute"), RuntimeError("internal failure")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                view = make_view()
                view.selected_channel_index = 0
                self._configure_existing_paths(view, directory)
                with mock.patch(
                    "labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger",
                    side_effect=error,
                ), mock.patch(
                    "labquake_explorer.ui.views.pzt_spectrum_view.messagebox.showerror"
                ) as showerror, self.assertRaises(type(error)):
                    view.recompute_preview()
                showerror.assert_not_called()

    def test_unexpected_plotting_error_propagates_without_user_error_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            view = make_view()
            view.selected_channel_index = 0
            self._configure_existing_paths(view, directory)
            result = make_result(view._build_selected_trace())
            with mock.patch(
                "labquake_explorer.ui.views.pzt_spectrum_view.compute_spectrum_at_trigger",
                return_value=result,
            ), mock.patch.object(
                view, "_draw_preview", side_effect=AttributeError("plot contract bug")
            ), mock.patch(
                "labquake_explorer.ui.views.pzt_spectrum_view.messagebox.showerror"
            ) as showerror, self.assertRaisesRegex(AttributeError, "plot contract bug"):
                view.recompute_preview()
            showerror.assert_not_called()

    def test_expected_exception_tuple_excludes_programming_errors(self):
        self.assertNotIn(AttributeError, _EXPECTED_PREVIEW_EXCEPTIONS)
        self.assertNotIn(AssertionError, _EXPECTED_PREVIEW_EXCEPTIONS)
        self.assertNotIn(RuntimeError, _EXPECTED_PREVIEW_EXCEPTIONS)
        self.assertNotIn(Exception, _EXPECTED_PREVIEW_EXCEPTIONS)

    def test_real_multi_pulse_preview_stays_at_canonical_trigger(self):
        time = np.linspace(12.0, 13.0, 1001)
        relative = time - time[0]
        raw = 0.002 * np.sin(2 * np.pi * 127 * relative)
        raw[200] += 2.0
        raw[700] += 0.5
        event = make_event(time=time, raw=raw, event_time=12.7)
        view = make_view(event)
        view.selected_channel_index = 0
        view.pre_sec_var.set("0.01")
        view.post_sec_var.set("0.02")
        view.nfft_var.set("2048")
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "cal.csv"
            q_path = Path(directory) / "q.csv"
            calibration.write_text("frequency,gain\n100,2\n500,2\n", encoding="utf-8")
            q_path.write_text("1,0.01\n100,0.01\n", encoding="utf-8")
            view.calibration_path_var.set(str(calibration))
            view.q_path_var.set(str(q_path))
            view.recompute_preview()
        self.assertAlmostEqual(view.preview_result.time_window.peak_time, 0.7)
        self.assertEqual((view.preview_result.time_window.i0, view.preview_result.time_window.i1), (690, 719))
        self.assertEqual(view.time_ax.lines[-1].get_xdata()[0], 0.0)


class WiringAndLifecycleTests(unittest.TestCase):
    def test_view_is_exported_and_controller_method_exists(self):
        self.assertIs(ExportedView, PZTSpectrumView)
        self.assertTrue(callable(LabquakeExplorer.preview_pzt_spectrum))

    @mock.patch("labquake_explorer.ui.labquake_explorer.PZTSpectrumView")
    def test_controller_uses_shared_parser_without_analysis_csv_or_persistence(self, view_class):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", "[7]"))
        explorer._extract_run_event_indices = mock.Mock(return_value=(3, 7))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        view = view_class.return_value
        explorer.preview_pzt_spectrum()
        explorer._extract_run_event_indices.assert_called_once_with(r"runs\[3]\events\[7]")
        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])

    def test_context_menu_has_spectrum_entry(self):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.root = object()

        class FakeMenu:
            def __init__(self, *args, **kwargs):
                self.commands = []

            def add_command(self, **kwargs):
                self.commands.append(kwargs)

        with mock.patch("labquake_explorer.ui.labquake_explorer.tk.Menu", FakeMenu):
            explorer.create_context_menus()
        self.assertIn("Preview PZT Spectrum", [item["label"] for item in explorer.event_menu.commands])

    def test_close_removes_view_without_persistence(self):
        view = PZTSpectrumView.__new__(PZTSpectrumView)
        view.parent = mock.Mock()
        view.parent.child_windows = [view]
        view.destroy = mock.Mock()
        view.on_close()
        self.assertNotIn(view, view.parent.child_windows)
        view.destroy.assert_called_once()
        view.parent.data_manager.set_data.assert_not_called()

    def test_source_has_no_forbidden_tim_or_persistence_dependencies(self):
        source = Path(PZTSpectrumView.__module__.replace(".", "/") + ".py")
        text = (Path(__file__).parents[1] / source).read_text(encoding="utf-8")
        for forbidden in ("TPC5Dataset", "TDEvent", "AS02", "set_data(", "compute_spectrum(", "fit_omega_n_q"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
