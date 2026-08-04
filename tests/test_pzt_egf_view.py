"""Headless-safe tests for the explicit canonical-trigger PZT EGF view."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.analysis.pzt_analysis_egf import (
    EGF_DEFAULT_FIT_PARAMETER_LB,
    EGF_DEFAULT_FIT_PARAMETER_UB,
    EGF_DEFAULT_LNF_MAX,
    EGF_DEFAULT_LNF_MIN,
    EGFFitResult,
    EGFSpectrumResult,
    EGFTimeWindowResult,
)
from labquake_explorer.analysis.pzt_analysis_seismology import (
    DEFAULT_NFFT,
    DEFAULT_POST_SEC,
    DEFAULT_PRE_SEC,
)
from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import PZTEGFView as ExportedView
from labquake_explorer.ui.views.pzt_egf_view import (
    _EXPECTED_FIT_EXCEPTIONS,
    _EXPECTED_SPECTRUM_EXCEPTIONS,
    PZTEGFView,
    parse_egf_fit_parameters,
    parse_egf_spectrum_parameters,
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


def make_event(time=None, raw=None, event_time=12.56):
    time = np.linspace(12.5, 12.7, 2001) if time is None else time
    if raw is None:
        relative = time - time[0]
        raw = np.vstack(
            (np.sin(2 * np.pi * 700 * relative), np.cos(2 * np.pi * 900 * relative))
        )
    return {"event_time": event_time, "strain": {"original": {"time": time, "raw": raw}}}


def make_view(event=None, event_idx=0):
    event = make_event() if event is None else event
    view = PZTEGFView.__new__(PZTEGFView)
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
    view.fit_result = None
    view.fit_parameters = None
    view._initializing_parameters = False
    view.event_combobox = FakeWidget(str(event_idx))
    view.channel_combobox = FakeWidget()
    view.preview_button = FakeWidget()
    view.fit_button = FakeWidget()
    view.pre_sec_var = FakeVariable(str(DEFAULT_PRE_SEC))
    view.post_sec_var = FakeVariable(str(DEFAULT_POST_SEC))
    view.nfft_var = FakeVariable(str(DEFAULT_NFFT))
    view.calibration_path_var = FakeVariable("")
    view.lnf_min_var = FakeVariable(str(EGF_DEFAULT_LNF_MIN))
    view.lnf_max_var = FakeVariable(str(EGF_DEFAULT_LNF_MAX))
    view.omega0_lb_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_LB[0]))
    view.fc_lb_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_LB[1]))
    view.n_lb_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_LB[2]))
    view.omega0_ub_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_UB[0]))
    view.fc_ub_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_UB[1]))
    view.n_ub_var = FakeVariable(str(EGF_DEFAULT_FIT_PARAMETER_UB[2]))
    view.status_var = FakeVariable()
    view.fit_status_var = FakeVariable("Calculate an EGF spectrum before fitting")
    view.fit_value_vars = {
        name: FakeVariable("—")
        for name in ("Omega0", "fc", "n", "R²", "M0", "Mw", "ln(f)")
    }
    view.figure = Figure()
    view.time_ax, view.raw_ax, view.resampled_ax = view.figure.subplots(3, 1)
    view.canvas = FakeCanvas()
    return view


def make_result(trace):
    time = trace.time - trace.time[0]
    i0, i1 = 550, 680
    window = EGFTimeWindowResult(
        x_ms=(time[i0 : i1 + 1] - time[i0]) * 1e3,
        signal_windowed=np.asarray(trace.voltage[i0 : i1 + 1], dtype=float).copy(),
        window_scaled_signal=np.ones(i1 - i0 + 1),
        window=np.ones(i1 - i0 + 1),
        raw_voltage_segment=np.asarray(trace.voltage[i0 : i1 + 1], dtype=float).copy(),
        full_time=time.copy(),
        full_voltage_corrected=np.asarray(trace.voltage, dtype=float).copy(),
        i0=i0,
        i1=i1,
        noise_start=419,
        noise_end=550,
        dt=1.0 / trace.sampling_rate,
        coherent_gain=0.4,
        peak_time=0.06,
        block=trace.block,
        sensor=trace.sensor,
    )
    frequency = np.array([0.0, 100.0, 200.0, 300.0])
    return EGFSpectrumResult(
        time_window=window,
        freq_hz=frequency,
        amp_raw=np.array([0.0, 1.0, np.nan, 3.0]),
        amp_noise_raw=np.array([0.0, 0.5, 0.2, np.inf]),
        amp_cal=np.array([np.nan, 2.0, 0.0, 4.0]),
        amp_noise_cal=np.array([np.nan, 1.0, 0.5, np.nan]),
        valid_cal_mask=np.array([False, True, False, True]),
        valid_noise_cal_mask=np.array([False, True, True, False]),
        f7_hz=np.array([100.0, 200.0, 300.0]),
        y7_cal=np.array([2.0, 3.0, 4.0]),
        y7_noise_cal=np.array([1.0, np.nan, 2.0]),
        calibration_path=Path("cal.csv"),
    )


def make_fit_result():
    return EGFFitResult(
        6.9, 11.3, np.array([True, False, True]), -1.25, 4567.0, 2.4,
        np.nan, np.array([100.0, 300.0]), np.array([2.0, 1.0]), 5785.0, -3.56,
    )


class ParserTests(unittest.TestCase):
    def test_spectrum_parser_returns_typed_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.csv"
            path.write_text("x", encoding="utf-8")
            result = parse_egf_spectrum_parameters(
                str(DEFAULT_PRE_SEC), str(DEFAULT_POST_SEC), str(DEFAULT_NFFT), str(path)
            )
        self.assertEqual(result["pre_sec"], DEFAULT_PRE_SEC)
        self.assertEqual(result["post_sec"], DEFAULT_POST_SEC)
        self.assertEqual(result["nfft"], DEFAULT_NFFT)
        self.assertIsInstance(result["calibration_csv"], Path)

    def test_spectrum_parser_rejects_invalid_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "cal.csv"
            good.write_text("x", encoding="utf-8")
            cases = (
                ("nan", "0.1", "10", str(good), "pre_sec"),
                ("0.1", "-1", "10", str(good), "post_sec"),
                ("0.1", "0.1", "10.0", str(good), "nfft"),
                ("0.1", "0.1", "10", "", "calibration_csv"),
                ("0.1", "0.1", "10", str(Path(directory) / "missing"), "calibration_csv"),
                ("0.1", "0.1", "10", directory, "calibration_csv"),
            )
            for pre, post, nfft, path, field in cases:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    parse_egf_spectrum_parameters(pre, post, nfft, path)

    def test_fit_parser_defaults_and_validation(self):
        values = (
            str(EGF_DEFAULT_LNF_MIN), str(EGF_DEFAULT_LNF_MAX),
            *(str(value) for value in EGF_DEFAULT_FIT_PARAMETER_LB),
            *(str(value) for value in EGF_DEFAULT_FIT_PARAMETER_UB),
        )
        result = parse_egf_fit_parameters(*values)
        self.assertEqual(result["parameter_lb"], EGF_DEFAULT_FIT_PARAMETER_LB)
        self.assertEqual(result["parameter_ub"], EGF_DEFAULT_FIT_PARAMETER_UB)
        cases = ((0, "nan", "ln"), (1, values[0], "minimum"), (2, "0", "Omega0"), (3, "4e5", "fc"))
        for index, value, field in cases:
            invalid = list(values)
            invalid[index] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                parse_egf_fit_parameters(*invalid)


class ContextAndBindingTests(unittest.TestCase):
    def test_context_preserves_identity_and_channel_first_shape(self):
        event = make_event()
        time, raw, channels, error = PZTEGFView._resolve_strain_context(event)
        self.assertIs(time, event["strain"]["original"]["time"])
        self.assertIs(raw, event["strain"]["original"]["raw"])
        self.assertEqual(channels, [0, 1])
        self.assertIsNone(error)
        self.assertEqual(PZTEGFView._resolve_strain_context(make_event(raw=np.ones(2001)))[2], [0])

    def test_invalid_context_has_no_mechanical_fallback(self):
        invalid = (
            {"event_time": 1.0, "time": np.arange(3.0), "displacement": np.ones(3)},
            make_event(raw=np.ones((1, 1, 2001))),
            make_event(raw=np.ones(2001, dtype=bool)),
            make_event(raw=np.ones(2001, dtype=complex)),
            make_event(raw=np.r_[np.ones(2000), np.nan]),
            make_event(raw=np.ones((2, 5))),
        )
        for event in invalid:
            self.assertEqual(PZTEGFView._resolve_strain_context(event)[2], [])

    def test_refresh_is_blank_and_button_requires_explicit_channel_and_path(self):
        view = make_view()
        view._refresh_event_widgets()
        self.assertEqual(view.channel_combobox.get(), "")
        self.assertIsNone(view.selected_channel_index)
        self.assertEqual(view.preview_button.options["state"], "disabled")
        view.channel_combobox.set("strain[1]")
        view.on_channel_changed()
        view.calibration_path_var.set("chosen.csv")
        view._update_preview_button()
        self.assertEqual(view.selected_channel_index, 1)
        self.assertEqual(view.preview_button.options["state"], "normal")
        view.channel_combobox.set("AS01")
        view.on_channel_changed()
        self.assertIsNone(view.selected_channel_index)

    def test_block_trace_uses_absolute_event_time_and_channel_view(self):
        event = make_event(event_time=np.float64(12.56))
        original = copy.deepcopy(event)
        view = make_view(event, event_idx=7)
        view.selected_channel_index = 1
        trace = view._build_selected_trace()
        self.assertIs(trace.time, event["strain"]["original"]["time"])
        self.assertTrue(np.shares_memory(trace.voltage, event["strain"]["original"]["raw"]))
        self.assertEqual(trace.trigger_time, 12.56)
        self.assertEqual(trace.trigger_sample, 600)
        self.assertAlmostEqual(trace.sampling_rate, 10000.0)
        self.assertEqual((trace.block, trace.sensor), (7, "strain[1]"))
        np.testing.assert_array_equal(event["strain"]["original"]["raw"], original["strain"]["original"]["raw"])


class AnalysisFitAndPlotTests(unittest.TestCase):
    def _configure_path(self, view, directory):
        path = Path(directory) / "cal.csv"
        path.write_text("x", encoding="utf-8")
        view.calibration_path_var.set(str(path))
        return path

    def _ready_view(self):
        view = make_view()
        view.selected_channel_index = 0
        view.preview_trace = view._build_selected_trace()
        view.preview_result = make_result(view.preview_trace)
        view._update_fit_button()
        return view

    def test_spectrum_calls_only_explicit_analysis_once_and_enables_fit(self):
        view = make_view()
        view.selected_channel_index = 0
        with tempfile.TemporaryDirectory() as directory:
            path = self._configure_path(view, directory)
            trace = view._build_selected_trace()
            result = make_result(trace)
            with mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.compute_egf_spectrum_at_trigger",
                return_value=result,
            ) as compute, mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.fit_egf_omega_n"
            ) as fit:
                view.recompute_preview()
        compute.assert_called_once_with(
            mock.ANY, calibration_csv=path, nfft=DEFAULT_NFFT,
            pre_sec=DEFAULT_PRE_SEC, post_sec=DEFAULT_POST_SEC,
        )
        self.assertIs(view.preview_result, result)
        self.assertEqual(view.fit_button.options["state"], "normal")
        fit.assert_not_called()

    def test_preview_draws_only_returned_arrays_with_positive_safety(self):
        view = self._ready_view()
        result = view.preview_result
        original = copy.deepcopy(result)
        view._draw_preview(view.preview_trace, result)
        self.assertIn("Corrected waveform", [line.get_label() for line in view.time_ax.lines])
        self.assertIn("Signal calibrated", [line.get_label() for line in view.raw_ax.lines])
        self.assertIn("Calibrated", [line.get_label() for line in view.resampled_ax.lines])
        self.assertNotIn("Omega-n model", [line.get_label() for line in view.resampled_ax.lines])
        np.testing.assert_array_equal(result.y7_cal, original.y7_cal)
        self.assertFalse(PZTEGFView._plot_positive(view.raw_ax, [0, np.nan], [1, 2]))

    def test_fit_uses_existing_result_once_without_recomputing_or_deriving(self):
        view = self._ready_view()
        spectrum = view.preview_result
        fit_result = make_fit_result()
        original = spectrum.y7_cal.copy()
        with mock.patch(
            "labquake_explorer.ui.views.pzt_egf_view.fit_egf_omega_n",
            return_value=fit_result,
        ) as fit, mock.patch(
            "labquake_explorer.ui.views.pzt_egf_view.compute_egf_spectrum_at_trigger"
        ) as compute:
            view.recompute_fit()
        fit.assert_called_once_with(
            spectrum, lnf_min=EGF_DEFAULT_LNF_MIN, lnf_max=EGF_DEFAULT_LNF_MAX,
            parameter_lb=EGF_DEFAULT_FIT_PARAMETER_LB,
            parameter_ub=EGF_DEFAULT_FIT_PARAMETER_UB,
        )
        compute.assert_not_called()
        self.assertIs(view.fit_result, fit_result)
        np.testing.assert_array_equal(spectrum.y7_cal, original)
        self.assertEqual(view.fit_value_vars["Omega0"].get(), "-1.25")
        self.assertEqual(view.fit_value_vars["M0"].get(), "5785")
        self.assertEqual(view.fit_value_vars["Mw"].get(), "-3.56")
        self.assertEqual(view.fit_value_vars["R²"].get(), "nan")

    def test_fit_overlay_uses_mask_and_returned_model_without_accumulation(self):
        view = self._ready_view()
        result = make_fit_result()
        parameters = {"lnf_min": 6.9, "lnf_max": 11.3}
        for _ in range(2):
            view._draw_preview(view.preview_trace, view.preview_result, redraw=False)
            view._draw_fit_overlay(view.preview_result, result, parameters)
        labels = [line.get_label() for line in view.resampled_ax.lines]
        self.assertEqual(labels.count("Fit samples"), 1)
        self.assertEqual(labels.count("Omega-n model"), 1)
        model = next(line for line in view.resampled_ax.lines if line.get_label() == "Omega-n model")
        np.testing.assert_array_equal(model.get_xdata(), result.f_model_hz)
        np.testing.assert_array_equal(model.get_ydata(), result.m_model_amp)
        self.assertIn("Calibrated", labels)

    def test_fit_only_and_spectrum_invalidation_are_separate(self):
        view = self._ready_view()
        spectrum = view.preview_result
        view.fit_result = make_fit_result()
        with mock.patch(
            "labquake_explorer.ui.views.pzt_egf_view.compute_egf_spectrum_at_trigger"
        ) as compute:
            view.on_fit_parameters_changed()
        self.assertIs(view.preview_result, spectrum)
        self.assertIsNone(view.fit_result)
        self.assertEqual(view.fit_button.options["state"], "normal")
        compute.assert_not_called()
        view.on_spectrum_parameters_changed()
        self.assertIsNone(view.preview_result)
        self.assertEqual(view.fit_button.options["state"], "disabled")

    def test_expected_and_unexpected_error_boundaries(self):
        view = self._ready_view()
        for error in (ValueError("bad fit"), RuntimeError("solver failed")):
            with mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.fit_egf_omega_n", side_effect=error
            ), mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.messagebox.showerror"
            ) as show:
                view.recompute_fit()
                show.assert_called_once_with("PZT EGF fit", str(error), parent=view)
                self.assertIsNotNone(view.preview_result)
        for error in (AttributeError("bug"), AssertionError("bug")):
            with mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.fit_egf_omega_n", side_effect=error
            ), mock.patch(
                "labquake_explorer.ui.views.pzt_egf_view.messagebox.showerror"
            ) as show, self.assertRaises(type(error)):
                view.recompute_fit()
            show.assert_not_called()

    def test_spectrum_runtime_and_plotting_errors_propagate(self):
        view = make_view()
        view.selected_channel_index = 0
        with tempfile.TemporaryDirectory() as directory:
            self._configure_path(view, directory)
            for error in (RuntimeError("internal"), AttributeError("plot bug")):
                target = "compute_egf_spectrum_at_trigger" if isinstance(error, RuntimeError) else None
                patches = (
                    mock.patch("labquake_explorer.ui.views.pzt_egf_view.compute_egf_spectrum_at_trigger", side_effect=error)
                    if target else mock.patch.object(view, "_draw_preview", side_effect=error)
                )
                result_patch = mock.patch(
                    "labquake_explorer.ui.views.pzt_egf_view.compute_egf_spectrum_at_trigger",
                    return_value=make_result(view._build_selected_trace()),
                )
                with patches, (mock.patch("builtins.id") if target else result_patch), mock.patch(
                    "labquake_explorer.ui.views.pzt_egf_view.messagebox.showerror"
                ) as show, self.assertRaises(type(error)):
                    view.recompute_preview()
                show.assert_not_called()

    def test_exception_tuples_are_narrow(self):
        self.assertNotIn(RuntimeError, _EXPECTED_SPECTRUM_EXCEPTIONS)
        self.assertIn(RuntimeError, _EXPECTED_FIT_EXCEPTIONS)
        for error in (AttributeError, AssertionError, Exception):
            self.assertNotIn(error, _EXPECTED_FIT_EXCEPTIONS)


class BrowseSwitchWiringAndSafetyTests(unittest.TestCase):
    @mock.patch("labquake_explorer.ui.views.pzt_egf_view.filedialog.askopenfilename")
    def test_browse_cancel_and_selection_lifecycle(self, ask):
        view = make_view()
        view.calibration_path_var.set("old.csv")
        ask.return_value = ""
        view.browse_calibration()
        self.assertEqual(view.calibration_path_var.get(), "old.csv")
        ask.return_value = "new.csv"
        view.preview_result = object()
        view.browse_calibration()
        self.assertEqual(view.calibration_path_var.get(), "new.csv")
        self.assertIsNone(view.preview_result)

    def test_event_switch_preserves_controls_paths_and_fit_bounds(self):
        first, second = make_event(), make_event(event_time=12.66)
        view = make_view(first)
        view.data_manager = mock.Mock()
        view.data_manager.get_data.side_effect = [[first, second], second]
        view.event_combobox.set("1")
        view.pre_sec_var.set("0.3")
        view.calibration_path_var.set("cal.csv")
        view.lnf_min_var.set("8.0")
        view.selected_channel_index = 1
        view.preview_result = object()
        view.fit_result = object()
        view.on_event_changed()
        self.assertIs(view.event, second)
        self.assertIsNone(view.selected_channel_index)
        self.assertIsNone(view.preview_result)
        self.assertIsNone(view.fit_result)
        self.assertEqual((view.pre_sec_var.get(), view.calibration_path_var.get(), view.lnf_min_var.get()), ("0.3", "cal.csv", "8.0"))
        self.assertEqual(view.preview_button.options["state"], "disabled")
        self.assertEqual(view.fit_button.options["state"], "disabled")

    def test_view_export_controller_menu_and_close(self):
        self.assertIs(ExportedView, PZTEGFView)
        self.assertTrue(callable(LabquakeExplorer.preview_pzt_egf))
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", "[7]"))
        explorer._extract_run_event_indices = mock.Mock(return_value=(3, 7))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        with mock.patch("labquake_explorer.ui.labquake_explorer.PZTEGFView") as view_class:
            explorer.preview_pzt_egf()
        view = view_class.return_value
        explorer._extract_run_event_indices.assert_called_once_with(r"runs\[3]\events\[7]")
        view_class.assert_called_once_with(explorer, 3, 7)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])

        explorer.root = object()
        class FakeMenu:
            def __init__(self, *args, **kwargs): self.commands = []
            def add_command(self, **kwargs): self.commands.append(kwargs)
        with mock.patch("labquake_explorer.ui.labquake_explorer.tk.Menu", FakeMenu):
            explorer.create_context_menus()
        self.assertIn("Preview PZT EGF", [item["label"] for item in explorer.event_menu.commands])

        close_view = PZTEGFView.__new__(PZTEGFView)
        close_view.parent = mock.Mock()
        close_view.parent.child_windows = [close_view]
        close_view.destroy = mock.Mock()
        close_view.on_close()
        close_view.parent.data_manager.set_data.assert_not_called()

    def test_source_has_no_forbidden_analysis_or_persistence(self):
        source = (Path(__file__).parents[1] / "labquake_explorer/ui/views/pzt_egf_view.py").read_text(encoding="utf-8")
        forbidden = (
            "TPC5Dataset", "TDEvent", "AS01", "set_data(", "np.savez", "json.dump",
            "compute_egf_spectrum(", "find_peaks", "model_ln_amp", "calc_mw_from_amp",
            "spectral_ratio", "deconvol", "source_time", "except Exception",
        )
        for item in forbidden:
            with self.subTest(item=item):
                self.assertNotIn(item, source)


if __name__ == "__main__":
    unittest.main()
