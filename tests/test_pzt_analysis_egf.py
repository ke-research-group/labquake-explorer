"""Parity and scope tests for Tim's in-memory EGF numerical core."""

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from labquake_explorer import analysis
from labquake_explorer.analysis import pzt_analysis_egf as official
from labquake_explorer.analysis.pzt_analysis_seismology import BlockTrace


STUDENT_ROOT = Path(__file__).resolve().parents[2] / "student-tim"


def load_student_module():
    path = str(STUDENT_ROOT)
    sys.path.insert(0, path)
    try:
        return importlib.import_module("pzt_analysis_core.pzt_analysis_EGF")
    finally:
        sys.path.remove(path)


student = load_student_module()


class DatasetAdapter:
    def __init__(self, trace):
        self.trace = trace

    def get_trace(self, sensor, block):
        if (sensor, block) != (self.trace.sensor, self.trace.block):
            raise KeyError((sensor, block))
        return self.trace


def make_trace(count=6000, sample_rate=50000.0):
    time = np.arange(count, dtype=float) / sample_rate + 12.5
    relative = time - time[0]
    voltage = 0.003 + 0.02 * relative
    voltage += 0.002 * np.sin(2 * np.pi * 900 * relative)
    voltage += np.exp(-((relative - 0.06) / 0.0008) ** 2)
    return BlockTrace(
        time=time,
        voltage=voltage,
        sampling_rate=sample_rate,
        trigger_time=12.56,
        trigger_sample=3000,
        block=7,
        sensor="selected",
    )


def make_multi_pulse_trace(trigger_time=12.56):
    sample_rate = 10000.0
    time = np.arange(2001, dtype=float) / sample_rate + 12.5
    voltage = 0.001 * np.sin(2 * np.pi * 700 * (time - time[0]))
    voltage[200] = 0.5
    voltage[600] = 1.0
    return BlockTrace(time, voltage, sample_rate, trigger_time, 600, 9, "explicit")


def write_calibration(directory):
    path = Path(directory) / "egf-calibration.csv"
    frequency = np.logspace(2, 6, 80)
    gain = 2.0 + 0.1 * np.log10(frequency)
    path.write_text(
        "frequency,gain\n"
        + "\n".join(f"{f:.16g},{g:.16g}" for f, g in zip(frequency, gain)),
        encoding="utf-8",
    )
    return path


def assert_arrays_equal(testcase, left, right):
    testcase.assertEqual(left.shape, right.shape)
    testcase.assertEqual(left.dtype, right.dtype)
    np.testing.assert_allclose(left, right, equal_nan=True, rtol=1e-12, atol=1e-14)


def assert_dataclass_values_equal(testcase, left, right):
    testcase.assertEqual(type(left).__name__, type(right).__name__)
    for field in dataclasses.fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if dataclasses.is_dataclass(left_value):
            assert_dataclass_values_equal(testcase, left_value, right_value)
        elif isinstance(left_value, np.ndarray):
            assert_arrays_equal(testcase, left_value, right_value)
        elif isinstance(left_value, float):
            if np.isnan(left_value) and np.isnan(right_value):
                continue
            testcase.assertAlmostEqual(left_value, right_value, places=12)
        else:
            testcase.assertEqual(left_value, right_value)


class DataclassAndExportTests(unittest.TestCase):
    def test_dataclass_fields_order_and_defaults_match_student(self):
        for name in ("EGFTimeWindowResult", "EGFSpectrumResult", "EGFFitResult"):
            with self.subTest(name=name):
                expected = dataclasses.fields(getattr(student, name))
                actual = dataclasses.fields(getattr(official, name))
                self.assertEqual([field.name for field in actual], [field.name for field in expected])
                self.assertEqual([field.default for field in actual], [field.default for field in expected])

    def test_public_analysis_exports_only_future_gui_numerical_entry_points(self):
        expected = (
            "EGFTimeWindowResult",
            "EGFSpectrumResult",
            "EGFFitResult",
            "egf_calib_interp_for",
            "compute_egf_time_window",
            "compute_egf_time_window_at_trigger",
            "compute_egf_spectrum",
            "compute_egf_spectrum_at_trigger",
            "fit_egf_omega_n",
            "calc_mw_from_amp",
        )
        for name in expected:
            self.assertIs(getattr(analysis, name), getattr(official, name))
        for forbidden in ("save_egf_result", "load_saved_egf_bounds", "egf_npz_path"):
            self.assertFalse(hasattr(analysis, forbidden))


class NumericalParityTests(unittest.TestCase):
    def setUp(self):
        official.USE_EXTRA_FILTER = False
        student.USE_EXTRA_FILTER = False

    def test_unchanged_function_bodies_have_ast_parity(self):
        names = ("prezero_pad", "apply_extra_filter", "model_ln_amp", "fit_egf_omega_n", "calc_mw_from_amp")
        for name in names:
            with self.subTest(name=name):
                expected = ast.parse(inspect.getsource(getattr(student, name))).body[0].body
                actual = ast.parse(inspect.getsource(getattr(official, name))).body[0].body
                self.assertEqual(
                    ast.dump(ast.Module(body=actual, type_ignores=[]), include_attributes=False),
                    ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False),
                )

    def test_prezero_padding_and_disabled_filter_match_without_input_mutation(self):
        values = np.array([1.0, 2.0, 3.0])
        original = values.copy()
        assert_arrays_equal(self, official.prezero_pad(values, 7), student.prezero_pad(values, 7))
        self.assertIs(official.apply_extra_filter(values, 0.001), values)
        np.testing.assert_array_equal(values, original)

    def test_calibration_interpolation_matches_with_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_calibration(directory)
            frequency = np.array([0.0, 100.0, 1000.0, 10000.0, 1e6, 2e6])
            expected = student.egf_calib_interp_for(frequency, path)
            actual = official.egf_calib_interp_for(frequency, path)
        assert_arrays_equal(self, actual, expected)

    def test_time_window_matches_dataset_flow_after_explicit_trace_boundary(self):
        trace = make_trace()
        original_time = trace.time.copy()
        original_voltage = trace.voltage.copy()
        expected = student.compute_egf_time_window(
            DatasetAdapter(trace), trace.block, trace.sensor, pre_sec=0.01, post_sec=0.015
        )
        actual = official.compute_egf_time_window(trace, pre_sec=0.01, post_sec=0.015)
        assert_dataclass_values_equal(self, actual, expected)
        assert_arrays_equal(self, actual._noise_windowed, expected._noise_windowed)
        np.testing.assert_array_equal(trace.time, original_time)
        np.testing.assert_array_equal(trace.voltage, original_voltage)

    def test_spectrum_end_to_end_matches_dataset_flow(self):
        trace = make_trace()
        with tempfile.TemporaryDirectory() as directory:
            calibration = write_calibration(directory)
            expected = student.compute_egf_spectrum(
                DatasetAdapter(trace), trace.block, trace.sensor, calibration,
                nfft=8192, pre_sec=0.01, post_sec=0.015,
            )
            actual = official.compute_egf_spectrum(
                trace, calibration, nfft=8192, pre_sec=0.01, post_sec=0.015
            )
        assert_dataclass_values_equal(self, actual, expected)
        assert_dataclass_values_equal(self, actual.time_window, expected.time_window)

    def test_synthetic_fit_model_residual_quality_and_moment_match(self):
        frequency = np.logspace(2, 5, 180)
        params = np.array([2e-5, 8000.0, 2.4])
        amplitude = np.exp(student.model_ln_amp(params, np.log(frequency)))
        spectrum_fields = dict(
            time_window=None,
            freq_hz=np.r_[0.0, frequency],
            amp_raw=np.empty(0),
            amp_noise_raw=np.empty(0),
            amp_cal=np.empty(0),
            amp_noise_cal=np.empty(0),
            valid_cal_mask=np.empty(0, dtype=bool),
            valid_noise_cal_mask=np.empty(0, dtype=bool),
            f7_hz=frequency,
            y7_cal=amplitude,
            y7_noise_cal=np.ones_like(amplitude),
            calibration_path=Path("explicit.csv"),
        )
        expected = student.fit_egf_omega_n(
            student.EGFSpectrumResult(**spectrum_fields), np.log(200), np.log(80000)
        )
        official_input = official.EGFSpectrumResult(**spectrum_fields)
        original = official_input.y7_cal.copy()
        actual = official.fit_egf_omega_n(official_input, np.log(200), np.log(80000))
        assert_dataclass_values_equal(self, actual, expected)
        np.testing.assert_array_equal(official_input.y7_cal, original)
        self.assertAlmostEqual(official.calc_mw_from_amp(actual.omega0), student.calc_mw_from_amp(expected.omega0))

    def test_invalid_fit_points_and_bounds_match_student_errors(self):
        fields = dict(
            time_window=None,
            freq_hz=np.array([0.0, 100.0, 200.0]),
            amp_raw=np.empty(0), amp_noise_raw=np.empty(0), amp_cal=np.empty(0),
            amp_noise_cal=np.empty(0), valid_cal_mask=np.empty(0, dtype=bool),
            valid_noise_cal_mask=np.empty(0, dtype=bool),
            f7_hz=np.array([100.0, 200.0]), y7_cal=np.array([1.0, 2.0]),
            y7_noise_cal=np.array([0.5, 0.5]), calibration_path=Path("x"),
        )
        for module in (student, official):
            spectrum = module.EGFSpectrumResult(**fields)
            with self.assertRaisesRegex(ValueError, "at least 10 points"):
                module.fit_egf_omega_n(spectrum)

        frequency = np.logspace(3, 4, 20)
        fields.update(freq_hz=np.r_[0.0, frequency], f7_hz=frequency, y7_cal=np.ones(20), y7_noise_cal=np.ones(20))
        for module in (student, official):
            spectrum = module.EGFSpectrumResult(**fields)
            with self.assertRaisesRegex(ValueError, "positive and smaller"):
                module.fit_egf_omega_n(spectrum, parameter_lb=(0.0, 1.0, 1.0))

    def test_mismatched_and_nonfinite_fit_inputs_preserve_student_behavior(self):
        base = dict(
            time_window=None,
            freq_hz=np.r_[0.0, np.logspace(3, 4, 20)],
            amp_raw=np.empty(0), amp_noise_raw=np.empty(0), amp_cal=np.empty(0),
            amp_noise_cal=np.empty(0), valid_cal_mask=np.empty(0, dtype=bool),
            valid_noise_cal_mask=np.empty(0, dtype=bool),
            y7_noise_cal=np.ones(20), calibration_path=Path("x"),
        )
        cases = (
            (np.logspace(3, 4, 20), np.ones(19), ValueError),
            (np.full(20, np.nan), np.ones(20), ValueError),
            (np.logspace(3, 4, 20), np.r_[np.zeros(9), -np.ones(11)], None),
        )
        for frequency, amplitude, expected_error in cases:
            outcomes = []
            for module in (student, official):
                spectrum = module.EGFSpectrumResult(
                    **base, f7_hz=frequency, y7_cal=amplitude
                )
                try:
                    outcomes.append(module.fit_egf_omega_n(spectrum))
                except Exception as exc:
                    outcomes.append(exc)
            if expected_error is None:
                self.assertFalse(any(isinstance(outcome, Exception) for outcome in outcomes))
                assert_dataclass_values_equal(self, outcomes[1], outcomes[0])
            else:
                self.assertIsInstance(outcomes[0], expected_error)
                self.assertIs(type(outcomes[0]), type(outcomes[1]))

    def test_nonoverlapping_calibration_range_matches_student_failure(self):
        trace = make_trace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outside.csv"
            path.write_text("frequency,gain\n1e8,2\n2e8,2\n", encoding="utf-8")
            for module, args in (
                (student, (DatasetAdapter(trace), trace.block, trace.sensor, path)),
                (official, (trace, path)),
            ):
                with self.subTest(module=module.__name__), self.assertRaisesRegex(
                    ValueError, "Not enough EGF calibrated spectrum points"
                ):
                    module.compute_egf_spectrum(*args, nfft=8192, pre_sec=0.01, post_sec=0.015)


class CanonicalTriggerTests(unittest.TestCase):
    def test_api_signatures_keep_legacy_threshold_and_exclude_it_from_explicit(self):
        legacy_window = inspect.signature(official.compute_egf_time_window).parameters
        legacy_spectrum = inspect.signature(official.compute_egf_spectrum).parameters
        explicit_window = inspect.signature(
            official.compute_egf_time_window_at_trigger
        ).parameters
        explicit_spectrum = inspect.signature(
            official.compute_egf_spectrum_at_trigger
        ).parameters
        self.assertIn("threshold", legacy_window)
        self.assertIn("threshold", legacy_spectrum)
        self.assertNotIn("threshold", explicit_window)
        self.assertNotIn("threshold", explicit_spectrum)

    def test_legacy_peak_selection_threshold_and_argmax_are_preserved(self):
        trace = make_multi_pulse_trace()
        early = official.compute_egf_time_window(
            trace, pre_sec=0.005, post_sec=0.008, threshold=0.4
        )
        canonical = official.compute_egf_time_window(
            trace, pre_sec=0.005, post_sec=0.008, threshold=0.8
        )
        fallback = official.compute_egf_time_window(
            trace, pre_sec=0.005, post_sec=0.008, threshold=2.0
        )
        self.assertAlmostEqual(early.peak_time, 0.02)
        self.assertAlmostEqual(canonical.peak_time, 0.06)
        self.assertAlmostEqual(fallback.peak_time, 0.06)

    def test_explicit_window_uses_absolute_trigger_without_find_peaks(self):
        trace = make_multi_pulse_trace(trigger_time=12.56)
        with mock.patch.object(official, "find_peaks") as find:
            result = official.compute_egf_time_window_at_trigger(
                trace, pre_sec=0.005, post_sec=0.008
            )
        find.assert_not_called()
        self.assertAlmostEqual(result.peak_time, 0.06)
        expected_indices = np.where(
            (result.full_time >= result.peak_time - 0.005)
            & (result.full_time <= result.peak_time + 0.008)
        )[0]
        self.assertEqual((result.i0, result.i1), (expected_indices[0], expected_indices[-1]))
        expected_length = result.i1 - result.i0 + 1
        self.assertEqual(
            (result.noise_start, result.noise_end),
            (result.i0 - expected_length, result.i0),
        )
        self.assertTrue(hasattr(result, "_noise_windowed"))
        self.assertEqual(result._noise_windowed.shape, result.signal_windowed.shape)

    def test_same_peak_window_fields_are_identical_but_independently_owned(self):
        trace = make_multi_pulse_trace(trigger_time=12.52)
        legacy = official.compute_egf_time_window(
            trace, pre_sec=0.005, post_sec=0.008, threshold=0.4
        )
        explicit = official.compute_egf_time_window_at_trigger(
            trace, pre_sec=0.005, post_sec=0.008
        )
        assert_dataclass_values_equal(self, explicit, legacy)
        assert_arrays_equal(self, explicit._noise_windowed, legacy._noise_windowed)
        for field in dataclasses.fields(explicit):
            if isinstance(getattr(explicit, field.name), np.ndarray):
                self.assertIsNot(getattr(explicit, field.name), getattr(legacy, field.name))

    def test_absolute_trigger_is_converted_to_relative_coordinate(self):
        trace = make_multi_pulse_trace(trigger_time=np.float64(12.56))
        result = official.compute_egf_time_window_at_trigger(
            trace, pre_sec=0.005, post_sec=0.008
        )
        self.assertAlmostEqual(result.peak_time, 0.06)
        self.assertLess(result.peak_time, 1.0)

    def test_trim_adjustment_boundary_and_trimmed_away_trigger(self):
        trace = make_multi_pulse_trace(trigger_time=12.56)
        result = official._prepare_egf_time_window_at_trigger(
            trace, pre_sec=0.005, post_sec=0.008, trim_head_sec=0.01
        )
        self.assertAlmostEqual(result.peak_time, 0.05)

        boundary = make_multi_pulse_trace()
        normalized = boundary.time - boundary.time[0]
        first_retained = np.where(normalized >= 0.01)[0][0]
        boundary.trigger_time = float(boundary.time[first_retained])
        boundary_result = official._prepare_egf_time_window_at_trigger(
            boundary, pre_sec=0.005, post_sec=0.008, trim_head_sec=0.01
        )
        self.assertEqual(boundary_result.peak_time, boundary_result.full_time[0])

        removed = make_multi_pulse_trace(trigger_time=12.505)
        with self.assertRaisesRegex(ValueError, "trimmed-away"):
            official._prepare_egf_time_window_at_trigger(
                removed, pre_sec=0.005, post_sec=0.008, trim_head_sec=0.01
            )

    def test_nearest_sample_tie_uses_first_index(self):
        trace = BlockTrace(
            np.array([10.0, 12.0]), np.array([1.0, 2.0]), 0.5, 11.0, 1, 1, "tie"
        )
        result = official.compute_egf_time_window_at_trigger(trace, pre_sec=2.0, post_sec=2.0)
        self.assertEqual(result.peak_time, 0.0)

    def test_invalid_triggers_do_not_fallback_or_reach_window_body(self):
        base = make_multi_pulse_trace()
        invalid = (
            True,
            np.bool_(False),
            np.array([12.56]),
            "12.56",
            np.nan,
            np.inf,
            12.4,
            12.8,
        )
        for trigger in invalid:
            with self.subTest(trigger=repr(trigger)):
                trace = BlockTrace(
                    base.time, base.voltage, base.sampling_rate, trigger,
                    base.trigger_sample, base.block, base.sensor,
                )
                with mock.patch.object(official, "_finish_egf_time_window") as finish:
                    with self.assertRaises(ValueError):
                        official.compute_egf_time_window_at_trigger(trace)
                    finish.assert_not_called()
                with mock.patch.object(
                    official, "_compute_egf_spectrum_from_window"
                ) as spectrum:
                    with self.assertRaises(ValueError):
                        official.compute_egf_spectrum_at_trigger(
                            trace, Path("unused.csv")
                        )
                    spectrum.assert_not_called()

    def test_explicit_spectrum_wires_only_explicit_window_and_shared_body_once(self):
        trace = make_multi_pulse_trace()
        window = mock.Mock(spec=official.EGFTimeWindowResult)
        result = object()
        calibration = Path("chosen.csv")
        with mock.patch.object(
            official, "_prepare_egf_time_window_at_trigger", return_value=window
        ) as prepare, mock.patch.object(
            official, "_prepare_egf_time_window"
        ) as legacy, mock.patch.object(
            official, "_compute_egf_spectrum_from_window", return_value=result
        ) as spectrum:
            actual = official.compute_egf_spectrum_at_trigger(
                trace, calibration, nfft=4096, pre_sec=0.02, post_sec=0.03
            )
        self.assertIs(actual, result)
        prepare.assert_called_once_with(trace, pre_sec=0.02, post_sec=0.03)
        legacy.assert_not_called()
        spectrum.assert_called_once_with(window, calibration, 4096)

    def test_legacy_spectrum_wires_only_auto_window_and_shared_body_once(self):
        trace = make_multi_pulse_trace()
        window = mock.Mock(spec=official.EGFTimeWindowResult)
        result = object()
        with mock.patch.object(
            official, "_prepare_egf_time_window", return_value=window
        ) as prepare, mock.patch.object(
            official, "_prepare_egf_time_window_at_trigger"
        ) as explicit, mock.patch.object(
            official, "_compute_egf_spectrum_from_window", return_value=result
        ) as spectrum:
            actual = official.compute_egf_spectrum(
                trace, Path("chosen.csv"), nfft=2048,
                pre_sec=0.01, post_sec=0.02, threshold=0.7,
            )
        self.assertIs(actual, result)
        prepare.assert_called_once_with(
            trace, pre_sec=0.01, post_sec=0.02, threshold=0.7
        )
        explicit.assert_not_called()
        spectrum.assert_called_once_with(window, Path("chosen.csv"), 2048)

    def test_same_window_spectra_are_numerically_identical(self):
        trace = make_multi_pulse_trace(trigger_time=12.52)
        with tempfile.TemporaryDirectory() as directory:
            calibration = write_calibration(directory)
            legacy = official.compute_egf_spectrum(
                trace, calibration, nfft=4096,
                pre_sec=0.005, post_sec=0.008, threshold=0.4,
            )
            explicit = official.compute_egf_spectrum_at_trigger(
                trace, calibration, nfft=4096, pre_sec=0.005, post_sec=0.008
            )
        assert_dataclass_values_equal(self, explicit, legacy)

    def test_multi_pulse_spectra_keep_distinct_auto_and_canonical_windows(self):
        trace = make_multi_pulse_trace(trigger_time=12.56)
        with tempfile.TemporaryDirectory() as directory:
            calibration = write_calibration(directory)
            legacy = official.compute_egf_spectrum(
                trace, calibration, nfft=4096,
                pre_sec=0.005, post_sec=0.008, threshold=0.4,
            )
            explicit = official.compute_egf_spectrum_at_trigger(
                trace, calibration, nfft=4096, pre_sec=0.005, post_sec=0.008
            )
        self.assertAlmostEqual(legacy.time_window.peak_time, 0.02)
        self.assertAlmostEqual(explicit.time_window.peak_time, 0.06)
        self.assertNotEqual(legacy.time_window.i0, explicit.time_window.i0)

    def test_explicit_processing_does_not_modify_trace_or_calibration(self):
        trace = make_multi_pulse_trace()
        original_time = trace.time.copy()
        original_voltage = trace.voltage.copy()
        with tempfile.TemporaryDirectory() as directory:
            calibration = write_calibration(directory)
            original_csv = calibration.read_bytes()
            official.compute_egf_spectrum_at_trigger(
                trace, calibration, nfft=4096, pre_sec=0.005, post_sec=0.008
            )
            self.assertEqual(calibration.read_bytes(), original_csv)
        np.testing.assert_array_equal(trace.time, original_time)
        np.testing.assert_array_equal(trace.voltage, original_voltage)


class ScopeTests(unittest.TestCase):
    def test_module_has_no_forbidden_infrastructure_or_inference(self):
        source = inspect.getsource(official)
        forbidden = (
            "tkinter", "DataManager", "h5py", "TPC5Dataset", "TDEvent", "AS01",
            "SENSOR_TO_A_LABEL", "np.savez", "json.dump", "monkeypatch",
            "EGF_RESULTS_DIR", "EGF_BOUNDS_STATE_PATH", "output_dir",
        )
        for item in forbidden:
            with self.subTest(item=item):
                self.assertNotIn(item, source)

    def test_latest_tim_module_has_no_ratio_or_source_time_function_api(self):
        names = {name for name, value in vars(student).items() if callable(value)}
        self.assertFalse(any("ratio" in name.lower() for name in names))
        self.assertFalse(any("source_time" in name.lower() or "deconvol" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
