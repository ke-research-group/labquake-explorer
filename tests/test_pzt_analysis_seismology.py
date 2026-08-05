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

from labquake_explorer.analysis import pzt_analysis_seismology as official


STUDENT_ROOT = Path(__file__).resolve().parents[2] / "student-tim"


def load_student_module():
    path = str(STUDENT_ROOT)
    sys.path.insert(0, path)
    try:
        return importlib.import_module("pzt_analysis_core.pzt_analysis_seismology")
    finally:
        sys.path.remove(path)


student = load_student_module()


def make_trace(module, count=4000, sample_rate=20000.0):
    time = np.arange(count, dtype=float) / sample_rate + 12.5
    relative = time - time[0]
    voltage = 0.02 * relative + 0.005 * np.sin(2 * np.pi * 700 * relative)
    voltage += np.exp(-((relative - 0.1) / 0.002) ** 2)
    return module.BlockTrace(
        time=time,
        voltage=voltage,
        sampling_rate=sample_rate,
        trigger_time=12.6,
        trigger_sample=2000,
        block=7,
        sensor="selected",
    )


def assert_arrays_equal(testcase, left, right):
    testcase.assertEqual(left.shape, right.shape)
    testcase.assertEqual(left.dtype, right.dtype)
    np.testing.assert_allclose(left, right, equal_nan=True, rtol=1e-12, atol=1e-14)


class DataclassContractTests(unittest.TestCase):
    def test_dataclass_fields_match_student_source(self):
        for name in ("BlockTrace", "TimeWindowResult", "SpectrumResult", "FitResult"):
            with self.subTest(name=name):
                expected = dataclasses.fields(getattr(student, name))
                actual = dataclasses.fields(getattr(official, name))
                self.assertEqual([field.name for field in actual], [field.name for field in expected])
                self.assertEqual(
                    [field.default for field in actual],
                    [field.default for field in expected],
                )

    def test_block_trace_preserves_input_identity(self):
        time = np.arange(5.0)
        voltage = np.arange(5.0) * 2
        trace = official.BlockTrace(time, voltage, 1.0, 0.0, 0, 2, "explicit")
        self.assertIs(trace.time, time)
        self.assertIs(trace.voltage, voltage)


class WindowParityTests(unittest.TestCase):
    def test_window_helpers_match_student_source(self):
        for length in (0, 2, 17):
            with self.subTest(helper="general_cosine", length=length):
                coeffs = [0.4, -0.5, 0.1]
                assert_arrays_equal(
                    self,
                    official.general_cosine(length, coeffs),
                    student.general_cosine(length, coeffs),
                )
        for terms in (3, 4, 7):
            with self.subTest(helper="blackman_harris", terms=terms):
                assert_arrays_equal(
                    self,
                    official.blackman_harris_window(31, terms),
                    student.blackman_harris_window(31, terms),
                )
        for alpha in (0.0, 0.1, 0.5, 1.0):
            with self.subTest(helper="tukey", alpha=alpha):
                assert_arrays_equal(
                    self,
                    official.tukey_window(31, alpha),
                    student.tukey_window(31, alpha),
                )

    def test_postzero_pad_matches_and_does_not_modify_input(self):
        values = np.array([1.0, 2.0, 3.0])
        original = values.copy()
        assert_arrays_equal(
            self,
            official.postzero_pad(values, 6),
            student.postzero_pad(values, 6),
        )
        np.testing.assert_array_equal(values, original)

    def test_time_window_matches_trace_based_student_helper(self):
        source_trace = make_trace(student)
        official_trace = official.BlockTrace(**vars(source_trace))
        original_time = official_trace.time.copy()
        original_voltage = official_trace.voltage.copy()
        expected = student._prepare_time_window(source_trace, pre_sec=0.03, post_sec=0.04)
        actual = official.compute_time_window(official_trace, pre_sec=0.03, post_sec=0.04)
        for name in (
            "x_ms", "voltage_windowed", "displacement_windowed", "window",
            "raw_voltage_segment", "full_time", "full_voltage_corrected",
        ):
            assert_arrays_equal(self, getattr(actual, name), getattr(expected, name))
        for name in ("i0", "i1", "noise_start", "noise_end", "dt", "peak_time", "block", "sensor"):
            self.assertEqual(getattr(actual, name), getattr(expected, name))
        np.testing.assert_array_equal(official_trace.time, original_time)
        np.testing.assert_array_equal(official_trace.voltage, original_voltage)

    def test_auto_peak_contract_remains_first_threshold_peak(self):
        time = np.linspace(12.0, 13.0, 1001)
        voltage = np.zeros(time.shape)
        voltage[200] = 2.0
        voltage[700] = 0.5
        trace = official.BlockTrace(time, voltage, 1000.0, 12.7, 700, 1, "selected")
        result = official.compute_time_window(trace, pre_sec=0.01, post_sec=0.01, threshold=0.1)
        self.assertAlmostEqual(result.peak_time, 0.2)
        self.assertLess(result.i0, 200)
        self.assertLess(result.i1, 700)

    def test_explicit_trigger_uses_canonical_pulse_not_larger_pulse(self):
        time = np.linspace(12.0, 13.0, 1001)
        voltage = np.zeros(time.shape)
        voltage[200] = 2.0
        voltage[700] = 0.5
        trace = official.BlockTrace(time, voltage, 1000.0, 12.7, 700, 1, "selected")
        result = official.compute_time_window_at_trigger(trace, pre_sec=0.01, post_sec=0.02)
        self.assertAlmostEqual(result.peak_time, 0.7)
        self.assertEqual((result.i0, result.i1), (690, 719))
        self.assertEqual((result.noise_start, result.noise_end), (660, 690))
        self.assertLessEqual(time[result.i0] - trace.trigger_time, 0.0)
        self.assertGreaterEqual(time[result.i1] - trace.trigger_time, 0.0)

    def test_explicit_trigger_matches_latest_student_explicit_peak_flow(self):
        source = (STUDENT_ROOT / "labquake_explorer_pzt_td.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_time_window_with_peak"
        )
        namespace = {
            "np": np,
            "_pzt_seis": official,
            "DEFAULT_PRE_SEC": official.DEFAULT_PRE_SEC,
            "DEFAULT_POST_SEC": official.DEFAULT_POST_SEC,
            "DEFAULT_THRESHOLD": official.DEFAULT_THRESHOLD,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "student-explicit-peak", "exec"), namespace)
        student_explicit = namespace["_prepare_time_window_with_peak"]

        trace = make_trace(official)
        peak_time_rel = trace.trigger_time - trace.time[0]
        expected = student_explicit(trace, peak_time_rel, pre_sec=0.03, post_sec=0.04)
        actual = official.compute_time_window_at_trigger(trace, pre_sec=0.03, post_sec=0.04)
        for field in dataclasses.fields(official.TimeWindowResult):
            expected_value = getattr(expected, field.name)
            actual_value = getattr(actual, field.name)
            if isinstance(expected_value, np.ndarray):
                assert_arrays_equal(self, actual_value, expected_value)
            else:
                self.assertEqual(actual_value, expected_value)
        assert_arrays_equal(self, actual._noise_displacement, expected._noise_displacement)
        assert_arrays_equal(self, actual._noise_voltage, expected._noise_voltage)

    def test_explicit_trigger_rejects_invalid_or_out_of_range_time_without_fallback(self):
        trace = make_trace(official)
        for trigger in (True, np.array([trace.trigger_time]), np.nan, trace.time[0] - 1.0, trace.time[-1] + 1.0):
            invalid = official.BlockTrace(**{**vars(trace), "trigger_time": trigger})
            with self.subTest(trigger=trigger), self.assertRaisesRegex(ValueError, "trigger_time"):
                official.compute_time_window_at_trigger(invalid)


class FilteringParityTests(unittest.TestCase):
    def tearDown(self):
        official.USE_EXTRA_FILTER = False
        student.USE_EXTRA_FILTER = False

    def test_disabled_filter_returns_original_object_like_student(self):
        values = np.linspace(-1.0, 1.0, 100)
        self.assertIs(official.apply_extra_filter(values, 0.001), values)
        self.assertIs(student.apply_extra_filter(values, 0.001), values)

    def test_enabled_filter_matches_student_and_preserves_input(self):
        values = np.sin(np.linspace(0, 40 * np.pi, 1000))
        original = values.copy()
        for module in (official, student):
            module.USE_EXTRA_FILTER = True
            module.FILTER_MODE = "bandpass"
            module.FILTER_ORDER = 3
            module.FILTER_HP_HZ = 10.0
            module.FILTER_LP_HZ = 200.0
        expected = student.apply_extra_filter(values, 0.001)
        actual = official.apply_extra_filter(values, 0.001)
        assert_arrays_equal(self, actual, expected)
        np.testing.assert_array_equal(values, original)


class CalibrationAndQParityTests(unittest.TestCase):
    def setUp(self):
        official.load_csv_gain.cache_clear()
        student.load_csv_gain.cache_clear()
        official.load_q_data.cache_clear()
        student.load_q_data.cache_clear()

    def tearDown(self):
        official.USE_Q_ATTEN_CORR = True
        student.USE_Q_ATTEN_CORR = True

    def test_linear_and_db_calibration_match_student(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            linear = root / "linear.csv"
            linear.write_text("frequency,gain\n100,2\n1000,4\n10000,8\n", encoding="utf-8")
            db = root / "db.csv"
            db.write_text("frequency,response_db\n100,-60\n1000,-30\n10000,0\n", encoding="utf-8")
            for path in (linear, db):
                with self.subTest(path=path.name):
                    expected = student.load_csv_gain(str(path))
                    actual = official.load_csv_gain(str(path))
                    assert_arrays_equal(self, actual[0], expected[0])
                    assert_arrays_equal(self, actual[1], expected[1])
                    self.assertEqual(actual[2], expected[2])

    def test_calibration_interpolation_and_bounds_match_student(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gain.csv"
            path.write_text("frequency,gain\n100,2\n1000,4\n10000,8\n", encoding="utf-8")
            frequency = np.array([50.0, 100.0, 550.0, 10000.0, 20000.0])
            assert_arrays_equal(
                self,
                official.calib_interp_for(frequency, path),
                student.calib_interp_for(frequency, path),
            )

    def test_q_correction_factor_and_frequency_bounds_match_student(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.csv"
            path.write_text("1,0.01\n100,0.02\n450,0.03\n", encoding="utf-8")
            frequency = np.array([500.0, 1000.0, 100000.0, 450000.0, 500000.0])
            amplitude = np.ones(frequency.shape)
            expected = student.apply_q_correction(frequency, amplitude, path)
            actual = official.apply_q_correction(frequency, amplitude, path)
            assert_arrays_equal(self, actual, expected)
            self.assertEqual(actual[0], 1.0)
            self.assertEqual(actual[-1], 1.0)
            np.testing.assert_array_equal(amplitude, np.ones(frequency.shape))

    def test_disabled_q_returns_equal_copy(self):
        official.USE_Q_ATTEN_CORR = False
        student.USE_Q_ATTEN_CORR = False
        frequency = np.array([1000.0, 2000.0])
        amplitude = np.array([2.0, 3.0])
        actual = official.apply_q_correction(frequency, amplitude, Path("unused.csv"))
        expected = student.apply_q_correction(frequency, amplitude, Path("unused.csv"))
        assert_arrays_equal(self, actual, expected)
        self.assertIsNot(actual, amplitude)


class SpectrumAndFitParityTests(unittest.TestCase):
    def setUp(self):
        official.load_csv_gain.cache_clear()
        student.load_csv_gain.cache_clear()
        official.load_q_data.cache_clear()
        student.load_q_data.cache_clear()

    def test_spectrum_matches_student_dataset_wrapper(self):
        class Dataset:
            def __init__(self, trace):
                self.trace = trace

            def get_trace(self, sensor, block):
                return self.trace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "gain.csv"
            calibration.write_text("frequency,gain\n100,2\n1000,2\n10000,2\n", encoding="utf-8")
            q_path = root / "q.csv"
            q_path.write_text("1,0.01\n10,0.01\n100,0.01\n", encoding="utf-8")
            student_trace = make_trace(student)
            official_trace = official.BlockTrace(**vars(student_trace))
            expected = student.compute_spectrum(
                Dataset(student_trace), 7, "selected", calibration, q_path,
                nfft=8192, pre_sec=0.03, post_sec=0.04,
            )
            actual = official.compute_spectrum(
                official_trace, calibration, q_path,
                nfft=8192, pre_sec=0.03, post_sec=0.04,
            )
            for name in (
                "freq_hz", "amp_raw", "amp_noise_raw", "amp_voltage_raw",
                "amp_noise_voltage_raw", "amp_cal", "amp_noise_cal",
                "valid_cal_mask", "valid_noise_cal_mask", "f7_hz", "y7_cal",
                "y7_noise_cal", "y7_qcorr",
            ):
                assert_arrays_equal(self, getattr(actual, name), getattr(expected, name))

    @mock.patch.object(official, "compute_time_window_at_trigger")
    @mock.patch.object(official, "compute_time_window")
    @mock.patch.object(official, "_compute_spectrum_from_window")
    def test_original_spectrum_keeps_auto_peak_and_threshold_contract(
        self, spectrum_from_window, auto_window, trigger_window
    ):
        trace = make_trace(official)
        auto_window.return_value = mock.sentinel.window
        spectrum_from_window.return_value = mock.sentinel.spectrum
        result = official.compute_spectrum(
            trace, Path("cal.csv"), Path("q.csv"),
            nfft=321, pre_sec=0.02, post_sec=0.04, threshold=0.17,
        )
        self.assertIs(result, mock.sentinel.spectrum)
        auto_window.assert_called_once_with(
            trace, pre_sec=0.02, post_sec=0.04, threshold=0.17
        )
        trigger_window.assert_not_called()
        spectrum_from_window.assert_called_once_with(
            mock.sentinel.window, Path("cal.csv"), Path("q.csv"), 321
        )
        self.assertIn("threshold", inspect.signature(official.compute_spectrum).parameters)

    @mock.patch.object(official, "compute_time_window")
    @mock.patch.object(official, "compute_time_window_at_trigger")
    @mock.patch.object(official, "_compute_spectrum_from_window")
    def test_trigger_spectrum_calls_only_explicit_window_once(
        self, spectrum_from_window, trigger_window, auto_window
    ):
        trace = make_trace(official)
        trigger_window.return_value = mock.sentinel.window
        spectrum_from_window.return_value = mock.sentinel.spectrum
        result = official.compute_spectrum_at_trigger(
            trace, Path("cal.csv"), Path("q.csv"),
            nfft=654, pre_sec=0.03, post_sec=0.05,
        )
        self.assertIs(result, mock.sentinel.spectrum)
        trigger_window.assert_called_once_with(trace, pre_sec=0.03, post_sec=0.05)
        auto_window.assert_not_called()
        spectrum_from_window.assert_called_once_with(
            mock.sentinel.window, Path("cal.csv"), Path("q.csv"), 654
        )
        self.assertNotIn("threshold", inspect.signature(official.compute_spectrum_at_trigger).parameters)

    @mock.patch.object(official, "_compute_spectrum_from_window")
    def test_trigger_spectrum_rejects_invalid_trigger_before_partial_result(self, spectrum_from_window):
        trace = make_trace(official)
        invalid = official.BlockTrace(**{**vars(trace), "trigger_time": np.nan})
        with self.assertRaisesRegex(ValueError, "trigger_time"):
            official.compute_spectrum_at_trigger(
                invalid, Path("cal.csv"), Path("q.csv")
            )
        spectrum_from_window.assert_not_called()

    def test_multi_pulse_spectra_use_distinct_auto_and_trigger_windows(self):
        time = np.linspace(12.0, 13.0, 1001)
        relative = time - time[0]
        voltage = 0.002 * np.sin(2 * np.pi * 127 * relative)
        voltage[200] += 2.0
        voltage[700] += 0.5
        trace = official.BlockTrace(time, voltage, 1000.0, 12.7, 700, 1, "selected")
        original_time = time.copy()
        original_voltage = voltage.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "gain.csv"
            calibration.write_text("frequency,gain\n100,2\n500,2\n", encoding="utf-8")
            q_path = root / "q.csv"
            q_path.write_text("1,0.01\n100,0.01\n", encoding="utf-8")
            auto = official.compute_spectrum(
                trace, calibration, q_path, nfft=2048,
                pre_sec=0.01, post_sec=0.02, threshold=0.1,
            )
            explicit = official.compute_spectrum_at_trigger(
                trace, calibration, q_path, nfft=2048,
                pre_sec=0.01, post_sec=0.02,
            )
        self.assertAlmostEqual(auto.time_window.peak_time, 0.2)
        self.assertAlmostEqual(explicit.time_window.peak_time, 0.7)
        self.assertEqual((explicit.time_window.i0, explicit.time_window.i1), (690, 719))
        self.assertNotEqual(auto.time_window.i0, explicit.time_window.i0)
        np.testing.assert_array_equal(time, original_time)
        np.testing.assert_array_equal(voltage, original_voltage)

    def test_spectrum_bodies_match_when_auto_peak_and_trigger_match(self):
        time = np.arange(4000, dtype=float) / 20000.0 + 12.5
        relative = time - time[0]
        voltage = 0.002 * np.sin(2 * np.pi * 700 * relative)
        voltage[2000] += 1.0
        trace = official.BlockTrace(
            time, voltage, 20000.0, float(time[2000]), 2000, 7, "selected"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "gain.csv"
            calibration.write_text("frequency,gain\n100,2\n10000,2\n", encoding="utf-8")
            q_path = root / "q.csv"
            q_path.write_text("1,0.01\n100,0.01\n", encoding="utf-8")
            auto = official.compute_spectrum(
                trace, calibration, q_path, nfft=8192, pre_sec=0.03, post_sec=0.04,
            )
            explicit = official.compute_spectrum_at_trigger(
                trace, calibration, q_path, nfft=8192, pre_sec=0.03, post_sec=0.04,
            )
        self.assertEqual(auto.time_window.peak_time, explicit.time_window.peak_time)
        for field in dataclasses.fields(official.SpectrumResult):
            if field.name == "time_window":
                continue
            left = getattr(auto, field.name)
            right = getattr(explicit, field.name)
            if isinstance(left, np.ndarray):
                assert_arrays_equal(self, left, right)
                self.assertIsNot(left, right)
            else:
                self.assertEqual(left, right)

    def test_parameter_bounds_validation_matches_student(self):
        cases = [
            (None, None),
            ((1e-12, 10.0, 1.0), (1e-3, 1e5, 6.0)),
        ]
        for lower, upper in cases:
            expected = student.validate_fit_parameter_bounds(lower, upper)
            actual = official.validate_fit_parameter_bounds(lower, upper)
            assert_arrays_equal(self, actual[0], expected[0])
            assert_arrays_equal(self, actual[1], expected[1])
        for lower, upper in [((1, 2), (2, 3)), ((0, 1, 1), (2, 3, 4)), ((2, 1, 1), (1, 3, 4))]:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(ValueError, ".+"):
                    official.validate_fit_parameter_bounds(lower, upper)

    def test_synthetic_omega_n_fit_matches_student(self):
        frequency = np.logspace(2, 5, 500)
        params = np.array([2e-8, 7000.0, 2.2])
        amplitude = np.exp(student.model_ln_amp(params, np.log(frequency)))
        kwargs = dict(
            time_window=mock.sentinel.time_window,
            freq_hz=np.concatenate(([0.0], frequency)),
            amp_raw=np.ones(1), amp_noise_raw=np.ones(1),
            amp_voltage_raw=np.ones(1), amp_noise_voltage_raw=np.ones(1),
            amp_cal=np.ones(1), amp_noise_cal=np.ones(1),
            valid_cal_mask=np.ones(1, dtype=bool), valid_noise_cal_mask=np.ones(1, dtype=bool),
            f7_hz=frequency, y7_cal=amplitude, y7_noise_cal=amplitude,
            y7_qcorr=amplitude, calibration_path=Path("cal.csv"), q_path=Path("q.csv"),
        )
        expected = student.fit_omega_n_q(student.SpectrumResult(**kwargs), np.log(200), np.log(80000))
        actual = official.fit_omega_n_q(official.SpectrumResult(**kwargs), np.log(200), np.log(80000))
        for name in ("omega0", "fc_hz", "n", "c", "r2", "lnf_min", "lnf_max"):
            self.assertAlmostEqual(getattr(actual, name), getattr(expected, name), places=12)
        assert_arrays_equal(self, actual.mask_fit, expected.mask_fit)
        assert_arrays_equal(self, actual.f_model_hz, expected.f_model_hz)
        assert_arrays_equal(self, actual.m_model_amp, expected.m_model_amp)

    def test_fit_failure_for_too_few_points_matches_student(self):
        frequency = np.array([100.0, 200.0, 300.0])
        kwargs = dict(
            time_window=mock.sentinel.time_window,
            freq_hz=np.concatenate(([0.0], frequency)),
            amp_raw=np.ones(1), amp_noise_raw=np.ones(1), amp_voltage_raw=np.ones(1),
            amp_noise_voltage_raw=np.ones(1), amp_cal=np.ones(1), amp_noise_cal=np.ones(1),
            valid_cal_mask=np.ones(1, dtype=bool), valid_noise_cal_mask=np.ones(1, dtype=bool),
            f7_hz=frequency, y7_cal=np.ones(3), y7_noise_cal=np.ones(3), y7_qcorr=np.ones(3),
            calibration_path=Path("cal.csv"), q_path=Path("q.csv"),
        )
        for module in (student, official):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "at least 10 points"):
                    module.fit_omega_n_q(module.SpectrumResult(**kwargs))


class ScopeTests(unittest.TestCase):
    def test_unchanged_numerical_function_bodies_match_student_ast(self):
        names = (
            "scale_spectrum_for_seismology_fit",
            "general_cosine",
            "blackman_harris_window",
            "tukey_window",
            "postzero_pad",
            "apply_extra_filter",
            "_auto_is_db",
            "load_csv_gain",
            "calib_interp_for",
            "load_q_data",
            "apply_q_correction",
            "_prepare_time_window",
            "model_ln_amp",
            "validate_fit_parameter_bounds",
            "_initial_guess",
            "fit_omega_n_q",
        )
        for name in names:
            with self.subTest(name=name):
                expected = ast.parse(inspect.getsource(getattr(student, name))).body[0].body
                actual = ast.parse(inspect.getsource(getattr(official, name))).body[0].body
                if actual and isinstance(actual[0], ast.Expr) and isinstance(
                    actual[0].value, ast.Constant
                ) and isinstance(actual[0].value.value, str):
                    actual = actual[1:]
                self.assertEqual(
                    ast.dump(ast.Module(body=actual, type_ignores=[]), include_attributes=False),
                    ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False),
                )

    def test_module_has_no_forbidden_infrastructure(self):
        source = inspect.getsource(official)
        for forbidden in (
            "import tkinter", "import h5py", "class TPC5Dataset", "np.savez",
            "json.dump", "SENSOR_TO_CHANNEL", "SENSOR_TO_A_LABEL",
            "saving_event_data", "BAC_result", "monkeypatch",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
