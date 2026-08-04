"""Parity and scope tests for Tim's in-memory EGF numerical core."""

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

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
            "compute_egf_spectrum",
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
