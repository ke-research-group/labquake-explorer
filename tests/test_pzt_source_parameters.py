import dataclasses
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from labquake_explorer.analysis import (
    BACSourceParameterResult,
    TIM_DEFAULT_DENSITY_KG_M3,
    TIM_DEFAULT_P_WAVE_SPEED_M_S,
    TIM_DEFAULT_RADIATION_PATTERN_FACTOR,
    TIM_DEFAULT_SHEAR_WAVE_SPEED_M_S,
    TIM_DEFAULT_SOURCE_MODEL_K,
    calculate_bac_source_parameters,
)


def tim_parameter_row_expression(
    omega0,
    corner_frequency_hz,
    source_receiver_distance_m,
    radiation_coefficient,
    density_kg_m3=1148.0,
    p_wave_speed_m_s=2773.0,
    shear_wave_speed_m_s=1764.0,
    source_model_k=0.32,
    radiation_pattern_factor=0.64,
):
    """Direct scalar transcription of both current Tim parameter_row functions."""
    m0 = (
        4.0
        * math.pi
        * source_receiver_distance_m
        * p_wave_speed_m_s**3
        * density_kg_m3
        * omega0
        / (radiation_pattern_factor * radiation_coefficient)
    )
    stress_drop_mpa = (
        7.0
        * m0
        * corner_frequency_hz**3
        / (16.0 * (source_model_k * shear_wave_speed_m_s) ** 3)
        / 1.0e6
    )
    rupture_area_mm2 = (
        math.pi
        * ((source_model_k * shear_wave_speed_m_s) / corner_frequency_hz) ** 2
        * 1.0e6
    )
    return m0, stress_drop_mpa, rupture_area_mm2


class BACSourceParameterTests(unittest.TestCase):
    def assert_tim_parity(self, **inputs):
        expected_m0, expected_stress_mpa, expected_area_mm2 = (
            tim_parameter_row_expression(**inputs)
        )
        result = calculate_bac_source_parameters(**inputs)
        self.assertEqual(result.seismic_moment_nm, expected_m0)
        self.assertEqual(result.stress_drop_pa / 1.0e6, expected_stress_mpa)
        self.assertEqual(result.rupture_area_m2 * 1.0e6, expected_area_mm2)
        self.assertEqual(
            result.source_radius_m,
            TIM_DEFAULT_SOURCE_MODEL_K
            * TIM_DEFAULT_SHEAR_WAVE_SPEED_M_S
            / inputs["corner_frequency_hz"],
        )

    def test_dataclass_contract(self):
        self.assertEqual(
            [field.name for field in dataclasses.fields(BACSourceParameterResult)],
            [
                "seismic_moment_nm",
                "source_radius_m",
                "rupture_area_m2",
                "stress_drop_pa",
            ],
        )
        result = calculate_bac_source_parameters(
            omega0=2.5e-9,
            corner_frequency_hz=12000.0,
            source_receiver_distance_m=0.23,
            radiation_coefficient=1.2,
        )
        for field in dataclasses.fields(result):
            self.assertIsInstance(getattr(result, field.name), float)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.seismic_moment_nm = 0.0

    def test_tim_default_constants(self):
        self.assertEqual(TIM_DEFAULT_DENSITY_KG_M3, 1148.0)
        self.assertEqual(TIM_DEFAULT_P_WAVE_SPEED_M_S, 2773.0)
        self.assertEqual(TIM_DEFAULT_SHEAR_WAVE_SPEED_M_S, 1764.0)
        self.assertEqual(TIM_DEFAULT_SOURCE_MODEL_K, 0.32)
        self.assertEqual(TIM_DEFAULT_RADIATION_PATTERN_FACTOR, 0.64)

    def test_representative_1d_parameter_row_parity(self):
        self.assert_tim_parity(
            omega0=3.7e-9,
            corner_frequency_hz=14500.0,
            source_receiver_distance_m=0.284,
            radiation_coefficient=1.15,
        )

    def test_representative_2d_parameter_row_parity(self):
        self.assert_tim_parity(
            omega0=8.1e-10,
            corner_frequency_hz=8250.0,
            source_receiver_distance_m=0.317,
            radiation_coefficient=0.82,
        )

    def test_randomized_parity_including_signed_omega0(self):
        rng = np.random.default_rng(20260805)
        for _ in range(100):
            inputs = {
                "omega0": float(rng.uniform(-1.0e-8, 1.0e-8)),
                "corner_frequency_hz": float(rng.uniform(100.0, 50000.0)),
                "source_receiver_distance_m": float(rng.uniform(0.01, 2.0)),
                "radiation_coefficient": float(rng.uniform(0.1, 2.0)),
            }
            self.assert_tim_parity(**inputs)

    def test_signed_and_zero_omega0_behavior(self):
        common = {
            "corner_frequency_hz": 10000.0,
            "source_receiver_distance_m": 0.2,
            "radiation_coefficient": 1.0,
        }
        positive = calculate_bac_source_parameters(omega0=2.0e-9, **common)
        negative = calculate_bac_source_parameters(omega0=-2.0e-9, **common)
        zero = calculate_bac_source_parameters(omega0=0.0, **common)
        self.assertGreater(positive.seismic_moment_nm, 0.0)
        self.assertLess(negative.seismic_moment_nm, 0.0)
        self.assertEqual(negative.seismic_moment_nm, -positive.seismic_moment_nm)
        self.assertEqual(negative.stress_drop_pa, -positive.stress_drop_pa)
        self.assertEqual(zero.seismic_moment_nm, 0.0)
        self.assertEqual(zero.stress_drop_pa, 0.0)

    def test_numpy_real_scalars_are_accepted(self):
        result = calculate_bac_source_parameters(
            omega0=np.float64(1.0e-9),
            corner_frequency_hz=np.int64(10000),
            source_receiver_distance_m=np.float32(0.2),
            radiation_coefficient=np.float64(1.0),
        )
        self.assertIsInstance(result.seismic_moment_nm, float)

    def test_invalid_scalar_values_name_the_parameter(self):
        common = {
            "omega0": 1.0e-9,
            "corner_frequency_hz": 10000.0,
            "source_receiver_distance_m": 0.2,
            "radiation_coefficient": 1.0,
        }
        invalid_cases = [
            ("omega0", True),
            ("omega0", np.bool_(True)),
            ("omega0", np.array(1.0)),
            ("omega0", np.nan),
            ("omega0", np.inf),
            ("corner_frequency_hz", 0.0),
            ("source_receiver_distance_m", 0.0),
            ("radiation_coefficient", 0.0),
            ("density_kg_m3", 0.0),
            ("p_wave_speed_m_s", 0.0),
            ("shear_wave_speed_m_s", 0.0),
            ("source_model_k", 0.0),
            ("radiation_pattern_factor", 0.0),
        ]
        for name, value in invalid_cases:
            with self.subTest(name=name, value=value):
                arguments = dict(common)
                arguments[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    calculate_bac_source_parameters(**arguments)

    def test_inputs_are_not_modified_and_results_are_distinct(self):
        inputs = {
            "omega0": np.float64(1.5e-9),
            "corner_frequency_hz": np.float64(9000.0),
            "source_receiver_distance_m": np.float64(0.18),
            "radiation_coefficient": np.float64(1.1),
        }
        original = dict(inputs)
        first = calculate_bac_source_parameters(**inputs)
        second = calculate_bac_source_parameters(**inputs)
        self.assertEqual(inputs, original)
        self.assertIsNot(first, second)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in dataclasses.astuple(first)))

    def test_module_has_no_forbidden_runtime_dependencies_or_glue(self):
        module_path = (
            Path(__file__).parents[1]
            / "labquake_explorer"
            / "analysis"
            / "pzt_source_parameters.py"
        )
        source = module_path.read_text(encoding="utf-8")
        for forbidden in (
            "tkinter",
            "DataManager",
            "h5py",
            "pandas",
            "re.compile",
            "SENSOR_MAP",
            "CASE_DIAMETER",
            "matplotlib",
            "npz",
            "csv",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_api_does_not_mutate_external_state(self):
        with mock.patch("builtins.open") as open_mock:
            calculate_bac_source_parameters(
                omega0=1.0e-9,
                corner_frequency_hz=10000.0,
                source_receiver_distance_m=0.2,
                radiation_coefficient=1.0,
            )
        open_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
