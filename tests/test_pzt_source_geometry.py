import dataclasses
import math
from pathlib import Path
import unittest
import warnings

import numpy as np

from labquake_explorer.analysis import (
    BACGeometrySolution,
    BACGeometrySolutionSet,
    TIM_DEFAULT_GEOMETRY_GRID_RESOLUTION,
    TIM_RADIATION_ANGLE_TABLE_DEG,
    TIM_RADIATION_COEFFICIENT_TABLE,
    calculate_bac_source_parameters,
    calculate_circular_fault_geometry_solutions,
    calculate_rectangular_fault_geometry_solutions,
    interpolate_tim_radiation_coefficient,
)


def tim_interpolate_sa(angle):
    angles = np.arange(0.0, 90.0, 5.0)
    values = np.array(
        [
            2.00,
            1.99,
            1.96,
            1.92,
            1.86,
            1.79,
            1.70,
            1.60,
            1.49,
            1.38,
            1.26,
            1.14,
            1.02,
            0.90,
            0.79,
            0.67,
            0.54,
            0.35,
        ]
    )
    return np.interp(angle, angles, values, left=2.00, right=0.35)


def tim_geometry_reference(points, center, sensor):
    vectors = sensor - points
    distances = np.linalg.norm(vectors, axis=1)
    vertical = np.abs(points[:, 2] - sensor[2])
    angles = np.degrees(np.arccos(np.clip(vertical / distances, -1.0, 1.0)))
    sa = np.asarray(tim_interpolate_sa(angles))
    factor = (distances / 100.0) / sa

    def item(index):
        return points[index], distances[index], angles[index], sa[index], factor[index]

    center_vector = sensor - center
    center_r = float(np.linalg.norm(center_vector))
    center_angle = float(
        np.degrees(
            np.arccos(
                np.clip(abs(center[2] - sensor[2]) / center_r, -1.0, 1.0)
            )
        )
    )
    center_sa = float(tim_interpolate_sa(center_angle))
    return {
        "minimum": item(int(np.argmin(factor))),
        "middle": (
            center,
            center_r,
            center_angle,
            center_sa,
            (center_r / 100.0) / center_sa,
        ),
        "maximum": item(int(np.argmax(factor))),
    }


def rectangular_points(center, length, height, resolution):
    xs = np.linspace(center[0] - length / 2.0, center[0] + length / 2.0, resolution)
    zs = np.linspace(center[2] - height / 2.0, center[2] + height / 2.0, resolution)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    return np.column_stack((xx.ravel(), np.full(xx.size, center[1]), zz.ravel()))


def circular_points(center, radius, resolution):
    axis = np.linspace(-radius, radius, resolution)
    dx, dz = np.meshgrid(axis, axis, indexing="xy")
    inside = dx * dx + dz * dz <= radius * radius
    return (
        np.column_stack(
            (
                center[0] + dx[inside],
                np.full(np.count_nonzero(inside), center[1]),
                center[2] + dz[inside],
            )
        ),
        inside,
    )


class BACSourceGeometryTests(unittest.TestCase):
    def assert_solution_matches(self, actual, expected):
        point, distance_cm, angle, coefficient, factor = expected
        np.testing.assert_allclose(actual.source_position_cm, point, rtol=0.0, atol=0.0)
        self.assertEqual(actual.source_receiver_distance_m, float(distance_cm) / 100.0)
        self.assertEqual(actual.incidence_angle_deg, float(angle))
        self.assertEqual(actual.radiation_coefficient, float(coefficient))
        self.assertEqual(actual.geometry_factor_m, float(factor))

    def test_constants_match_current_tim_1d_and_2d(self):
        self.assertEqual(TIM_DEFAULT_GEOMETRY_GRID_RESOLUTION, 200)
        self.assertEqual(TIM_RADIATION_ANGLE_TABLE_DEG, tuple(np.arange(0.0, 90.0, 5.0)))
        self.assertEqual(
            TIM_RADIATION_COEFFICIENT_TABLE,
            (
                2.00,
                1.99,
                1.96,
                1.92,
                1.86,
                1.79,
                1.70,
                1.60,
                1.49,
                1.38,
                1.26,
                1.14,
                1.02,
                0.90,
                0.79,
                0.67,
                0.54,
                0.35,
            ),
        )

    def test_result_dataclasses_are_frozen_and_scalar(self):
        solution = BACGeometrySolution((1.0, 2.0, 3.0), 0.1, 45.0, 1.38, 0.1 / 1.38)
        result = BACGeometrySolutionSet(solution, solution, solution)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.minimum = solution
        self.assertEqual(
            [field.name for field in dataclasses.fields(BACGeometrySolution)],
            [
                "source_position_cm",
                "source_receiver_distance_m",
                "incidence_angle_deg",
                "radiation_coefficient",
                "geometry_factor_m",
            ],
        )
        self.assertTrue(all(isinstance(value, float) for value in solution.source_position_cm))

    def test_radiation_interpolation_endpoints_interior_and_out_of_range(self):
        for angle in (-10.0, 0.0, 25.0, 42.5, 85.0, 100.0):
            with self.subTest(angle=angle):
                self.assertEqual(
                    interpolate_tim_radiation_coefficient(angle),
                    float(tim_interpolate_sa(angle)),
                )

    def test_custom_radiation_table_preserves_numpy_interp_behavior(self):
        self.assertEqual(
            interpolate_tim_radiation_coefficient(
                5.0,
                radiation_angle_table_deg=(0.0, 10.0),
                radiation_coefficient_table=(4.0, 2.0),
            ),
            3.0,
        )

    def test_distance_angle_and_geometry_factor_match_tim_expression(self):
        center = np.array([25.0, 0.0, 5.0])
        sensor = np.array([10.0, 10.0, 5.0])
        result = calculate_rectangular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_length_cm=7.0,
            fault_height_cm=10.0,
            sensor_position_cm=sensor,
            grid_resolution=3,
        )
        expected = tim_geometry_reference(
            rectangular_points(center, 7.0, 10.0, 3), center, sensor
        )
        self.assert_solution_matches(result.middle, expected["middle"])

    def test_current_1d_rectangular_geometry_parity(self):
        center = np.array([25.0, 0.0, 5.0])
        sensor = np.array([10.0, 10.0, 5.0])
        points = rectangular_points(center, 7.0, 10.0, 200)
        expected = tim_geometry_reference(points, center, sensor)
        actual = calculate_rectangular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_length_cm=7.0,
            fault_height_cm=10.0,
            sensor_position_cm=sensor,
        )
        self.assertEqual(len(points), 200 * 200)
        for name in ("minimum", "middle", "maximum"):
            self.assert_solution_matches(getattr(actual, name), expected[name])

    def test_current_2d_circular_geometry_parity(self):
        center = np.array([25.0, 0.0, 25.0])
        sensor = np.array([10.0, 4.0, 0.0])
        points, inside = circular_points(center, 10.0, 200)
        expected = tim_geometry_reference(points, center, sensor)
        actual = calculate_circular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_radius_cm=10.0,
            sensor_position_cm=sensor,
        )
        self.assertEqual(len(points), int(np.count_nonzero(inside)))
        for name in ("minimum", "middle", "maximum"):
            self.assert_solution_matches(getattr(actual, name), expected[name])

    def test_source_parameter_integration_matches_tim_parameter_row(self):
        solutions = calculate_circular_fault_geometry_solutions(
            fault_center_cm=(25.0, 0.0, 25.0),
            fault_radius_cm=10.0,
            sensor_position_cm=(10.0, 4.0, 0.0),
        )
        omega0 = 2.7e-9
        fc = 12500.0
        for solution in (solutions.minimum, solutions.middle, solutions.maximum):
            result = calculate_bac_source_parameters(
                omega0=omega0,
                corner_frequency_hz=fc,
                source_receiver_distance_m=solution.source_receiver_distance_m,
                radiation_coefficient=solution.radiation_coefficient,
            )
            expected_m0 = (
                4.0
                * math.pi
                * solution.source_receiver_distance_m
                * 2773.0**3
                * 1148.0
                * omega0
                / (0.64 * solution.radiation_coefficient)
            )
            expected_area_mm2 = math.pi * ((0.32 * 1764.0) / fc) ** 2 * 1.0e6
            expected_stress_mpa = (
                7.0 * expected_m0 * fc**3 / (16.0 * (0.32 * 1764.0) ** 3) / 1.0e6
            )
            self.assertEqual(result.seismic_moment_nm, expected_m0)
            self.assertEqual(result.rupture_area_m2 * 1.0e6, expected_area_mm2)
            self.assertEqual(result.stress_drop_pa / 1.0e6, expected_stress_mpa)

    def test_randomized_distance_angle_and_interpolation_parity(self):
        rng = np.random.default_rng(20260805)
        for _ in range(50):
            center = rng.uniform(-100.0, 100.0, 3)
            sensor = rng.uniform(-100.0, 100.0, 3)
            if np.array_equal(center, sensor):
                sensor[0] += 1.0
            points = rectangular_points(center, 5.0, 7.0, 2)
            expected = tim_geometry_reference(points, center, sensor)
            actual = calculate_rectangular_fault_geometry_solutions(
                fault_center_cm=center,
                fault_length_cm=5.0,
                fault_height_cm=7.0,
                sensor_position_cm=sensor,
                grid_resolution=2,
            )
            for name in ("minimum", "middle", "maximum"):
                self.assert_solution_matches(getattr(actual, name), expected[name])

    def test_grid_endpoints_masks_and_minimum_resolution(self):
        center = np.array([2.0, 3.0, 4.0])
        rectangle = rectangular_points(center, 6.0, 8.0, 2)
        np.testing.assert_array_equal(
            rectangle,
            np.array([[-1.0, 3.0, 0.0], [5.0, 3.0, 0.0], [-1.0, 3.0, 8.0], [5.0, 3.0, 8.0]]),
        )
        _, inside = circular_points(center, 2.0, 3)
        self.assertTrue(inside[0, 1])
        self.assertTrue(inside[1, 0])
        self.assertTrue(inside[1, 2])
        self.assertTrue(inside[2, 1])
        result = calculate_rectangular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_length_cm=6.0,
            fault_height_cm=8.0,
            sensor_position_cm=(20.0, 3.0, 4.0),
            grid_resolution=1,
        )
        self.assertIsInstance(result, BACGeometrySolutionSet)

    def test_zero_distance_preserves_tim_nan_behavior(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = tim_geometry_reference(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            )
            actual = calculate_rectangular_fault_geometry_solutions(
                fault_center_cm=(0.0, 0.0, 0.0),
                fault_length_cm=1.0,
                fault_height_cm=1.0,
                sensor_position_cm=(-0.5, 0.0, -0.5),
                grid_resolution=1,
            )
        self.assertTrue(math.isnan(actual.minimum.incidence_angle_deg))
        self.assertTrue(math.isnan(actual.minimum.radiation_coefficient))
        self.assertTrue(math.isnan(expected["minimum"][2]))

    def test_zero_radiation_coefficient_preserves_numpy_division_behavior(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = calculate_rectangular_fault_geometry_solutions(
                fault_center_cm=(0.0, 0.0, 0.0),
                fault_length_cm=2.0,
                fault_height_cm=2.0,
                sensor_position_cm=(10.0, 0.0, 0.0),
                radiation_angle_table_deg=(0.0, 90.0),
                radiation_coefficient_table=(0.0, 0.0),
                grid_resolution=2,
            )
        self.assertTrue(math.isinf(result.minimum.geometry_factor_m))
        self.assertEqual(result.minimum.radiation_coefficient, 0.0)

    def test_empty_circular_mask_preserves_tim_argmin_failure(self):
        with self.assertRaisesRegex(ValueError, "argmin"):
            calculate_circular_fault_geometry_solutions(
                fault_center_cm=(0.0, 0.0, 0.0),
                fault_radius_cm=1.0,
                sensor_position_cm=(10.0, 0.0, 0.0),
                grid_resolution=1,
            )

    def test_sensor_outside_fault_and_scale_extremes_are_supported(self):
        small = calculate_circular_fault_geometry_solutions(
            fault_center_cm=(0.0, 0.0, 0.0),
            fault_radius_cm=1.0e-6,
            sensor_position_cm=(1000.0, 10.0, 20.0),
            grid_resolution=3,
        )
        large = calculate_circular_fault_geometry_solutions(
            fault_center_cm=(0.0, 0.0, 0.0),
            fault_radius_cm=1.0e6,
            sensor_position_cm=(2.0e6, 10.0, 20.0),
            grid_resolution=3,
        )
        self.assertGreater(small.middle.source_receiver_distance_m, 0.0)
        self.assertGreater(large.middle.source_receiver_distance_m, 0.0)

    def test_structural_validation_names_invalid_field(self):
        base = {
            "fault_center_cm": (0.0, 0.0, 0.0),
            "fault_length_cm": 2.0,
            "fault_height_cm": 2.0,
            "sensor_position_cm": (3.0, 0.0, 0.0),
            "grid_resolution": 2,
        }
        invalid = [
            ("fault_center_cm", (0.0, 0.0)),
            ("fault_center_cm", (0.0, np.nan, 0.0)),
            ("sensor_position_cm", (0.0, True, 0.0)),
            ("fault_length_cm", 0.0),
            ("fault_height_cm", -1.0),
            ("grid_resolution", True),
            ("grid_resolution", 1.5),
        ]
        for name, value in invalid:
            with self.subTest(name=name):
                arguments = dict(base)
                arguments[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    calculate_rectangular_fault_geometry_solutions(**arguments)

    def test_radiation_table_validation(self):
        cases = [
            ((0.0,), (1.0,), "at least two"),
            ((0.0, 1.0), (1.0,), "equal length"),
            ((0.0, 0.0), (1.0, 2.0), "strictly increasing"),
            ((0.0, np.nan), (1.0, 2.0), "radiation_angle_table_deg"),
            ((0.0, 1.0), (1.0, np.inf), "radiation_coefficient_table"),
        ]
        for angles, coefficients, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    interpolate_tim_radiation_coefficient(
                        0.5,
                        radiation_angle_table_deg=angles,
                        radiation_coefficient_table=coefficients,
                    )

    def test_input_arrays_are_not_modified_and_results_are_new(self):
        center = np.array([25.0, 0.0, 5.0])
        sensor = np.array([10.0, 10.0, 5.0])
        original_center = center.copy()
        original_sensor = sensor.copy()
        first = calculate_rectangular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_length_cm=7.0,
            fault_height_cm=10.0,
            sensor_position_cm=sensor,
            grid_resolution=3,
        )
        second = calculate_rectangular_fault_geometry_solutions(
            fault_center_cm=center,
            fault_length_cm=7.0,
            fault_height_cm=10.0,
            sensor_position_cm=sensor,
            grid_resolution=3,
        )
        np.testing.assert_array_equal(center, original_center)
        np.testing.assert_array_equal(sensor, original_sensor)
        self.assertIsNot(first, second)
        self.assertIsNot(first.minimum, second.minimum)

    def test_module_has_no_forbidden_infrastructure_or_mapping(self):
        source = (
            Path(__file__).parents[1]
            / "labquake_explorer"
            / "analysis"
            / "pzt_source_geometry.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "tkinter",
            "DataManager",
            "h5py",
            "pandas",
            "matplotlib",
            "npz",
            "csv",
            "A1",
            "AS01",
            "CASE_DIAMETER",
            "importlib",
            "open(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
