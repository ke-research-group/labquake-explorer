"""Tim-compatible BAC source-geometry calculations.

The grid, angle, and radiation-coefficient behavior is imported from the
current student-tim 1-D and 2-D production scaling scripts.  Coordinates and
fault dimensions are explicit centimetre values; no case, specimen, sensor,
channel, schema, or persistence identity is inferred here.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np


TIM_DEFAULT_GEOMETRY_GRID_RESOLUTION = 200
TIM_RADIATION_ANGLE_TABLE_DEG = (
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
)
TIM_RADIATION_COEFFICIENT_TABLE = (
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
)


@dataclass(frozen=True)
class BACGeometrySolution:
    """One Tim geometry solution and its source point.

    ``source_position_cm`` uses Tim's X/Y/Z centimetre coordinates.
    ``incidence_angle_deg`` is measured from the absolute Z separation.
    ``geometry_factor_m`` is source-receiver distance in metres divided by
    the radiation coefficient.
    """

    source_position_cm: tuple[float, float, float]
    source_receiver_distance_m: float
    incidence_angle_deg: float
    radiation_coefficient: float
    geometry_factor_m: float


@dataclass(frozen=True)
class BACGeometrySolutionSet:
    """Tim Min/Mid/Max geometry-grid representative solutions.

    Minimum and maximum are selected by the geometry factor.  Middle is the
    fault-center source, not a median or confidence interval.
    """

    minimum: BACGeometrySolution
    middle: BACGeometrySolution
    maximum: BACGeometrySolution


def _finite_real_scalar(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_scalar(name: str, value: Real) -> float:
    converted = _finite_real_scalar(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return converted


def _coordinate(name: str, value: Sequence[Real]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three coordinates")
    values = list(value)
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in values
    ):
        raise ValueError(f"{name} must contain real numeric coordinates")
    converted = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return converted


def _grid_resolution(value: Integral) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("grid_resolution must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError("grid_resolution must be a positive integer")
    return converted


def _radiation_tables(
    angle_table_deg: Sequence[Real],
    coefficient_table: Sequence[Real],
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(angle_table_deg)
    coefficients = np.asarray(coefficient_table)
    if angles.ndim != 1:
        raise ValueError("radiation_angle_table_deg must be one-dimensional")
    if coefficients.ndim != 1:
        raise ValueError("radiation_coefficient_table must be one-dimensional")
    if len(angles) != len(coefficients):
        raise ValueError("radiation angle and coefficient tables must have equal length")
    if len(angles) < 2:
        raise ValueError("radiation tables must contain at least two points")
    if angles.dtype.kind not in "iuf" or coefficients.dtype.kind not in "iuf":
        raise ValueError("radiation tables must contain real numeric values")
    angles = np.asarray(angles, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("radiation_angle_table_deg must contain only finite values")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("radiation_coefficient_table must contain only finite values")
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("radiation_angle_table_deg must be strictly increasing")
    return angles, coefficients


def interpolate_tim_radiation_coefficient(
    angle_deg: float,
    *,
    radiation_angle_table_deg: Sequence[Real] = TIM_RADIATION_ANGLE_TABLE_DEG,
    radiation_coefficient_table: Sequence[Real] = TIM_RADIATION_COEFFICIENT_TABLE,
) -> float:
    """Interpolate Tim's radiation coefficient for an angle in degrees.

    Values outside the table use the first or last coefficient, matching
    ``numpy.interp`` in both current Tim production paths.
    """

    angle = _finite_real_scalar("angle_deg", angle_deg)
    angles, coefficients = _radiation_tables(
        radiation_angle_table_deg, radiation_coefficient_table
    )
    return float(
        np.interp(
            angle,
            angles,
            coefficients,
            left=coefficients[0],
            right=coefficients[-1],
        )
    )


def _solution(
    point_cm: np.ndarray,
    distance_cm: float,
    angle_deg: float,
    radiation_coefficient: float,
) -> BACGeometrySolution:
    distance_m = float(distance_cm) / 100.0
    return BACGeometrySolution(
        source_position_cm=tuple(float(value) for value in point_cm),
        source_receiver_distance_m=distance_m,
        incidence_angle_deg=float(angle_deg),
        radiation_coefficient=float(radiation_coefficient),
        geometry_factor_m=float(np.divide(distance_m, radiation_coefficient)),
    )


def _calculate_geometry_solutions(
    points_cm: np.ndarray,
    fault_center_cm: np.ndarray,
    sensor_position_cm: np.ndarray,
    radiation_angles: np.ndarray,
    radiation_coefficients: np.ndarray,
    clip_center_angle: bool,
) -> BACGeometrySolutionSet:
    vectors = sensor_position_cm - points_cm
    distances = np.linalg.norm(vectors, axis=1)
    vertical = np.abs(points_cm[:, 2] - sensor_position_cm[2])
    angles = np.degrees(np.arccos(np.clip(vertical / distances, -1.0, 1.0)))
    coefficients = np.asarray(
        np.interp(
            angles,
            radiation_angles,
            radiation_coefficients,
            left=radiation_coefficients[0],
            right=radiation_coefficients[-1],
        )
    )
    factors = (distances / 100.0) / coefficients

    center_vector = sensor_position_cm - fault_center_cm
    center_distance = float(np.linalg.norm(center_vector))
    center_ratio = (
        abs(fault_center_cm[2] - sensor_position_cm[2]) / center_distance
    )
    if clip_center_angle:
        center_ratio = np.clip(center_ratio, -1.0, 1.0)
    center_angle = float(np.degrees(np.arccos(center_ratio)))
    center_coefficient = float(
        np.interp(
            center_angle,
            radiation_angles,
            radiation_coefficients,
            left=radiation_coefficients[0],
            right=radiation_coefficients[-1],
        )
    )

    minimum_index = int(np.argmin(factors))
    maximum_index = int(np.argmax(factors))
    return BACGeometrySolutionSet(
        minimum=_solution(
            points_cm[minimum_index],
            distances[minimum_index],
            angles[minimum_index],
            coefficients[minimum_index],
        ),
        middle=_solution(
            fault_center_cm,
            center_distance,
            center_angle,
            center_coefficient,
        ),
        maximum=_solution(
            points_cm[maximum_index],
            distances[maximum_index],
            angles[maximum_index],
            coefficients[maximum_index],
        ),
    )


def calculate_rectangular_fault_geometry_solutions(
    *,
    fault_center_cm: Sequence[Real],
    fault_length_cm: float,
    fault_height_cm: float,
    sensor_position_cm: Sequence[Real],
    radiation_angle_table_deg: Sequence[Real] = TIM_RADIATION_ANGLE_TABLE_DEG,
    radiation_coefficient_table: Sequence[Real] = TIM_RADIATION_COEFFICIENT_TABLE,
    grid_resolution: int = TIM_DEFAULT_GEOMETRY_GRID_RESOLUTION,
) -> BACGeometrySolutionSet:
    """Calculate Tim Min/Mid/Max solutions for an X-Z rectangle in cm."""

    center = _coordinate("fault_center_cm", fault_center_cm)
    sensor = _coordinate("sensor_position_cm", sensor_position_cm)
    length = _positive_scalar("fault_length_cm", fault_length_cm)
    height = _positive_scalar("fault_height_cm", fault_height_cm)
    resolution = _grid_resolution(grid_resolution)
    radiation_angles, radiation_coefficients = _radiation_tables(
        radiation_angle_table_deg, radiation_coefficient_table
    )

    xs = np.linspace(center[0] - length / 2.0, center[0] + length / 2.0, resolution)
    zs = np.linspace(center[2] - height / 2.0, center[2] + height / 2.0, resolution)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    points = np.column_stack(
        (xx.ravel(), np.full(xx.size, center[1]), zz.ravel())
    )
    return _calculate_geometry_solutions(
        points,
        center,
        sensor,
        radiation_angles,
        radiation_coefficients,
        True,
    )


def calculate_circular_fault_geometry_solutions(
    *,
    fault_center_cm: Sequence[Real],
    fault_radius_cm: float,
    sensor_position_cm: Sequence[Real],
    radiation_angle_table_deg: Sequence[Real] = TIM_RADIATION_ANGLE_TABLE_DEG,
    radiation_coefficient_table: Sequence[Real] = TIM_RADIATION_COEFFICIENT_TABLE,
    grid_resolution: int = TIM_DEFAULT_GEOMETRY_GRID_RESOLUTION,
) -> BACGeometrySolutionSet:
    """Calculate Tim Min/Mid/Max solutions for an X-Z circular disk in cm."""

    center = _coordinate("fault_center_cm", fault_center_cm)
    sensor = _coordinate("sensor_position_cm", sensor_position_cm)
    radius = _positive_scalar("fault_radius_cm", fault_radius_cm)
    resolution = _grid_resolution(grid_resolution)
    radiation_angles, radiation_coefficients = _radiation_tables(
        radiation_angle_table_deg, radiation_coefficient_table
    )

    axis = np.linspace(-radius, radius, resolution)
    dx, dz = np.meshgrid(axis, axis, indexing="xy")
    inside = dx * dx + dz * dz <= radius * radius
    points = np.column_stack(
        (
            center[0] + dx[inside],
            np.full(np.count_nonzero(inside), center[1]),
            center[2] + dz[inside],
        )
    )
    return _calculate_geometry_solutions(
        points,
        center,
        sensor,
        radiation_angles,
        radiation_coefficients,
        False,
    )
