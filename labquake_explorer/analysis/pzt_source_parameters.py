"""Tim-compatible scalar BAC source-parameter calculations.

The constants and formulas preserve the current student-tim 1-D and 2-D
``parameter_row`` production expressions and return SI-unit source parameters.
The fitted ``omega0`` sign is retained rather than converted to a magnitude.
Callers must provide the geometry-derived
source-receiver distance and radiation coefficient explicitly; this module
does not infer sensor, specimen, case, schema, or persistence details.
"""

from dataclasses import dataclass
import math
from numbers import Real


TIM_DEFAULT_DENSITY_KG_M3 = 1148.0
TIM_DEFAULT_P_WAVE_SPEED_M_S = 2773.0
TIM_DEFAULT_SHEAR_WAVE_SPEED_M_S = 1764.0
TIM_DEFAULT_SOURCE_MODEL_K = 0.32
TIM_DEFAULT_RADIATION_PATTERN_FACTOR = 0.64


@dataclass(frozen=True)
class BACSourceParameterResult:
    """BAC source parameters in SI units."""

    seismic_moment_nm: float
    source_radius_m: float
    rupture_area_m2: float
    stress_drop_pa: float


def _finite_real_scalar(name: str, value: Real) -> float:
    if isinstance(value, (bool,)) or not isinstance(value, Real):
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


def calculate_bac_source_parameters(
    *,
    omega0: float,
    corner_frequency_hz: float,
    source_receiver_distance_m: float,
    radiation_coefficient: float,
    density_kg_m3: float = TIM_DEFAULT_DENSITY_KG_M3,
    p_wave_speed_m_s: float = TIM_DEFAULT_P_WAVE_SPEED_M_S,
    shear_wave_speed_m_s: float = TIM_DEFAULT_SHEAR_WAVE_SPEED_M_S,
    source_model_k: float = TIM_DEFAULT_SOURCE_MODEL_K,
    radiation_pattern_factor: float = TIM_DEFAULT_RADIATION_PATTERN_FACTOR,
) -> BACSourceParameterResult:
    """Calculate Tim-compatible BAC source parameters from explicit scalars.

    ``omega0`` is the signed displacement amplitude in metres,
    ``corner_frequency_hz`` is in Hz, and ``source_receiver_distance_m`` is in
    metres.  Density and wave speeds use kg/m^3 and m/s.  Results are returned
    as seismic moment (N m), source radius (m), rupture area (m^2), and stress
    drop (Pa).  No absolute value is applied to ``omega0``.
    """

    omega0 = _finite_real_scalar("omega0", omega0)
    corner_frequency_hz = _positive_scalar(
        "corner_frequency_hz", corner_frequency_hz
    )
    source_receiver_distance_m = _positive_scalar(
        "source_receiver_distance_m", source_receiver_distance_m
    )
    radiation_coefficient = _positive_scalar(
        "radiation_coefficient", radiation_coefficient
    )
    density_kg_m3 = _positive_scalar("density_kg_m3", density_kg_m3)
    p_wave_speed_m_s = _positive_scalar("p_wave_speed_m_s", p_wave_speed_m_s)
    shear_wave_speed_m_s = _positive_scalar(
        "shear_wave_speed_m_s", shear_wave_speed_m_s
    )
    source_model_k = _positive_scalar("source_model_k", source_model_k)
    radiation_pattern_factor = _positive_scalar(
        "radiation_pattern_factor", radiation_pattern_factor
    )

    seismic_moment_nm = (
        4.0
        * math.pi
        * source_receiver_distance_m
        * p_wave_speed_m_s**3
        * density_kg_m3
        * omega0
        / (radiation_pattern_factor * radiation_coefficient)
    )
    source_radius_m = (
        source_model_k * shear_wave_speed_m_s / corner_frequency_hz
    )
    rupture_area_m2 = math.pi * (
        (source_model_k * shear_wave_speed_m_s) / corner_frequency_hz
    ) ** 2
    stress_drop_pa = (
        7.0
        * seismic_moment_nm
        * corner_frequency_hz**3
        / (16.0 * (source_model_k * shear_wave_speed_m_s) ** 3)
    )

    return BACSourceParameterResult(
        seismic_moment_nm=float(seismic_moment_nm),
        source_radius_m=float(source_radius_m),
        rupture_area_m2=float(rupture_area_m2),
        stress_drop_pa=float(stress_drop_pa),
    )
