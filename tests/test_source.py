import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis.source import (
    PHASE_CONSTANTS, Geometry, RadiationPattern, SourceParameters,
    far_field_ok, free_surface_amplification, geometry, moment_from_magnitude,
    moment_magnitude, plateau_from_moment, radiation_coefficient, radiation_pattern,
    seismic_moment, source_parameters, source_radius, stress_drop_eshelby,
    wavenumber_distance,
)

# a "PMMA-like" medium, chosen by the TEST (the module has no defaults)
RHO, VP, VS = 1180.0, 2700.0, 1350.0


def test_phase_constants():
    assert PHASE_CONSTANTS["P"]["radiation_rms"] == pytest.approx(math.sqrt(4 / 15))
    assert PHASE_CONSTANTS["S"]["radiation_rms"] == pytest.approx(math.sqrt(2 / 5))
    assert PHASE_CONSTANTS["P"]["k"] == 0.32 and PHASE_CONSTANTS["S"]["k"] == 0.372


def test_eshelby_anchor():
    # M0 = 1 N m, r = 1 cm -> 7 / (16 * 1e-6) Pa = 437.5 kPa
    assert stress_drop_eshelby(1.0, 0.01) == pytest.approx(437.5e3)


def test_moment_magnitude_anchor_and_inverse():
    assert moment_magnitude(10 ** 9.1) == pytest.approx(0.0, abs=1e-12)
    assert moment_magnitude(10 ** 10.6) == pytest.approx(1.0)
    assert moment_from_magnitude(moment_magnitude(3.7e2)) == pytest.approx(3.7e2)
    with pytest.raises(ValueError):
        moment_magnitude(0.0)


def test_seismic_moment_formula_and_round_trip():
    omega0 = 2.5e-9
    m0 = seismic_moment(omega0, RHO, VP, 0.05, 0.8, 1.5)
    assert m0 == pytest.approx(4 * math.pi * RHO * VP ** 3 * 0.05 * omega0 / (0.8 * 1.5))
    assert plateau_from_moment(m0, RHO, VP, 0.05, 0.8, 1.5) == pytest.approx(omega0)
    # polarity sign of the radiation coefficient does not change the moment
    assert seismic_moment(omega0, RHO, VP, 0.05, -0.8) == seismic_moment(omega0, RHO, VP, 0.05, 0.8)


def test_seismic_moment_rejects_nonpositive_plateau_never_abs():
    with pytest.raises(ValueError):
        seismic_moment(-2.5e-9, RHO, VP, 0.05, 0.8)
    with pytest.raises(ValueError):
        seismic_moment(0.0, RHO, VP, 0.05, 0.8)
    with pytest.raises(ValueError):
        seismic_moment(float("nan"), RHO, VP, 0.05, 0.8)
    with pytest.raises(ValueError):
        seismic_moment(1e-9, RHO, VP, 0.05, 0.0)       # node
    with pytest.raises(ValueError):
        seismic_moment(1e-9, RHO, VP, 0.05, 0.8, 0.0)  # bad free-surface factor


def test_source_radius_uses_vs():
    assert source_radius(50e3, VS, 0.32) == pytest.approx(0.32 * VS / 50e3)
    assert wavenumber_distance(10e3, 0.05, VP) == pytest.approx(2 * math.pi * 10e3 * 0.05 / VP)
    assert far_field_ok(3.0) and not far_field_ok(2.99) and not far_field_ok(float("nan"))


# --- radiation pattern -----------------------------------------------------

def test_radiation_fault_normal_sensor_is_p_node():
    n, s = (0, 0, 1), (1, 0, 0)
    pat = radiation_pattern(n, s, (0, 0, 1))
    assert pat.p == pytest.approx(0.0)
    assert pat.s == pytest.approx(1.0)          # S maximum on the fault normal
    assert pat.p_near_node
    assert radiation_coefficient("P", n, s, (0, 0, 1)) == pytest.approx(0.0)
    assert radiation_coefficient("S", n, s, (0, 0, 1)) == pytest.approx(1.0)


def test_radiation_45_degree_sensor_in_slip_normal_plane():
    n, s = (0, 0, 1), (1, 0, 0)
    g = (1, 0, 1)                                # not normalised on purpose
    pat = radiation_pattern(n, s, g)
    assert pat.p == pytest.approx(1.0)
    assert pat.s == pytest.approx(0.0, abs=1e-12)
    assert not pat.p_near_node
    # opposite quadrant: polarity flips, magnitude preserved
    assert radiation_pattern(n, s, (-1, 0, 1)).p == pytest.approx(-1.0)
    # sensor in the fault plane along the slip direction: n.g = 0 -> P node, S = |n| = 1
    pat = radiation_pattern(n, s, (1, 0, 0))
    assert pat.p == pytest.approx(0.0) and pat.s == pytest.approx(1.0)


def test_radiation_rms_over_focal_sphere_matches_constants():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(200000, 3))
    g /= np.linalg.norm(g, axis=1)[:, None]
    n, s = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    ng, sg = g @ n, g @ s
    rp = 2 * ng * sg
    rs = np.linalg.norm(ng[:, None] * s + sg[:, None] * n - (2 * ng * sg)[:, None] * g, axis=1)
    assert math.sqrt(np.mean(rp ** 2)) == pytest.approx(PHASE_CONSTANTS["P"]["radiation_rms"], rel=0.01)
    assert math.sqrt(np.mean(rs ** 2)) == pytest.approx(PHASE_CONSTANTS["S"]["radiation_rms"], rel=0.01)
    # consistency of the vectorised check with the scalar function
    pat = radiation_pattern(n, s, g[0])
    assert pat.p == pytest.approx(rp[0]) and pat.s == pytest.approx(rs[0])


def test_radiation_bad_inputs():
    with pytest.raises(ValueError):
        radiation_pattern((0, 0, 0), (1, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        radiation_pattern((0, 0, 1), (1, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        radiation_coefficient("Rayleigh", (0, 0, 1), (1, 0, 0), (0, 0, 1))


# --- free surface ----------------------------------------------------------

FREE_SURFACE_TABLE = (2.00, 1.99, 1.96, 1.92, 1.86, 1.79, 1.70, 1.60, 1.49, 1.38,
                      1.26, 1.14, 1.02, 0.90, 0.79, 0.67, 0.54, 0.35)


def test_free_surface_poisson_solid_table():
    vp, vs = math.sqrt(3.0), 1.0
    angles = np.arange(0, 90, 5)
    values = free_surface_amplification(angles, vp, vs)
    assert values.shape == angles.shape
    assert np.all(np.abs(values - np.array(FREE_SURFACE_TABLE)) < 0.03)
    assert free_surface_amplification(0.0, vp, vs) == pytest.approx(2.0)
    assert free_surface_amplification(60.0, vp, vs) == pytest.approx(1.0, abs=0.03)
    assert free_surface_amplification(90.0, vp, vs) == 0.0
    assert isinstance(free_surface_amplification(30.0, vp, vs), float)
    # scale invariance: only vp/vs matters
    assert free_surface_amplification(40.0, VP, VP / math.sqrt(3)) == pytest.approx(
        free_surface_amplification(40.0, vp, vs))


def test_free_surface_bad_inputs():
    with pytest.raises(ValueError):
        free_surface_amplification(95.0, VP, VS)
    with pytest.raises(ValueError):
        free_surface_amplification(-1.0, VP, VS)
    with pytest.raises(ValueError):
        free_surface_amplification(10.0, VS, VP)   # vs >= vp


# --- geometry ---------------------------------------------------------------

def test_geometry_sensor_on_fault_centre_normal():
    g = geometry(sensor_xyz_m=(0.0, 0.0, 0.05), source_xyz_m=(0.0, 0.0, 0.0),
                 sensor_normal=(0.0, 0.0, -1.0))
    assert isinstance(g, Geometry)
    assert g.distance_m == pytest.approx(0.05)
    assert g.ray == pytest.approx((0.0, 0.0, 1.0))
    assert g.incidence_deg == pytest.approx(0.0)    # normal sign does not matter
    assert g.as_dict()["ray"] == [0.0, 0.0, 1.0]


def test_geometry_oblique_and_without_normal():
    g = geometry((0.05, 0.0, 0.05), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert g.distance_m == pytest.approx(0.05 * math.sqrt(2))
    assert g.incidence_deg == pytest.approx(45.0)
    assert math.isnan(geometry((0.0, 0.0, 0.05), (0.0, 0.0, 0.0)).incidence_deg)
    with pytest.raises(ValueError):
        geometry((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        geometry((1.0, 2.0), (0.0, 0.0, 0.0))


# --- combined estimate ------------------------------------------------------

def test_source_parameters_p_phase_uses_vp_for_moment_and_vs_for_radius():
    omega0, fc, dist, rad = 3.0e-13, 40e3, 0.06, 0.9
    sp = source_parameters(omega0, fc, "P", RHO, VP, VS, dist, rad, free_surface_factor=1.0,
                           f_plateau_hz=30e3)
    assert isinstance(sp, SourceParameters) and sp.valid and sp.reason == ""
    m0 = 4 * math.pi * RHO * VP ** 3 * dist * omega0 / rad
    r = 0.32 * VS / fc
    assert sp.seismic_moment_nm == pytest.approx(m0)
    assert sp.mw == pytest.approx(moment_magnitude(m0))
    assert sp.source_radius_m == pytest.approx(r)
    assert sp.rupture_area_m2 == pytest.approx(math.pi * r ** 2)
    assert sp.stress_drop_pa == pytest.approx(7 * m0 / (16 * r ** 3))
    assert sp.kR == pytest.approx(2 * math.pi * 30e3 * dist / VP)
    assert sp.far_field_ok is True
    assert sp.constants["wave_speed_m_s"] == VP and sp.constants["k"] == 0.32
    assert sp.constants["radiation_coefficient"] == rad
    assert sp.warnings == []
    # JSON-safe
    json.dumps(sp.as_dict())


def test_source_parameters_s_phase():
    sp = source_parameters(3.0e-10, 40e3, "s", RHO, VP, VS, 0.06)      # lower-case accepted
    assert sp.phase == "S"
    assert sp.constants["wave_speed_m_s"] == VS and sp.constants["k"] == 0.372
    assert sp.constants["radiation_coefficient"] == pytest.approx(math.sqrt(2 / 5))
    assert sp.seismic_moment_nm == pytest.approx(
        4 * math.pi * RHO * VS ** 3 * 0.06 * 3.0e-10 / math.sqrt(2 / 5))
    assert sp.source_radius_m == pytest.approx(0.372 * VS / 40e3)
    assert math.isnan(sp.kR) and sp.far_field_ok is None


def test_source_parameters_round_trip_through_plateau():
    m0_true = 0.5
    omega0 = plateau_from_moment(m0_true, RHO, VP, 0.04, 0.7)
    sp = source_parameters(omega0, 30e3, "P", RHO, VP, VS, 0.04, 0.7)
    assert sp.seismic_moment_nm == pytest.approx(m0_true)


def test_source_parameters_warnings_fire():
    # near node, near field, enormous moment -> stress drop > 1 GPa and Mw > -1, vp/vs too small
    sp = source_parameters(1e-3, 200e3, "P", RHO, 1500.0, 1300.0, 0.02, 0.05, f_plateau_hz=1e3)
    assert sp.valid
    joined = " | ".join(sp.warnings)
    assert "node" in joined
    assert "kR" in joined and sp.far_field_ok is False
    assert "1 GPa" in joined
    assert "Mw" in joined
    assert "vp/vs" in joined
    # a benign case fires nothing
    ok = source_parameters(1e-13, 50e3, "P", RHO, VP, VS, 0.05, 0.8, f_plateau_hz=40e3)
    assert 1e3 < ok.stress_drop_pa < 1e8 and ok.mw < -5
    assert ok.warnings == [] and ok.far_field_ok is True


def test_source_parameters_degenerate_fit_is_invalid_not_exception():
    for omega0, fc in [(-1e-10, 40e3), (0.0, 40e3), (float("nan"), 40e3), (1e-10, 0.0),
                       (1e-10, float("inf")), ("x", 40e3)]:
        sp = source_parameters(omega0, fc, "P", RHO, VP, VS, 0.05, 0.8)
        assert not sp.valid and sp.reason
        assert math.isnan(sp.seismic_moment_nm) and math.isnan(sp.stress_drop_pa)
        assert sp.far_field_ok is None
        json.dumps(sp.as_dict())
    node = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.0)
    assert not node.valid and "node" in node.reason


def test_source_parameters_rejects_bad_material_and_phase():
    with pytest.raises(ValueError):
        source_parameters(1e-10, 40e3, "L", RHO, VP, VS, 0.05, 0.8)
    with pytest.raises(ValueError):
        source_parameters(1e-10, 40e3, "P", 0.0, VP, VS, 0.05, 0.8)
    with pytest.raises(ValueError):
        source_parameters(1e-10, 40e3, "P", RHO, VP, VS, -0.05, 0.8)
    with pytest.raises(ValueError):
        source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=0.0)
    with pytest.raises(ValueError):
        source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, k=-1.0)


def test_end_to_end_geometry_radiation_free_surface():
    # fault in the x-y plane at z=0, slip along x, sensor on the free surface at 45 deg
    n, s = (0, 0, 1), (1, 0, 0)
    sensor, source = (0.03, 0.0, 0.03), (0.0, 0.0, 0.0)
    geo = geometry(sensor, source, sensor_normal=(0, 0, 1))
    rad = radiation_coefficient("P", n, s, geo.ray)
    fsf = free_surface_amplification(geo.incidence_deg, VP, VS)
    assert rad == pytest.approx(1.0)
    assert geo.incidence_deg == pytest.approx(45.0)
    sp = source_parameters(1e-10, 60e3, "P", RHO, VP, VS, geo.distance_m, rad, fsf, f_plateau_hz=10e3)
    assert sp.valid
    assert sp.seismic_moment_nm == pytest.approx(
        seismic_moment(1e-10, RHO, VP, geo.distance_m, rad, fsf))
    assert isinstance(RadiationPattern(1.0, 0.0, False).as_dict(), dict)
