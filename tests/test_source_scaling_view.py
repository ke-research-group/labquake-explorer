import numpy as np
import pytest

from labquake_explorer.analysis.scaling import fit_power_law
from labquake_explorer.analysis.source import moment_magnitude
from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.source_scaling_view import (
    ALL_CHANNELS, FLAG_AT_BOUND, FLAG_BAND_LIMITED, FLAG_NEAR_FIELD, FLAG_NOT_CONVERGED,
    RESULT_VERSION, Y_FC, Y_MECH_STRESS_DROP, Y_RADIUS, Y_SLIP, Y_STRESS_DROP,
    SourceScalingView, available_channels, channel_flags, combine_channels, fit_pairs,
    gather_records, pzt_channel_records,
)

A_FC = 1.0e4           # fc = A_FC * M0**(-1/3)
M0_VALUES = (1.0e-2, 1.0e-1, 1.0, 1.0e1)
STRESS_DROP_PA = 1.0e5
VS = 3000.0
K = 0.32
RHO = 2650.0
VP = 5500.0


def source_record(m0: float, fc_scale: float = 1.0, valid=True, phase: str = "P",
                  k: float = K, fc: float = None, band_limited: bool = False,
                  at_bound: bool = False, converged: bool = True, far_field_ok=True) -> dict:
    """One saved PZT channel record (fit + source) in the PZT spectrum view's layout."""
    if fc is None:
        fc = fc_scale * A_FC * m0 ** (-1.0 / 3.0)
    r = k * VS / fc
    source = {
        "phase": phase, "omega0_ms": 1e-12, "fc_hz": fc,
        "seismic_moment_nm": m0, "mw": moment_magnitude(m0),
        "source_radius_m": r, "rupture_area_m2": np.pi * r ** 2,
        "stress_drop_pa": STRESS_DROP_PA, "kR": 5.0 if far_field_ok else 1.0,
        "far_field_ok": far_field_ok,
        "constants": {"phase": phase, "rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS,
                      "wave_speed_m_s": VP if phase == "P" else VS, "distance_m": 0.1,
                      "k": k, "radiation_coefficient": 0.52, "radiation_rms": 0.516,
                      "free_surface_factor": 1.0, "f_plateau_hz": 1e3},
        "warnings": [], "valid": valid, "reason": "", "version": 1,
    }
    fit = {"omega0": 1e-12, "fc": fc, "n": 2.0, "n_fixed": True, "rms": 0.1, "n_bins": 40,
           "band_used": [1e3, 3e5], "converged": converged,
           "at_bounds": {"ln_omega0": False, "ln_fc": at_bound, "n": False},
           "band_limited": band_limited, "valid": True, "reason": "", "amp_units": "m*s",
           "warnings": [], "version": 1}
    return {"version": 1, "channel": 0, "fit": fit, "source": source}


def populate(app, channels=("0", "1"), n=None, mech=False):
    """Write exact power-law PZT records (and optional event_analysis) into run 0."""
    events = app.data_manager.get_data("runs/[0]/events")
    n = len(events) if n is None else n
    for j, event in enumerate(events[:n]):
        m0 = M0_VALUES[j % len(M0_VALUES)]
        event["pzt_spectrum"] = {
            "version": 1,
            "channels": {ch: source_record(m0, fc_scale=1.0 + 0.5 * i)
                         for i, ch in enumerate(channels)},
        }
        if mech:
            event["event_analysis"] = {
                "version": 2,
                "stress_drop": 0.4,                  # constant MPa
                "stress_drop_trend": 0.5,
                "displacement": 30.0 * m0 ** 0.5,    # slip ~ M0^0.5, um
                "displacement_trend": 60.0 * m0 ** 0.5,
            }
    return events


def synthetic_run(m0_values, fc_values, flagged_mask, channel="0") -> dict:
    """A plain run dict (no app) with one channel per event; ``flagged`` records are clipped fits."""
    events = []
    for m0, fc, flagged in zip(m0_values, fc_values, flagged_mask):
        rec = source_record(m0, fc=fc, band_limited=bool(flagged), at_bound=bool(flagged))
        events.append({"pzt_spectrum": {"version": 1, "channels": {channel: rec}}})
    return {"name": "synthetic", "events": events}


@pytest.fixture
def view(app):
    populate(app)
    v = SourceScalingView(app, 0)
    yield v
    if v in app.child_windows:
        v.on_close()


# ---------------------------------------------------------------- helpers
def test_gather_records_and_channels(app):
    events = populate(app, channels=("3", "12"))
    run = app.data_manager.get_data("runs/[0]")
    assert available_channels(run) == ["3", "12"]
    rows = gather_records(run, "3")
    assert [r["event"] for r in rows] == list(range(len(events)))
    assert rows[2]["seismic_moment_nm"] == pytest.approx(1.0)
    assert rows[2]["fc_hz"] == pytest.approx(A_FC)
    assert rows[2]["stress_drop_pa"] == pytest.approx(STRESS_DROP_PA)   # single channel: as recorded
    assert np.isnan(rows[2]["mech_stress_drop"])
    assert rows[2]["flags"] == [] and rows[2]["flagged"] is False
    assert rows[2]["phases"] == ["P"] and rows[2]["constants"][0]["k"] == K
    # 'all' combines in log space: fc scales 1.0 and 1.5 -> geometric median sqrt(1.5)
    rows_all = gather_records(run, ALL_CHANNELS)
    assert rows_all[2]["fc_hz"] == pytest.approx(np.sqrt(1.5) * A_FC)
    assert rows_all[2]["seismic_moment_nm"] == pytest.approx(1.0)
    assert rows_all[2]["n_channels"] == 2
    # an unknown channel gives NaN PZT fields but keeps the row
    rows_missing = gather_records(run, "99")
    assert len(rows_missing) == len(events) and np.isnan(rows_missing[0]["seismic_moment_nm"])


def test_all_channels_aggregate_is_eshelby_consistent(app):
    """The combined M0, r and stress drop obey 7 M0 / (16 r^3) and r = k Vs / fc."""
    populate(app, channels=("0", "1", "2"))
    run = app.data_manager.get_data("runs/[0]")
    events = run["events"]
    # three channels with different M0 and fc so the per-quantity medians would be inconsistent
    ch = events[2]["pzt_spectrum"]["channels"]
    for name, (m0, fc) in zip(("0", "1", "2"), ((1.0, 1e4), (2.0, 3e4), (0.5, 2e4))):
        ch[name] = source_record(m0, fc=fc)
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["seismic_moment_nm"] == pytest.approx(1.0)          # geometric median of 1, 2, 0.5
    assert row["fc_hz"] == pytest.approx(2e4)                       # median of 1e4, 3e4, 2e4
    assert row["source_radius_m"] == pytest.approx(K * VS / 2e4)
    assert row["stress_drop_pa"] == pytest.approx(7.0 * 1.0 / (16.0 * (K * VS / 2e4) ** 3))
    assert row["mw"] == pytest.approx(moment_magnitude(1.0))
    assert row["constants_consistent"] is True
    # the helper alone, with a channel lacking a radius: r and stress drop come from the others
    recs = list(pzt_channel_records(events[2]).values())
    recs[0]["source_radius_m"] = float("nan")
    combined = combine_channels(recs)
    assert combined["source_radius_m"] == pytest.approx(K * VS / 2e4)
    assert combine_channels([])["seismic_moment_nm"] != combine_channels([])["seismic_moment_nm"]  # NaN


def test_invalid_source_records_are_skipped(app):
    events = populate(app, channels=("0",))
    events[1]["pzt_spectrum"]["channels"]["0"]["source"]["valid"] = False
    events[2]["pzt_spectrum"]["channels"]["0"]["source"]["valid"] = np.bool_(False)
    run = app.data_manager.get_data("runs/[0]")
    rows = gather_records(run, "0")
    assert np.isnan(rows[1]["seismic_moment_nm"])
    assert np.isnan(rows[2]["seismic_moment_nm"])
    assert np.isfinite(rows[0]["seismic_moment_nm"])


def test_channel_flags():
    rec = source_record(1.0)
    assert channel_flags(rec, rec["source"]) == []
    rec = source_record(1.0, band_limited=True, at_bound=True, converged=False, far_field_ok=False)
    assert channel_flags(rec, rec["source"]) == [FLAG_BAND_LIMITED, FLAG_AT_BOUND,
                                                 FLAG_NOT_CONVERGED, FLAG_NEAR_FIELD]
    # numpy bools count; None (not reported) does not flag
    rec = source_record(1.0, far_field_ok=None)
    rec["fit"]["band_limited"] = np.bool_(True)
    assert channel_flags(rec, rec["source"]) == [FLAG_BAND_LIMITED]
    rec = source_record(1.0, far_field_ok=None)
    rec["fit"]["converged"] = None
    assert channel_flags(rec, rec["source"]) == []
    per_channel = pzt_channel_records({"pzt_spectrum": {"channels": {"5": rec}}})
    assert per_channel["5"]["far_field_ok"] is None and per_channel["5"]["flagged"] is False


def test_clipped_fc_records_are_gated_out_of_the_fit():
    """fc clipped at a fit bound is a bound, not a measurement: excluded by default."""
    m0 = np.logspace(-2, 1, 8)                       # 3 decades
    fc_true = 1e5 * m0 ** (-1.0 / 3.0)
    fc_clip = 1.2e5
    clipped = fc_true > fc_clip
    assert clipped.sum() == 5
    run = synthetic_run(m0, np.minimum(fc_true, fc_clip), clipped)
    assert fit_power_law(m0, fc_true).exponent == pytest.approx(-1.0 / 3.0, abs=1e-9)
    rows = gather_records(run, "0")
    assert [r["flagged"] for r in rows] == list(clipped)
    x, y, idx = fit_pairs(rows, "fc_hz")
    assert list(idx) == [5, 6, 7]
    assert fit_power_law(x, y).exponent == pytest.approx(-1.0 / 3.0, abs=1e-9)
    x_all, y_all, idx_all = fit_pairs(rows, "fc_hz", include_flagged=True)
    assert list(idx_all) == list(range(8))
    biased = fit_power_law(x_all, y_all).exponent
    assert abs(biased) < 0.25                        # the clipped fits drag the exponent toward 0
    # 'all' with the same single flagged channel: the row is flagged and still excluded
    rows_all = gather_records(run, ALL_CHANNELS)
    assert [r["flagged"] for r in rows_all] == list(clipped)
    assert list(fit_pairs(rows_all, "fc_hz")[2]) == [5, 6, 7]


def test_all_channels_prefers_unflagged_channels(app):
    populate(app, channels=("0", "1"))
    events = app.data_manager.get_data("runs/[0]/events")
    ch = events[2]["pzt_spectrum"]["channels"]
    ch["1"] = source_record(1.0, fc=1e3, band_limited=True)     # a clipped channel on event 2
    run = app.data_manager.get_data("runs/[0]")
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["n_channels"] == 1 and row["flagged"] is False
    assert row["fc_hz"] == pytest.approx(A_FC)                    # channel 0 only
    row_incl = gather_records(run, ALL_CHANNELS, include_flagged=True)[2]
    assert row_incl["n_channels"] == 2 and row_incl["flagged"] is True
    assert row_incl["fc_hz"] == pytest.approx(np.sqrt(A_FC * 1e3))
    # when every channel is flagged the row keeps the flagged values and is itself flagged
    ch["0"] = source_record(1.0, fc=2e3, at_bound=True)
    row_only_flagged = gather_records(run, ALL_CHANNELS)[2]
    assert row_only_flagged["flagged"] is True and row_only_flagged["n_channels"] == 2
    assert FLAG_AT_BOUND in row_only_flagged["flags"] and FLAG_BAND_LIMITED in row_only_flagged["flags"]


# ------------------------------------------------------------------- view
def test_registered_for_runs():
    labels = [a.label for a in actions_for(RUN)]
    assert "Source Scaling" in labels


def test_open_lists_channels_and_table(app, view):
    assert view.title() == "Source Scaling - run00"
    assert view in app.child_windows
    assert list(view.channel_combo["values"]) == [ALL_CHANNELS, "0", "1"]
    assert view.channel_combo.get() == ALL_CHANNELS
    assert view.y_combo.get() == Y_FC
    assert view.include_flagged_var.get() is False
    n_events = len(app.data_manager.get_data("runs/[0]/events"))
    assert len(view.table.get_children()) == n_events
    first = view.table.item("0")["values"]
    assert str(first[0]) == "0"
    assert float(first[1]) == pytest.approx(M0_VALUES[0])
    assert float(first[2]) == pytest.approx(moment_magnitude(M0_VALUES[0]), abs=1e-3)
    assert str(first[-1]) == ""                                     # flags column empty
    assert view.fit_line_artist is None
    assert view.flagged_scatter_artist is None
    assert "events have PZT source parameters" in view.status_var.get()
    assert "excluded" not in view.status_var.get()
    # reference line: exponent exactly -1/3 through the median (M0, fc) of the plotted pairs
    line = view.reference_line_artist
    assert line is not None and line.get_label() == "fc ~ M0^-1/3"
    xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    slope = np.polyfit(np.log10(xdata), np.log10(ydata), 1)[0]
    assert slope == pytest.approx(-1.0 / 3.0, abs=1e-9)
    x, y, _ = view.pairs()
    expected = np.median(y) * (xdata / np.median(x)) ** (-1.0 / 3.0)
    assert np.allclose(ydata, expected, rtol=1e-9)
    assert xdata.min() == pytest.approx(x.min()) and xdata.max() == pytest.approx(x.max())


def test_negative_m0_and_numpy_false_valid_are_excluded(app):
    events = populate(app, channels=("0",))
    events[1]["pzt_spectrum"]["channels"]["0"]["source"]["seismic_moment_nm"] = -1.0
    events[2]["pzt_spectrum"]["channels"]["0"]["source"]["valid"] = np.bool_(False)
    v = SourceScalingView(app, 0)
    try:
        v.set_channel("0")
        _, _, idx = v.pairs()
        assert list(idx) == [0, 3]
        fit = v.fit_power_law()
        assert not fit.valid and fit.n == 2 and v.fit_indices == []
        assert v.table.item("2")["values"][1] == "n/a"
    finally:
        v.on_close()


def test_fit_recovers_minus_one_third(app, view):
    view.set_channel("0")
    fit = view.fit_power_law()
    assert fit.valid
    assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert fit.exponent_stderr == pytest.approx(0.0, abs=1e-6)
    assert fit.exponent_rma == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    assert fit.n == len(app.data_manager.get_data("runs/[0]/events"))
    lo, hi = view.bootstrap
    assert lo == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert hi == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert view.fit_line_artist is not None
    text = view.fit_text.get()
    assert "OLS b = -0.3333" in text and "RMA b = -0.3333" in text
    assert "16-84" in text
    # 'all' (log-space median of two exact power laws with different prefactors) has the same exponent
    view.set_channel(ALL_CHANNELS)
    assert view.fit is None
    fit_all = view.fit_power_law()
    assert fit_all.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert fit_all.coefficient == pytest.approx(np.sqrt(1.5) * A_FC, rel=1e-6)


def test_constant_stress_drop_gives_zero_exponent(app, view):
    view.set_channel("0")
    view.set_y_variable(Y_STRESS_DROP)
    assert view.reference_line_artist is None
    fit = view.fit_power_law()
    assert fit.valid
    assert fit.exponent == pytest.approx(0.0, abs=1e-9)
    # radius r = k Vs / fc ~ M0^(+1/3)
    view.set_y_variable(Y_RADIUS)
    fit = view.fit_power_law()
    assert fit.exponent == pytest.approx(1.0 / 3.0, abs=1e-6)
    # 'all': the rederived stress drop 7 M0 / (16 r^3) is constant for an exact -1/3 law too
    view.set_channel(ALL_CHANNELS)
    view.set_y_variable(Y_STRESS_DROP)
    fit = view.fit_power_law()
    assert fit.valid and fit.exponent == pytest.approx(0.0, abs=1e-6)


def test_mechanical_variables(app):
    populate(app, mech=True)
    v = SourceScalingView(app, 0)
    try:
        v.set_y_variable(Y_MECH_STRESS_DROP)
        fit = v.fit_power_law()
        assert fit.valid and fit.exponent == pytest.approx(0.0, abs=1e-9)
        assert fit.coefficient == pytest.approx(0.4)
        v.trend_var.set(True)
        v.refresh()
        fit = v.fit_power_law()
        assert fit.coefficient == pytest.approx(0.5)
        v.set_y_variable(Y_SLIP)
        fit = v.fit_power_law()
        assert fit.exponent == pytest.approx(0.5, abs=1e-6)
        assert fit.coefficient == pytest.approx(60.0)
    finally:
        v.on_close()


def test_missing_channel_on_one_event_reduces_n(app):
    events = populate(app)
    del events[1]["pzt_spectrum"]["channels"]["1"]
    v = SourceScalingView(app, 0)
    try:
        v.set_channel("1")
        fit = v.fit_power_law()
        assert fit.valid and fit.n == len(events) - 1
        assert 1 not in v.fit_indices
        assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
    finally:
        v.on_close()


def test_quality_gating_in_view(app):
    """A clipped event is listed and drawn open, excluded from the fit, recorded on save."""
    events = populate(app, channels=("0",))
    clipped_fc = 2.0e4                                  # true fc of event 0 is 4.64e4
    events[0]["pzt_spectrum"]["channels"]["0"] = source_record(
        M0_VALUES[0], fc=clipped_fc, band_limited=True, at_bound=True)
    v = SourceScalingView(app, 0)
    try:
        v.set_channel("0")
        assert v.table.item("0")["values"][-1] == "band,bound"
        assert float(v.table.item("0")["values"][3]) == pytest.approx(clipped_fc)
        assert v.table.item("1")["values"][-1] == ""
        assert v.excluded == [{"event": 0, "flags": [FLAG_BAND_LIMITED, FLAG_AT_BOUND]}]
        assert "1 event(s) excluded by quality flags" in v.status_var.get()
        assert v.flagged_scatter_artist is not None
        assert len(v.flagged_scatter_artist.get_offsets()) == 1
        assert "excluded" in v.flagged_scatter_artist.get_label()
        assert len(v.scatter_artist.get_offsets()) == 3
        fit = v.fit_power_law()
        assert fit.valid and fit.n == 3 and v.fit_indices == [1, 2, 3]
        assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
        assert "1 event(s) excluded" in v.fit_text.get()
        saved = v.save()
        assert saved["include_flagged"] is False
        assert saved["n_excluded"] == 1
        assert saved["excluded"] == [{"event": 0, "flags": [FLAG_BAND_LIMITED, FLAG_AT_BOUND]}]
        assert saved["event_indices"] == [1, 2, 3]
        # override: the clipped record enters the fit and biases the exponent
        v.set_include_flagged(True)
        assert v.excluded == [] and "INCLUDED" in v.status_var.get()
        assert v.flagged_scatter_artist is not None and "in fit" in v.flagged_scatter_artist.get_label()
        assert len(v.scatter_artist.get_offsets()) == 4
        fit = v.fit_power_law()
        assert fit.n == 4 and v.fit_indices == [0, 1, 2, 3]
        assert fit.exponent > -1.0 / 3.0 + 0.05
        saved = v.save()
        assert saved["include_flagged"] is True and saved["n_excluded"] == 0
    finally:
        v.on_close()


def test_inconsistent_constants_are_reported(app):
    events = populate(app, channels=("0", "1"))
    for event in events:
        m0 = event["pzt_spectrum"]["channels"]["0"]["source"]["seismic_moment_nm"]
        event["pzt_spectrum"]["channels"]["1"] = source_record(5.0 * m0, phase="S", k=0.372)
    v = SourceScalingView(app, 0)
    try:
        v.set_channel("0")
        phases, constants, consistent = v.constants_summary()
        assert phases == ["P"] and len(constants) == 1 and consistent
        assert "WARNING" not in v.status_var.get()
        v.set_channel(ALL_CHANNELS)
        phases, constants, consistent = v.constants_summary()
        assert phases == ["P", "S"] and len(constants) == 2 and not consistent
        assert "WARNING: phase/constants differ" in v.status_var.get()
        assert all(r["constants_consistent"] is False for r in v.rows)
        v.fit_power_law()
        saved = v.save()
        assert saved["phases"] == ["P", "S"]
        assert saved["constants_consistent"] is False
        assert sorted(c["phase"] for c in saved["constants"]) == ["P", "S"]
        assert {c["k"] for c in saved["constants"]} == {K, 0.372}
    finally:
        v.on_close()


def test_no_records_opens_with_status(app):
    v = SourceScalingView(app, 0)
    try:
        assert v in app.child_windows
        assert list(v.channel_combo["values"]) == [ALL_CHANNELS]
        assert "no saved pzt results" in v.status_var.get().lower()
        n_events = len(app.data_manager.get_data("runs/[0]/events"))
        assert len(v.table.get_children()) == n_events
        assert v.table.item("0")["values"][1] == "n/a"
        fit = v.fit_power_law()
        assert not fit.valid
        assert "Fit failed" in v.fit_text.get()
        assert v.save() is None
        assert "source_scaling" not in app.data_manager.get_data("runs/[0]")
    finally:
        v.on_close()


def test_too_few_events_invalid_fit(app):
    populate(app, n=2)
    v = SourceScalingView(app, 0)
    try:
        fit = v.fit_power_law()
        assert not fit.valid and fit.n == 2
        assert np.isnan(v.bootstrap[0])
    finally:
        v.on_close()


def test_save_writes_run_result(app, view):
    view.set_channel("0")
    view.fit_power_law()
    saved = view.save()
    stored = app.data_manager.get_data("runs/[0]/source_scaling")
    assert stored is saved
    assert stored is view.run["source_scaling"]
    assert stored["version"] == RESULT_VERSION
    assert stored["channel"] == "0"
    assert stored["y_variable"] == Y_FC and stored["y_field"] == "fc_hz"
    assert stored["trend_corrected"] is False and stored["include_flagged"] is False
    assert stored["exponent"] == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert stored["exponent_stderr"] == pytest.approx(0.0, abs=1e-6)
    assert stored["exponent_rma"] == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert stored["bootstrap_16_84"] == pytest.approx([-1.0 / 3.0, -1.0 / 3.0], abs=1e-6)
    assert stored["n_boot"] == 1000 and stored["seed"] == 0
    assert stored["r2"] == pytest.approx(1.0)
    n_events = len(app.data_manager.get_data("runs/[0]/events"))
    assert stored["n"] == len(stored["event_indices"]) == n_events
    assert stored["event_indices"] == list(range(n_events))
    assert stored["n_excluded"] == 0 and stored["excluded"] == []
    assert stored["phases"] == ["P"] and stored["constants_consistent"] is True
    assert stored["constants"] == [{"phase": "P", "rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS, "k": K}]
    # the fit is reproducible from the saved dict alone
    x, y = np.asarray(stored["x"]), np.asarray(stored["y"])
    assert len(x) == len(y) == stored["n"]
    assert list(x) == pytest.approx(list(M0_VALUES))
    assert list(y) == pytest.approx([A_FC * m ** (-1.0 / 3.0) for m in M0_VALUES])
    line = 10.0 ** (stored["log10_coefficient"] + stored["exponent"] * np.log10(x))
    assert np.allclose(line, y, rtol=1e-6)
    assert np.allclose(stored["coefficient"] * x ** stored["exponent"], y, rtol=1e-6)
    refit = fit_power_law(x, y)
    assert refit.exponent == pytest.approx(stored["exponent"], abs=1e-12)
    assert refit.log10_coefficient == pytest.approx(stored["log10_coefficient"], abs=1e-12)
    assert "Saved" in view.status_var.get()
    assert all(isinstance(k, str) for k in stored)


def test_saved_result_is_restored_on_open(app):
    events = populate(app)
    v = SourceScalingView(app, 0)
    v.set_channel("1")
    v.set_y_variable(Y_RADIUS)
    v.trend_var.set(True)
    v.refresh()
    v.fit_power_law()
    saved = v.save()
    v.on_close()

    v2 = SourceScalingView(app, 0)
    try:
        assert v2.channel_combo.get() == "1"
        assert v2.y_combo.get() == Y_RADIUS
        assert v2.trend_var.get() is True
        assert v2.saved_result is saved
        assert v2.saved_agrees is True
        text = v2.fit_text.get()
        assert "Saved fit" in text and "agrees" in text and "0.3333" in text
        assert v2.fit is not None and v2.fit.valid                   # the fit was re-run on open
        assert v2.fit.exponent == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert v2.fit_line_artist is not None
    finally:
        v2.on_close()

    # a record changed since the save: the restored summary says so
    events[2]["pzt_spectrum"]["channels"]["1"]["source"]["seismic_moment_nm"] *= 3.0
    v3 = SourceScalingView(app, 0)
    try:
        assert v3.saved_agrees is False
        assert "DIFFERS" in v3.fit_text.get()
    finally:
        v3.on_close()

    # the saved channel no longer exists: fall back to 'all', still show the saved summary
    for event in events:
        del event["pzt_spectrum"]["channels"]["1"]
    v4 = SourceScalingView(app, 0)
    try:
        assert v4.channel_combo.get() == ALL_CHANNELS
        assert v4.y_combo.get() == Y_RADIUS
        assert "Saved fit" in v4.fit_text.get() and v4.saved_agrees is False
    finally:
        v4.on_close()


def test_no_saved_result_opens_without_summary(app, view):
    assert view.saved_result is None and view.saved_agrees is None
    assert view.fit_text.get() == "No fit yet"


def test_close_unregisters(app, view):
    assert view in app.child_windows
    view.on_close()
    assert view not in app.child_windows
