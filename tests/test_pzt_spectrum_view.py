"""Headless-Tk tests for PZTSpectrumView on synthetic strain data."""
import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.ui.views.pzt_spectrum_view import (
    PZTSpectrumView, RESULT_VERSION, TRIGGER_PICKED, TRIGGER_EVENT_TIME, TUKEY_ALPHA,
    channel_key, parse_channel_key,
)

EVENT = 1
BRUNE_CH = 3          # channel replaced by a Brune displacement pulse
TANH_CH = 0           # untouched tanh step (arrival = event_time)
OMEGA0_V = 1.0        # plateau of the synthetic pulse in V*s
FC = 10e3             # Hz
GAIN = 2.0            # V/m -> displacement plateau = OMEGA0_V / GAIN
N_WIN = 500           # samples in the default 0.5 + 2.0 ms window at 200 kHz
RHO, VP, VS, DIST = 2700.0, 6000.0, 3500.0, 0.1


@pytest.fixture
def brune_app(app_with_strain, tmp_path):
    """The strain app with event 1 / channel 3 replaced by a Brune pulse plus
    white noise 30 dB below the plateau, tiny noise on every other channel
    (so the SNR gate has a finite noise spectrum), and a V/m calibration CSV."""
    app = app_with_strain
    event = app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]")
    original = event["strain"]["original"]
    time = np.asarray(original["time"], dtype=float)
    raw = np.asarray(original["raw"], dtype=float)
    fs = sp.sampling_rate(time)
    rng = np.random.default_rng(7)
    raw = raw + rng.normal(0.0, 5e-5, raw.shape)
    t0 = float(event["event_time"]) - time[0]
    _, pulse = sp.brune_pulse(fs, time.size, OMEGA0_V, FC, t0)
    sigma = OMEGA0_V / math.sqrt(1000.0) * fs / math.sqrt(N_WIN)
    raw[BRUNE_CH] = pulse + rng.normal(0.0, sigma, time.size)
    original["raw"] = raw
    csv = tmp_path / "sensor.csv"
    f = np.geomspace(100.0, 100e3, 25)
    csv.write_text("frequency_hz,gain\n" + "\n".join(f"{fi:.6g},{GAIN}" for fi in f) + "\n")
    app.calibration_csv = str(csv)
    app.fs = fs
    return app


@pytest.fixture
def view(brune_app):
    v = PZTSpectrumView(brune_app, 0, EVENT)
    yield v
    if v.winfo_exists():
        v.on_close()


def calibrated_compute(view, channel=BRUNE_CH):
    view.set_channel(channel)
    view.calibration_path_var.set(view.app.calibration_csv)
    view.unit_combobox.set("V/m")
    assert view.compute(), view.status_var.get()
    return view.fit


def set_medium(view, phase="P", radiation=None):
    view.phase_combobox.set(phase)
    view.on_phase_selected()
    view.rho_var.set(str(RHO)); view.c_var.set(str(VP)); view.vs_var.set(str(VS))
    view.distance_var.set(str(DIST))
    if radiation is not None:
        view.radiation_var.set(str(radiation))


# ---------------------------------------------------------------------------
def test_open_defaults(brune_app, view):
    assert view.event_idx == EVENT
    assert view.title() == "PZT Spectrum - Event 1"
    assert view in brune_app.child_windows
    assert list(view.channel_combobox["values"]) == [str(i) for i in range(16)]
    assert view.current_channel() == 0
    assert view.trigger_combobox.get() == TRIGGER_EVENT_TIME
    assert (view.pre_var.get(), view.post_var.get()) == ("0.5", "2.0")
    assert float(view.fmax_var.get()) == pytest.approx(0.4 * brune_app.fs)
    assert float(view.fmin_var.get()) == 1e3
    assert view.unit_combobox.get() == "V/V" and view.calibration_path_var.get() == ""
    assert view.n_fixed_var.get() and view.snr_var.get() == "3"
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
    assert view.radiation_hint_var.get() == "RMS for P: 0.516"
    assert view.record is None and view.fit is None
    assert view.result_vars["omega0"].get() == "n/a"
    assert "no saved record" in view.status_var.get()


def test_compute_uncalibrated_brune_channel(view):
    view.set_channel(BRUNE_CH)
    assert view.compute(), view.status_var.get()
    fit = view.fit
    assert fit.valid, fit.reason
    assert fit.amp_units == "V*s"
    assert fit.omega0 == pytest.approx(OMEGA0_V, rel=0.05)
    assert fit.fc == pytest.approx(FC, rel=0.05)
    assert fit.n == 2.0 and fit.n_fixed and fit.converged and not fit.band_limited
    assert fit.warnings == ()
    assert view.window.n == N_WIN and view.window.noise_available
    assert view.spectrum.meta["transient_in_flat_region"] is True
    assert view.record["channel"] == BRUNE_CH and view.record["fit"]["valid"]
    assert view.record["parameter_warnings"] == []
    assert "V*s" in view.result_vars["omega0"].get()
    assert view.result_vars["fc"].get().endswith("Hz")
    assert "converged: yes" in view.result_vars["flags"].get()
    assert "band-limited: no" in view.result_vars["flags"].get()
    assert "WARNINGS" not in view.result_vars["flags"].get()
    assert "flat region: yes" in view.result_vars["taper"].get()
    assert view.result_vars["n"].get().startswith("2")
    assert "fit OK" in view.status_var.get()
    # the bins-used label mirrors the fit (all bins clear the 3x gate at 30 dB below the plateau)
    assert int(view.result_vars["n_bins"].get()) == fit.n_bins >= 8


def test_compute_calibrated_recovers_displacement_plateau(view):
    fit = calibrated_compute(view)
    assert fit.valid, fit.reason
    assert fit.amp_units == "m*s"
    assert fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert fit.fc == pytest.approx(FC, rel=0.05)
    assert view.spectrum.calibrated
    assert view.record["calibration_unit"] == "V/m"
    assert view.record["calibration_path"] == view.app.calibration_csv
    assert view.record["spectrum"]["meta"]["calibration"]["unit"] == "V/m"
    assert "m*s" in view.result_vars["omega0"].get()


def test_free_n_and_soft_l1(view):
    view.set_channel(BRUNE_CH)
    view.n_fixed_var.set(False)
    view.loss_combobox.set("soft_l1")
    assert view.compute()
    assert view.fit.valid and not view.fit.n_fixed
    assert view.fit.n == pytest.approx(2.0, abs=0.3)
    assert view.fit.fc == pytest.approx(FC, rel=0.10)
    assert view.record["loss"] == "soft_l1" and view.record["n_fixed"] is False
    assert "(free)" in view.result_vars["n"].get()


def test_remove_step_on_tanh_channel(view):
    view.set_channel(TANH_CH)
    view.remove_step_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.window.step_removed
    assert view.window.step_amplitude == pytest.approx(0.5, rel=0.05)
    assert view.fit.valid, view.fit.reason
    assert view.record["remove_step"] is True
    assert view.record["spectrum"]["meta"]["step_removed"] is True
    # the step is fully removed: the window ends near zero
    assert abs(np.mean(view.window.signal[-50:])) < 0.02


def test_picked_arrival_trigger(brune_app, view):
    event = view.event
    arrivals = np.asarray(event["strain_truth"]["arrival_times"], dtype=float)
    view.set_channel(TANH_CH)
    view.trigger_combobox.set(TRIGGER_PICKED)
    # no picks yet -> status, no dialog
    assert not view.compute()
    assert "picked arrival" in view.status_var.get()
    event["strain"]["original"]["rupture_arrival_time"] = arrivals
    assert view.compute(), view.status_var.get()
    time = np.asarray(event["strain"]["original"]["time"], dtype=float)
    assert view.window.i_trigger == int(np.argmin(np.abs(time - arrivals[TANH_CH])))
    assert view.record["trigger"] == TRIGGER_PICKED
    assert view.record["trigger_time_s"] == pytest.approx(arrivals[TANH_CH])


def test_invalid_entries_go_to_status_not_dialog(view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("abc")
    assert not view.compute()
    assert "pre window" in view.status_var.get()
    view.pre_var.set("0.5")
    view.fmax_var.set("500")           # below fmin
    assert not view.compute()
    assert "fmax" in view.status_var.get()
    view.fmax_var.set("80000")
    view.post_var.set("50")            # 50 ms: window leaves the 20 ms trace
    assert not view.compute()
    assert "not inside the trace" in view.status_var.get()
    view.post_var.set("2.0")
    view.baseline_combobox.set("highpass")
    view.highpass_var.set("")
    assert not view.compute()
    assert "highpass" in view.status_var.get()
    view.highpass_var.set("500")
    assert view.compute()
    assert view.record["baseline"] == "highpass" and view.record["highpass_hz"] == 500.0
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 500.0
    view.baseline_combobox.set("pre_linear")
    view.unit_combobox.set("V/m")      # unit without a CSV
    view.calibration_path_var.set("")
    assert not view.compute()
    assert "needs a CSV" in view.status_var.get()
    view.calibration_path_var.set("/nonexistent/sensor.csv")
    assert not view.compute()
    assert "calibration CSV" in view.status_var.get()
    assert view.record is not None  # last good record kept


def test_parameters_reach_the_analysis_calls(brune_app, view):
    """Every entry is applied, not merely stored in the record."""
    event = view.event
    arrivals = np.asarray(event["strain_truth"]["arrival_times"], dtype=float)
    event["strain"]["original"]["rupture_arrival_time"] = arrivals
    view.set_channel(BRUNE_CH)
    view.trigger_combobox.set(TRIGGER_PICKED)
    view.pre_var.set("0.6")
    view.post_var.set("1.8")
    view.taper_combobox.set("tukey")
    view.baseline_combobox.set("highpass")
    view.highpass_var.set("300")
    view.t_star_var.set("1e-6")
    view.fmin_var.set("2000")
    view.fmax_var.set("70000")
    view.bins_var.set("20")
    view.snr_var.set("2")
    view.n_fixed_var.set(False)
    view.loss_combobox.set("soft_l1")
    assert view.compute(), view.status_var.get()
    fs = brune_app.fs
    assert view.window.n_pre == round(0.6e-3 * fs) and view.window.n_post == round(1.8e-3 * fs)
    assert view.window.trigger_time == pytest.approx(arrivals[BRUNE_CH])
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 300.0
    assert view.window.baseline_region == "highpass"
    assert view.spectrum.t_star_s == 1e-6
    assert view.spectrum.meta["taper"] == "tukey" and view.spectrum.meta["alpha"] == TUKEY_ALPHA
    assert view.spectrum.meta["fmin"] == 2000.0 and view.spectrum.meta["fmax"] == 70000.0
    assert view.spectrum.meta["bins_per_decade"] == 20
    assert view.spectrum.meta["highpass_hz"] == 300.0
    assert view.fit.snr_min == 2.0 and view.fit.loss == "soft_l1" and not view.fit.n_fixed
    assert view.fit.band_used[0] >= 2000.0 and view.fit.band_used[1] <= 70000.0
    # the same calls with t* off give a different plateau (the correction was applied)
    view.t_star_var.set("")
    omega_t = view.fit.omega0
    assert view.compute()
    assert view.spectrum.t_star_s is None
    assert view.fit.omega0 != pytest.approx(omega_t, rel=1e-6)


def test_short_pre_window_warning_is_visible_and_blocks_source(view):
    """A pre window shorter than the tukey ramp attenuates Omega0; the view
    must say so (flags, status, taper row) and refuse source parameters."""
    view.set_channel(BRUNE_CH)
    view.calibration_path_var.set(view.app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.pre_var.set("0.1")
    view.post_var.set("2.5")
    assert view.compute(), view.status_var.get()
    assert view.fit.valid
    assert view.fit.omega0 < 0.5 * OMEGA0_V / GAIN               # attenuated well below the truth
    assert view.spectrum.meta["transient_in_flat_region"] is False
    assert any(w.startswith("transient not in the taper's flat region") for w in view.fit.warnings)
    flags = view.result_vars["flags"].get()
    assert "WARNINGS" in flags and "attenuated" in flags and "flat region" in flags
    assert "shorter than the tukey ramp" in flags                # the parameter warning too
    assert view.record["parameter_warnings"] and "tukey ramp" in view.record["parameter_warnings"][0]
    assert "flat region: no" in view.result_vars["taper"].get()
    status = view.status_var.get()
    assert "fit OK" not in status
    assert "warnings" in status and "attenuated" in status
    # compute source is refused for the attenuated plateau
    set_medium(view, "P", 0.52)
    assert not view.compute_source()
    assert "attenuated" in view.status_var.get()
    assert view.record["source"] is None
    # the parameter warning fires on its own from the entries
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.3, "post_ms": 2.5})
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.5, "post_ms": 2.0}) == []
    # a proper pre window: no warnings, source accepted
    view.pre_var.set("0.5")
    assert view.compute()
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["parameter_warnings"] == [] and view.fit.warnings == ()
    assert "fit OK" in view.status_var.get()
    assert "WARNINGS" not in view.result_vars["flags"].get()
    assert view.compute_source(), view.status_var.get()


def test_compute_source(view):
    fit = calibrated_compute(view)
    # uncalibrated / missing inputs are refused with a status message
    view.rho_var.set("")
    assert not view.compute_source()
    assert "rho" in view.status_var.get()
    set_medium(view, "P", 0.52)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "P"
    m0 = sd["seismic_moment_nm"]
    assert math.isfinite(m0) and m0 > 0 and math.isfinite(sd["mw"])
    assert m0 == pytest.approx(src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52), rel=1e-12)
    assert sd["constants"]["wave_speed_m_s"] == VP
    assert sd["mw"] == pytest.approx(src.moment_magnitude(m0))
    assert sd["source_radius_m"] == pytest.approx(0.32 * VS / fit.fc)
    assert sd["stress_drop_pa"] == pytest.approx(7 * m0 / (16 * sd["source_radius_m"] ** 3))
    assert sd["constants"]["f_plateau_hz"] == pytest.approx(fit.band_used[0])
    assert math.isfinite(sd["kR"])
    assert view.source_vars["m0"].get().endswith("N m")
    assert view.source_vars["mw"].get() != "n/a"
    assert view.source_vars["warnings"].get() != "n/a"
    # an uncalibrated spectrum is refused
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute()
    assert not view.compute_source()
    assert "uncalibrated" in view.status_var.get()


def test_free_surface_factor_reaches_the_moment(brune_app, view):
    """The typed free-surface factor divides M0, is stored in the constants and
    restored on reopen; an unusable value is a status message, not a dialog."""
    fit = calibrated_compute(view)
    set_medium(view, "P", 0.52)
    view.free_surface_var.set("1.7")
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["seismic_moment_nm"] == pytest.approx(
        src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52, 1.7), rel=1e-12)
    assert sd["seismic_moment_nm"] == pytest.approx(
        src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52) / 1.7, rel=1e-12)
    assert sd["constants"]["free_surface_factor"] == 1.7
    assert view.save()
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.free_surface_var.get() == "1.7"
        assert v2.record["source"]["constants"]["free_surface_factor"] == 1.7
        for bad in ("0", "-2", "nan", "abc", ""):
            v2.free_surface_var.set(bad)
            assert not v2.compute_source()
            assert "free-surface factor" in v2.status_var.get()
        v2.free_surface_var.set("1.0")
        assert v2.compute_source()
        assert v2.record["source"]["seismic_moment_nm"] == pytest.approx(
            src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52), rel=1e-12)
    finally:
        v2.on_close()


def test_compute_source_s_phase_uses_vs(view):
    fit = calibrated_compute(view)
    set_medium(view, "S", 0.6)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "S"
    assert sd["constants"]["wave_speed_m_s"] == VS and sd["constants"]["vp_m_s"] == VP
    assert sd["seismic_moment_nm"] == pytest.approx(src.seismic_moment(fit.omega0, RHO, VS, DIST, 0.6), rel=1e-12)
    assert sd["seismic_moment_nm"] != pytest.approx(src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.6), rel=1e-3)
    assert sd["source_radius_m"] == pytest.approx(0.372 * VS / fit.fc)
    assert sd["kR"] == pytest.approx(2 * math.pi * fit.band_used[0] * DIST / VS)


def test_compute_source_guards_malformed_fit(view):
    calibrated_compute(view)
    set_medium(view, "P")
    del view.record["fit"]["fc"]
    assert not view.compute_source()
    assert "fc" in view.status_var.get()
    view.record["fit"]["fc"] = {"bad": 1}
    assert not view.compute_source()
    assert view.record["source"]["valid"] is False       # the analysis reports the bad fc, no raise
    assert "invalid" in view.source_vars["warnings"].get()


def test_phase_switch_updates_default_radiation(view):
    view.phase_combobox.set("S")
    view.on_phase_selected()
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["S"]["radiation_rms"])
    assert view.radiation_hint_var.get() == "RMS for S: 0.632"
    view.phase_combobox.set("P")
    view.on_phase_selected()
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])


def test_typed_radiation_survives_phase_switch(brune_app, view):
    view.radiation_var.set("0.9")
    view.phase_combobox.set("S")
    view.on_phase_selected()
    assert view.radiation_var.get() == "0.9"
    assert view.radiation_hint_var.get() == "RMS for S: 0.632"
    view.phase_combobox.set("P")
    view.on_phase_selected()
    assert view.radiation_var.get() == "0.9"
    # a saved S-phase source at the S default, reopened: a fresh channel still follows the phase
    calibrated_compute(view)
    set_medium(view, "S", src.PHASE_CONSTANTS["S"]["radiation_rms"])
    assert view.radiation_is_default()
    assert view.compute_source() and view.save()
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.phase_combobox.get() == "S"
        v2.set_channel(5)
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert float(v2.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
        # ... but a restored typed value is kept across a phase switch
        v2.set_channel(BRUNE_CH)
        v2.radiation_var.set("0.45")
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert v2.radiation_var.get() == "0.45"
    finally:
        v2.on_close()


def test_save_merges_channels_and_is_json_like(brune_app, view):
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source()
    assert view.save()
    saved = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/pzt_spectrum")
    assert saved is view.event["pzt_spectrum"]
    assert saved["version"] == RESULT_VERSION
    assert set(saved["channels"]) == {channel_key(BRUNE_CH)} == {"ch3"}
    rec = saved["channels"]["ch3"]
    for key in ("version", "channel", "trigger", "trigger_time_s", "fs_hz", "pre_ms", "post_ms", "taper",
                "baseline", "highpass_hz", "remove_step", "calibration_path", "calibration_unit",
                "calibration_db", "t_star_s", "fmin_hz", "fmax_hz", "bins_per_decade", "snr_min",
                "n_fixed", "loss", "parameter_warnings", "spectrum", "fit", "source"):
        assert key in rec, key
    assert rec["channel"] == BRUNE_CH and rec["fit"]["valid"] and rec["source"]["valid"]
    assert rec["spectrum"]["amp_units"] == "m*s"
    assert isinstance(rec["fit"]["warnings"], list) and isinstance(rec["fit"]["band_used"], list)
    json.dumps(saved)  # JSON-like (NaN allowed)

    # a second channel is merged, the first kept
    view.set_channel(TANH_CH)
    assert view.record is None and view.result_vars["fc"].get() == "n/a"
    view.remove_step_var.set(True)
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute()
    assert view.save()
    saved = view.event["pzt_spectrum"]
    assert set(saved["channels"]) == {"ch3", "ch0"}
    assert saved["channels"]["ch3"]["source"]["valid"]
    assert saved["channels"]["ch0"]["source"] is None
    assert saved["channels"]["ch0"]["remove_step"] is True
    # save without anything computed is refused
    view.set_channel(7)
    assert not view.save()
    assert "nothing to save" in view.status_var.get()


def test_channel_keys_are_not_digit_strings_and_legacy_keys_are_read(brune_app, view):
    assert channel_key(3) == "ch3" and parse_channel_key("ch3") == 3
    assert parse_channel_key("3") == 3 and parse_channel_key("CH12") == 12
    assert parse_channel_key("chx") is None and parse_channel_key("source") is None
    calibrated_compute(view)
    rec = view.record
    # a legacy all-digit key is read and renamed on the next save
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION, "channels": {"5": dict(rec, channel=5)}}
    assert set(view.saved_channels()) == {5}
    view.set_channel(5)
    assert view.record is not None and "saved record" in view.status_var.get()
    view.set_channel(BRUNE_CH)
    assert view.compute() and view.save()
    channels = view.event["pzt_spectrum"]["channels"]
    assert set(channels) == {"ch3", "ch5"}
    assert not any(k.isdigit() for k in channels)


def test_saved_record_without_channel_field_can_be_saved(brune_app, view):
    """A partial / hand-edited record lacking ``channel`` opens as a saved
    record; Save files it under the channel it was found at instead of raising."""
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION, "channels": {"ch2": {"pre_ms": 1.0}}}
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == 2
        assert "showing saved record" in v2.status_var.get()
        assert v2.pre_var.get() == "1"
        assert v2.record["channel"] == 2
        assert v2.save()
        channels = v2.event["pzt_spectrum"]["channels"]
        assert set(channels) == {"ch2"} and channels["ch2"]["channel"] == 2
        assert "channel 2 saved" in v2.status_var.get()
        # a record with no channel and no selection is refused with a status line
        v2.record = {"pre_ms": 1.0}
        v2.channel_combobox.set("")
        assert not v2.save()
        assert "no channel" in v2.status_var.get()
    finally:
        v2.on_close()


def test_legacy_digit_keyed_channels_loaded_as_a_list_are_read(brune_app, view):
    """A legacy ``channels`` map keyed '0'..'n-1' comes back from an unmarked
    (older) HDF5 file as a LIST; its entries are channels 0..n-1 and a later
    Save keeps them (renamed) instead of silently dropping them."""
    fit = calibrated_compute(view)
    rec = view.record
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION,
                                  "channels": [dict(rec, channel=0), dict(rec, channel=1), "junk"]}
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert set(v2.saved_channels()) == {0, 1}
        assert v2.current_channel() == 0
        assert "showing saved record" in v2.status_var.get()
        assert v2.result_vars["fc"].get() == f"{fit.fc:.4g} Hz"
        v2.set_channel(1)
        assert "showing saved record" in v2.status_var.get()
        v2.set_channel(BRUNE_CH)
        v2.calibration_path_var.set(brune_app.calibration_csv)
        v2.unit_combobox.set("V/m")
        assert v2.compute() and v2.save()
        channels = v2.event["pzt_spectrum"]["channels"]
        assert isinstance(channels, dict)
        assert set(channels) == {"ch0", "ch1", "ch3"}          # the legacy records are kept and renamed
        assert channels["ch1"]["channel"] == 1
    finally:
        v2.on_close()


def test_restore_accepts_array_valued_fields(brune_app, view):
    """A loader that hands back numpy arrays for stored lists must not break
    restore (truth-testing an array raises)."""
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    fit_fc = view.fit.fc
    rec = view.event["pzt_spectrum"]["channels"]["ch3"]
    rec["fit"]["band_used"] = np.asarray(rec["fit"]["band_used"])
    rec["fit"]["warnings"] = np.asarray(["band_limited: synthetic"])
    for key in ("f_binned", "amp_binned", "noise_binned", "snr", "counts"):
        rec["spectrum"][key] = np.asarray(rec["spectrum"][key])
    rec["source"]["warnings"] = np.asarray(["kR synthetic warning"])
    rec["parameter_warnings"] = np.asarray([], dtype=object)
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        assert "showing saved record" in v2.status_var.get()
        assert v2.result_vars["fc"].get() == f"{fit_fc:.4g} Hz"
        assert v2.result_vars["band"].get().endswith("Hz")
        assert "band_limited: synthetic" in v2.result_vars["flags"].get()
        assert v2.source_vars["warnings"].get() == "kR synthetic warning"
        assert v2.source_vars["m0"].get().endswith("N m")
        set_medium(v2, "P")
        assert v2.compute_source(), v2.status_var.get()   # band_used array -> f_plateau
    finally:
        v2.on_close()


def test_malformed_saved_record_does_not_block_the_window(brune_app, view, monkeypatch):
    calibrated_compute(view)
    assert view.save()
    view.on_close()
    channels = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/pzt_spectrum")["channels"]
    good = channels["ch3"]
    # fit stored as a string, at_bounds not a dict, band garbage, source constants odd
    channels["ch3"] = dict(good, fit="garbage", source={"valid": True, "constants": "x", "warnings": 3})
    channels["ch4"] = dict(good, channel=4, pre_ms="abc",
                           fit=dict(good["fit"], at_bounds="oops", band_used="abc", omega0="?", warnings=None),
                           spectrum=dict(good["spectrum"], f_binned="nope", meta="meta"))
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        assert v2.result_vars["fc"].get() == "n/a"
        assert v2.pre_var.get() == "0.5"                 # parameters were still applied
        v2.set_channel(4)
        assert "saved record" in v2.status_var.get()
        assert v2.result_vars["omega0"].get() == "n/a" and v2.result_vars["band"].get() == "n/a"
        assert v2.pre_var.get() == "abc"
        assert not v2.compute() and "pre window" in v2.status_var.get()
        assert not v2.compute_source()
        # an exception while applying a record lands in the status line, not in the caller
        monkeypatch.setattr(PZTSpectrumView, "apply_record",
                            lambda self, rec: (_ for _ in ()).throw(KeyError("boom")))
        v2.set_channel(BRUNE_CH)
        assert "saved record unreadable" in v2.status_var.get() and "boom" in v2.status_var.get()
        assert v2.record is None and v2.result_vars["fc"].get() == "n/a"
        assert not v2.save()
    finally:
        v2.on_close()
    monkeypatch.undo()
    v3 = PZTSpectrumView(brune_app, 0, EVENT)   # opening again with the broken records works too
    v3.on_close()


def test_reopen_restores_parameters_and_numbers(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("0.4")
    view.post_var.set("1.5")
    view.calibration_path_var.set(brune_app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.taper_combobox.set("bh4")          # a bell taper: the pulse at 21% of the window is attenuated
    assert view.compute()
    assert "flat region: no" in view.result_vars["taper"].get()
    set_medium(view, "P", 0.6)
    assert not view.compute_source() and "attenuated" in view.status_var.get()
    view.taper_combobox.set("tukey")        # ramp 0.24 ms < pre 0.4 ms: flat region
    view.snr_var.set("2")
    view.bins_var.set("25")
    view.fmin_var.set("1500")
    view.t_star_var.set("1e-6")
    assert view.compute(), view.status_var.get()
    fit_saved = view.fit
    set_medium(view, "P", 0.6)
    assert view.compute_source()
    assert view.save()
    rec_saved = view.event["pzt_spectrum"]["channels"]["ch3"]
    view.on_close()

    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH          # opens on the saved channel
        assert v2.fit is None and v2.window is None      # nothing recomputed
        assert v2.record is not None and v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        assert (v2.pre_var.get(), v2.post_var.get()) == ("0.4", "1.5")
        assert v2.taper_combobox.get() == "tukey"
        assert v2.snr_var.get() == "2"
        assert float(v2.t_star_var.get()) == pytest.approx(1e-6)
        assert v2.calibration_path_var.get() == brune_app.calibration_csv
        assert v2.unit_combobox.get() == "V/m"
        assert (v2.rho_var.get(), v2.c_var.get(), v2.vs_var.get(), v2.distance_var.get()) == (
            "2700", "6000", "3500", "0.1")
        assert float(v2.radiation_var.get()) == 0.6
        assert v2.result_vars["fc"].get() == f"{fit_saved.fc:.4g} Hz"
        assert v2.source_vars["m0"].get().endswith("N m")
        assert "saved record" in v2.status_var.get()
        # every parameter the controls produce equals the saved record's
        p = v2.read_parameters()
        assert all(p[k] == rec_saved[k] for k in p), {k: (p[k], rec_saved[k]) for k in p if p[k] != rec_saved[k]}
        # switching to an unsaved channel clears, switching back restores
        v2.set_channel(5)
        assert v2.record is None and v2.result_vars["fc"].get() == "n/a"
        assert v2.pre_var.get() == "0.4"                 # parameters are left as they were
        v2.set_channel(BRUNE_CH)
        assert v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        # Compute on the restored channel recomputes with the restored parameters
        assert v2.compute()
        assert v2.fit.fc == pytest.approx(fit_saved.fc)
        assert v2.window.n_pre == round(0.4e-3 * brune_app.fs)
        assert v2.spectrum.t_star_s == 1e-6 and v2.fit.snr_min == 2.0
        assert v2.spectrum.meta["bins_per_decade"] == 25 and v2.spectrum.meta["fmin"] == 1500.0
        assert v2.spectrum.meta["taper"] == "tukey"
        # switching event keeps the channel and clears (no record on event 2)
        v2.set_event(2)
        assert v2.current_channel() == BRUNE_CH and v2.record is None
        assert v2.title() == "PZT Spectrum - Event 2"
    finally:
        v2.on_close()


def test_event_without_strain(app):
    v = PZTSpectrumView(app, 0, 1)
    try:
        assert list(v.channel_combobox["values"]) == []
        assert "no strain" in v.status_var.get()
        assert not v.compute()
    finally:
        v.on_close()


def test_close_unregisters(brune_app):
    v = PZTSpectrumView(brune_app, 0, EVENT)
    assert v in brune_app.child_windows
    v.on_close()
    assert v not in brune_app.child_windows


def test_db_calibration_flag_reaches_the_loader(view, tmp_path):
    csv = tmp_path / "sensor_db.csv"
    f = np.geomspace(100.0, 100e3, 25)
    db = 20.0 * math.log10(GAIN)
    csv.write_text("Freq,Response\n" + "\n".join(f"{fi:.6g},{db:.6f}" for fi in f) + "\n")
    view.set_channel(BRUNE_CH)
    view.calibration_path_var.set(str(csv))
    view.unit_combobox.set("V/m")
    view.calibration_db_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.fit.valid and view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["calibration_db"] is True
    # without the flag the 6 dB column is read as a linear gain of ~6
    view.calibration_db_var.set(False)
    assert view.compute()
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / db, rel=0.05)
