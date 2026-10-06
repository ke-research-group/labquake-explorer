import numpy as np
import pytest

from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, picks_from_windows, windows_from_picks, RESULT_VERSION,
)
from tests.synthetic import make_stick_slip_run, extract_events

RESULT_KEYS = {"version", "x_field", "y_field", "loading_indices", "unloading_indices", "delta_indices",
               "loading_slope", "unloading_slope", "delta_x", "delta_y"}


@pytest.fixture
def event_and_truth():
    ssr = make_stick_slip_run(recurrence=12.0, post_slope=0.02, fs=1000.0)
    run = ssr.as_dict()
    events = extract_events(run, 5.0)
    return events[1], ssr


def idx_at(event, t_rel):
    return int(np.argmin(np.abs(event["time"] - event["event_time"] - t_rel)))


def test_analyze_event_against_synthetic_truth(event_and_truth):
    event, truth = event_and_truth
    picks = EventPicks(loading=(idx_at(event, -4.0), idx_at(event, -1.0)),
                       unloading=(idx_at(event, -0.002), idx_at(event, 0.002)),
                       delta=(idx_at(event, -0.001), idx_at(event, 0.001)))
    res = analyze_event(event["displacement"], event["shear_stress"], picks,
                        x_field="displacement", y_field="shear_stress")
    assert set(res) == RESULT_KEYS and res["version"] == RESULT_VERSION == 3
    assert res["x_field"] == "displacement" and res["y_field"] == "shear_stress"
    assert res["loading_slope"] == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert np.isfinite(res["unloading_slope"]) and res["unloading_slope"] < 0     # stress falls while slip grows
    # the differences across the picks are the imposed drop and slip; loading over 2 ms is negligible
    assert -res["delta_y"] == pytest.approx(truth.stress_drop, abs=2 * truth.loading_rate * 0.002)
    assert res["delta_x"] == pytest.approx(truth.slip, abs=1e-6 + 2 * truth.creep_fraction * truth.lp_velocity * 0.002)
    assert res["loading_indices"] == list(picks.loading) and res["delta_indices"] == list(picks.delta)
    assert all(isinstance(v, (int, float, str, list)) for v in res.values())


def test_analyze_event_signs(event_and_truth):
    event, truth = event_and_truth
    picks = EventPicks(loading=(100, 200), unloading=(300, 400),
                       delta=(idx_at(event, 0.001), idx_at(event, -0.001)))
    res = analyze_event(event["displacement"], event["shear_stress"], picks)
    assert res["delta_y"] > 0 and res["delta_x"] < 0          # swapped picks flip both differences


def test_analyze_event_never_raises_on_degenerate_picks(event_and_truth):
    event, truth = event_and_truth
    res = analyze_event(event["displacement"], event["shear_stress"], EventPicks((5, 5), (5, 5), (5, 5)))
    assert np.isnan(res["loading_slope"]) and np.isnan(res["unloading_slope"])
    assert res["delta_x"] == 0.0 and res["delta_y"] == 0.0
    with pytest.raises(ValueError):
        analyze_event([], [], EventPicks((0, 0), (0, 0), (0, 0)))
    with pytest.raises(ValueError):
        analyze_event([1.0, 2.0], [1.0], EventPicks((0, 0), (0, 0), (0, 0)))


def test_picks_defaults_and_clipping():
    p = EventPicks.defaults(1000)
    assert p.to_list() == [250, 350, 500, 600, 400, 700]
    assert EventPicks.from_list(range(8)).to_list() == [0, 1, 2, 3, 4, 5]     # a legacy post range is dropped
    with pytest.raises(ValueError):
        EventPicks.from_list([1, 2, 3])
    clipped = EventPicks.from_list([0, 5000, 2, 3, 4, 5]).clipped(100)
    assert clipped.loading == (0, 99) and clipped.pairs()["delta"] == (4, 5)


def test_picks_from_saved_results_v1_v2_v3():
    v1 = {"loading_indices": [1, 2], "unloading_indices": [3, 4],
          "rupture_start_index": 5, "rupture_end_index": 6, "stress_drop": 0.1}
    p = picks_from_result(v1, 1000)
    assert p.loading == (1, 2) and p.delta == (5, 6)
    v2 = dict(v1, post_indices=[7, 8], version=2)
    assert picks_from_result(v2, 1000).to_list() == [1, 2, 3, 4, 5, 6]
    v3 = {"version": 3, "loading_indices": [1, 2], "unloading_indices": [3, 4], "delta_indices": [5, 6]}
    assert picks_from_result(v3, 1000).delta == (5, 6)
    assert picks_from_result(v3, 5) is None       # out of range
    assert picks_from_result({"x": 1}, 10) is None
    assert picks_from_result(None, 10) is None


def test_windows_round_trip(event_and_truth):
    event, truth = event_and_truth
    t_rel = event["time"] - event["event_time"]
    picks = EventPicks((idx_at(event, -4.0), idx_at(event, -1.0)), (idx_at(event, -0.002), idx_at(event, 0.002)),
                       (idx_at(event, -0.001), idx_at(event, 0.001)))
    windows = windows_from_picks(t_rel, picks)
    assert windows["loading"] == pytest.approx((-4.0, -1.0), abs=1e-3)
    assert windows["delta"][0] < 0 < windows["delta"][1]
    assert picks_from_windows(t_rel, windows) == picks
    assert picks_from_windows(t_rel[::2], windows).loading[0] == pytest.approx(picks.loading[0] / 2, abs=1)
    with pytest.raises(ValueError):
        picks_from_windows([], windows)
    with pytest.raises(ValueError):
        windows_from_picks([], picks)
