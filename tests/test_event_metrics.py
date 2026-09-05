import numpy as np
import pytest

from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, trend_drop, RESULT_VERSION,
)
from tests.synthetic import make_stick_slip_run, extract_events


@pytest.fixture
def event_and_truth():
    ssr = make_stick_slip_run(recurrence=12.0, post_slope=0.02, fs=1000.0)
    run = ssr.as_dict()
    events = extract_events(run, 5.0)
    return events[1], ssr


def idx_at(event, t_rel):
    return int(np.argmin(np.abs(event["time"] - event["event_time"] - t_rel)))


def test_trend_drop_recovers_step_with_different_slopes():
    t = np.linspace(-5, 5, 10001)
    pre_slope, post_slope, drop = 0.1, -0.03, 0.4
    y = np.where(t < 0, 1.0 + pre_slope * t, 1.0 - drop + post_slope * t)
    td = trend_drop(t, y, pre=(1000, 3000), post=(7000, 9000), t_ref=0.0)
    assert td.valid
    assert td.drop == pytest.approx(drop, abs=1e-9)
    assert td.pre.slope == pytest.approx(pre_slope)
    assert td.post.slope == pytest.approx(post_slope)
    assert td.pre_at_ref == pytest.approx(1.0)
    assert td.post_at_ref == pytest.approx(1.0 - drop)


def test_trend_drop_reference_time_shifts_result():
    t = np.linspace(-5, 5, 10001)
    y = np.where(t < 0, 0.1 * t, -0.4 - 0.03 * t)
    at0 = trend_drop(t, y, (1000, 3000), (7000, 9000), t_ref=0.0).drop
    at1 = trend_drop(t, y, (1000, 3000), (7000, 9000), t_ref=1.0).drop
    assert at1 - at0 == pytest.approx(0.1 * 1.0 - (-0.03) * 1.0)


def test_trend_drop_invalid_when_window_degenerate():
    t = np.linspace(-5, 5, 101)
    y = t.copy()
    td = trend_drop(t, y, (10, 10), (70, 90))
    assert not td.valid and np.isnan(td.drop)


def test_analyze_event_against_synthetic_truth(event_and_truth):
    event, truth = event_and_truth
    n = len(event["time"])
    # loading range well before the event, post range after the post_slope segment (3 s) ended
    picks = EventPicks(
        loading=(idx_at(event, -4.0), idx_at(event, -1.0)),
        unloading=(idx_at(event, -0.002), idx_at(event, 0.002)),
        rupture=(idx_at(event, -0.001), idx_at(event, 0.001)),
        post=(idx_at(event, 0.5), idx_at(event, 2.5)),
    )
    res = analyze_event(event["time"], event["displacement"], event["shear_stress"],
                        event["event_time"], picks, x_field="displacement", y_field="shear_stress")
    assert res["version"] == RESULT_VERSION
    assert res["loading_stiffness"] == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert res["loading_fit"]["valid"] and res["loading_fit"]["r2"] == pytest.approx(1.0)
    # the drop at the pick is the imposed drop, loading over 2 ms is negligible
    assert res["stress_drop"] == pytest.approx(truth.stress_drop, abs=2 * truth.loading_rate * 0.002)
    assert res["displacement"] == pytest.approx(truth.slip, abs=1e-6 + 2 * truth.creep_fraction * truth.lp_velocity * 0.002)
    # trend-extrapolated drop removes the post-event relaxation trend and interseismic loading
    assert res["stress_drop_trend"] == pytest.approx(truth.stress_drop, abs=1e-6)
    assert res["displacement_trend"] == pytest.approx(truth.slip, abs=1e-6)
    assert res["pre_trend"]["slope"] == pytest.approx(truth.loading_rate, rel=1e-6)
    assert res["post_trend"]["slope"] == pytest.approx(truth.loading_rate + truth.post_slope, rel=1e-6)
    # windows are stored relative to event_time
    assert res["loading_window"] == pytest.approx([-4.0, -1.0], abs=1e-3)
    assert res["post_window"] == pytest.approx([0.5, 2.5], abs=1e-3)
    assert res["rupture_window"][0] < 0 < res["rupture_window"][1]
    assert res["loading_indices"] == list(picks.loading)
    assert res["post_indices"] == list(picks.post)
    assert "positive" in res["sign_convention"]


def test_analyze_event_signs(event_and_truth):
    event, truth = event_and_truth
    picks = EventPicks(loading=(100, 200), unloading=(300, 400),
                       rupture=(idx_at(event, 0.001), idx_at(event, -0.001)))
    res = analyze_event(event["time"], event["displacement"], event["shear_stress"],
                        event["event_time"], picks)
    # swapped rupture picks flip both signs
    assert res["stress_drop"] < 0 and res["displacement"] < 0
    assert res["post_indices"] is None and np.isnan(res["stress_drop_trend"])


def test_analyze_event_never_raises_on_degenerate_picks(event_and_truth):
    event, truth = event_and_truth
    picks = EventPicks(loading=(5, 5), unloading=(5, 5), rupture=(5, 5), post=(6, 6))
    res = analyze_event(event["time"], event["displacement"], event["shear_stress"],
                        event["event_time"], picks)
    assert np.isnan(res["loading_stiffness"]) and not res["loading_fit"]["valid"]
    assert res["stress_drop"] == 0.0


def test_picks_defaults_and_clipping():
    p = EventPicks.defaults(1000)
    assert p.to_list() == [250, 350, 500, 600, 400, 700, 800, 900]
    assert EventPicks.defaults(1000, with_post=False).post is None
    assert EventPicks.from_list([1, 2, 3, 4, 5, 6]).post is None
    assert EventPicks.from_list(range(8)).post == (6, 7)
    with pytest.raises(ValueError):
        EventPicks.from_list([1, 2, 3])
    clipped = EventPicks.from_list([0, 5000, 2, 3, 4, 5, 6, 7]).clipped(100)
    assert clipped.loading == (0, 99)


def test_picks_from_saved_results_v1_and_v2():
    v1 = {"loading_indices": [1, 2], "unloading_indices": [3, 4],
          "rupture_start_index": 5, "rupture_end_index": 6, "stress_drop": 0.1}
    p = picks_from_result(v1, 1000)
    assert p.loading == (1, 2) and p.rupture == (5, 6)
    assert p.post == EventPicks.defaults(1000).post
    v2 = dict(v1, post_indices=[7, 8], version=2)
    assert picks_from_result(v2, 1000).post == (7, 8)
    assert picks_from_result(v2, 5) is None       # out of range
    assert picks_from_result({"x": 1}, 10) is None
    assert picks_from_result(None, 10) is None
