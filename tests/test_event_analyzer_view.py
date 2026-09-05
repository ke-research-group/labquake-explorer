import numpy as np
import pytest

from labquake_explorer.analysis.event_metrics import RESULT_VERSION
from labquake_explorer.ui.views import EventAnalyzerView


@pytest.fixture
def view(app):
    v = EventAnalyzerView(app, 0, 1)
    yield v
    v.on_close()


def idx_at(view, t_rel):
    return int(np.argmin(np.abs(view.data_t - view.event["event_time"] - t_rel)))


def test_open_and_defaults(app, view):
    truth = app.truth[0]
    assert view.event_idx == 1
    assert view.title() == "Event Analyzer - Event 1"
    assert view.data_x_combo.get() == "displacement"
    assert view.data_y_combo.get() == "shear_stress"
    assert view.fit_combo.get() == "OLS"
    assert len(view.picked_idx) == 8 and len(view.markers) == 8
    assert view in app.child_windows
    # default loading range lies inside the loading phase: slope = d(tau)/d(displacement)
    k = float(view.loading_slope_text.get())
    assert k == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert float(view.result_entries["loading_r2"].get()) == pytest.approx(1.0)
    assert float(view.stress_drop_text.get()) > 0


def test_moving_markers_recomputes_trend_drop(app, view):
    truth = app.truth[0]
    # rupture picks straddle the event; loading range before; post range after
    for point, t_rel in ((0, -4.0), (1, -1.0), (4, -0.001), (5, 0.001), (6, 0.5), (7, 2.5)):
        view.move_point(point, idx_at(view, t_rel))
    r = view.result
    assert r["stress_drop"] == pytest.approx(truth.stress_drop, abs=1e-3)
    assert r["displacement"] == pytest.approx(truth.slip, abs=3e-3)
    assert r["stress_drop_trend"] == pytest.approx(truth.stress_drop, abs=1e-6)
    assert r["displacement_trend"] == pytest.approx(truth.slip, abs=1e-6)
    assert view.result_entries["stress_drop_trend"].get() == f"{r['stress_drop_trend']:.6g}"


def test_fit_method_switch(app, view):
    view.fit_combo.set("Theil-Sen")
    view.update_analysis()
    assert view.result["fit_method"] == "theilsen"
    assert float(view.loading_slope_text.get()) == pytest.approx(app.truth[0].stiffness_fault, rel=1e-6)


def test_degenerate_picks_show_na(app, view):
    view.move_point(1, view.picked_idx[0])
    assert view.loading_slope_text.get() == "n/a"
    assert view.result_entries["loading_r2"].get() == "n/a"


def test_switch_event_reloads(app, view):
    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Event Analyzer - Event 2"
    assert view.event is app.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"


def test_save_writes_versioned_event_analysis(app, view):
    view.fit_combo.set("Theil-Sen")
    view.update_analysis()
    view.save_event()
    saved = app.data_manager.get_data("runs/[0]/events/[1]/event_analysis")
    assert saved is view.event["event_analysis"]
    assert saved["version"] == RESULT_VERSION
    assert saved["fit_method"] == "theilsen"
    assert saved["x_field"] == "displacement" and saved["y_field"] == "shear_stress"
    assert saved["loading_indices"] == view.picked_idx[:2]
    assert saved["post_indices"] == view.picked_idx[6:]
    assert saved["loading_window"][0] < saved["loading_window"][1] < 0
    # reopening restores picks and fit method
    view.fit_combo.set("OLS")
    view.picked_idx[4] = 3
    view.set_event(1)
    assert view.picked_idx == (saved["loading_indices"] + saved["unloading_indices"]
                               + [saved["rupture_start_index"], saved["rupture_end_index"]]
                               + saved["post_indices"])
    assert view.fit_combo.get() == "Theil-Sen"


def test_legacy_v1_results_load_with_default_post_range(app):
    event = app.data_manager.get_data("runs/[0]/events/[0]")
    n = len(event["time"])
    event["event_analysis"] = {
        "loading_indices": [10, 20], "unloading_indices": [30, 40],
        "rupture_start_index": 25, "rupture_end_index": 45,
        "loading_stiffness": 1.0, "unloading_stiffness": 1.0,
        "stress_drop": 0.1, "displacement": 1.0,
    }
    v = EventAnalyzerView(app, 0, 0)
    assert v.picked_idx[:6] == [10, 20, 30, 40, 25, 45]
    assert v.picked_idx[6:] == [int(n * 0.8), int(n * 0.9)]
    v.on_close()


def test_close_unregisters(app):
    v = EventAnalyzerView(app, 0, 0)
    assert v in app.child_windows
    v.on_close()
    assert v not in app.child_windows


def test_apply_to_all_events_uses_relative_windows(app, view):
    truth = app.truth[0]
    for point, t_rel in ((0, -4.0), (1, -1.0), (4, -0.001), (5, 0.001), (6, 0.5), (7, 2.5)):
        view.move_point(point, idx_at(view, t_rel))
    n = view.apply_to_all_events(confirm=False)
    events = app.data_manager.get_data("runs/[0]/events")
    assert n == len(events)
    for event in events:
        r = event["event_analysis"]
        assert r["version"] == RESULT_VERSION
        assert r["loading_window"] == pytest.approx([-4.0, -1.0], abs=2e-3)
        assert r["stress_drop_trend"] == pytest.approx(truth.stress_drop, abs=1e-6)
        assert r["displacement"] == pytest.approx(truth.slip, abs=3e-3)
    # the view is still on the same event, with the saved picks
    assert view.event_idx == 1
    assert view.picked_idx == (events[1]["event_analysis"]["loading_indices"]
                               + events[1]["event_analysis"]["unloading_indices"]
                               + [events[1]["event_analysis"]["rupture_start_index"],
                                  events[1]["event_analysis"]["rupture_end_index"]]
                               + events[1]["event_analysis"]["post_indices"])


def test_apply_to_all_events_cancelled(app, view, monkeypatch):
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: False)
    assert view.apply_to_all_events() == 0
    assert "event_analysis" not in app.data_manager.get_data("runs/[0]/events/[0]")
