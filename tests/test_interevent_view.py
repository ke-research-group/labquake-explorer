import numpy as np
import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views.interevent_view import InterEventView, aligned_arrays


@pytest.fixture
def view(app):
    v = InterEventView(app, 0)
    yield v
    v.on_close()


def test_open_defaults_and_compute(app, view):
    truth = app.truth[0]
    assert view.title() == "Inter-event Metrics - run00"
    assert view.lp_combo.get() == "LP_displacement"
    assert view.slip_combo.get() == "displacement"
    r = view.result
    n = len(truth.event_times)
    assert len(r["event_times"]) == n
    assert np.isnan(r["recurrence"][0])
    assert r["recurrence"][1:] == pytest.approx([12.0] * (n - 1), abs=1e-9)
    assert r["lp_per_cycle"][1:] == pytest.approx([truth.lp_velocity * 12.0] * (n - 1), rel=1e-9)
    expected_slip = truth.slip + truth.creep_fraction * truth.lp_velocity * 12.0
    assert r["slip_per_cycle"][1:] == pytest.approx([expected_slip] * (n - 1), rel=1e-9)
    assert all(np.isnan(r["coseismic_slip"]))   # no event_analysis saved yet
    assert len(view.table.get_children()) == n
    assert view in app.child_windows


def test_creep_uses_saved_event_analysis(app):
    truth = app.truth[0]
    events = app.data_manager.get_data("runs/[0]/events")
    for event in events:
        event["event_analysis"] = {"version": 2, "displacement": truth.slip}
    v = InterEventView(app, 0)
    creep = v.result["creep"]
    assert np.isnan(creep[0])
    assert creep[1:] == pytest.approx([truth.creep_fraction * truth.lp_velocity * 12.0] * (len(creep) - 1), rel=1e-9)
    v.on_close()


def test_parameters_and_save(app, view):
    view.delay_var.set("0.1")
    view.width_var.set("0")
    view.compute()
    assert view.result["delay_s"] == 0.1 and view.result["width_s"] == 0.0
    view.save()
    saved = app.data_manager.get_data("runs/[0]/interevent")
    assert saved["version"] == 1
    assert saved["lp_field"] == "LP_displacement" and saved["delay_s"] == 0.1
    assert "saved" in view.status_var.get()
    # reopening restores the parameters
    v2 = InterEventView(app, 0)
    assert v2.delay_var.get() == "0.1" and v2.width_var.get() == "0"
    v2.on_close()


def test_bad_parameters_report_status(app, view):
    view.delay_var.set("abc")
    view.compute()
    assert "number" in view.status_var.get()


def test_registered_on_run_node(app):
    ctx = app.context_at(app.find_item("runs/[0]"))
    action = [a for a in actions_for(ctx.kind) if a.label == "Inter-event Metrics"][0]
    app.run_action(action, ctx)
    v = app.child_windows[-1]
    assert isinstance(v, InterEventView) and v.run_idx == 0
    v.on_close()
    assert v not in app.child_windows


def test_aligned_arrays_filters_by_length_and_kind():
    run = {"time": np.arange(5.0), "a": np.arange(5.0), "b": np.arange(4.0),
           "c": np.array([True] * 5), "events": [{}], "name": "x", "d": np.zeros((5, 2))}
    assert aligned_arrays(run) == ["a"]
