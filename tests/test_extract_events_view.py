import numpy as np
import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import ExtractEventsView


@pytest.fixture
def view(app):
    v = ExtractEventsView(app, 0)
    yield v
    v.on_close()


def test_defaults(app, view):
    truth = app.truth[0]
    assert view.title() == "Extract Events - run00"
    assert view.x_combo.get() == "time" and view.y_combo.get() == "shear_stress"
    assert view.start_var.get() == "-5" and view.end_var.get() == "+5"
    assert list(view.event_combo["values"]) == [str(i) for i in range(len(truth.event_indices))]
    assert view.event_combo.get() == "0" and view.indices == list(truth.event_indices)
    assert "time" not in view.y_combo["values"] and "index" in view.x_combo["values"]
    assert view.ax.get_xlabel() == "time" and len(view.ax.patches) == len(truth.event_indices)
    assert "picked events" in view.status_var.get()


def test_zoom_and_whole_run(app, view):
    t_event = app.truth[0].event_times[1]
    view.event_combo.set("1")
    view.on_event_selected()
    lo, hi = view.ax.get_xlim()
    assert lo == pytest.approx(t_event - 15.0, abs=0.01) and hi == pytest.approx(t_event + 15.0, abs=0.01)
    view.show_whole_run()
    lo, hi = view.ax.get_xlim()
    assert lo < app.truth[0].time[0] + 1 and hi > app.truth[0].time[-1] - 1


def test_extract_with_asymmetric_window(app, view):
    truth = app.truth[0]
    view.start_var.set("-2")
    view.end_var.set("3")
    n = view.extract(confirm=False)
    assert n == len(truth.event_indices)
    events = app.data_manager.get_data("runs/[0]/events")
    e = events[1]
    assert e["event_time"] == pytest.approx(truth.event_times[1])
    assert e["time"][0] == pytest.approx(truth.event_times[1] - 2.0, abs=2e-3)
    assert e["time"][-1] == pytest.approx(truth.event_times[1] + 3.0, abs=2e-3)
    assert e["shear_stress"].shape == e["time"].shape
    saved = app.data_manager.get_data("runs/[0]/event_window")
    assert saved == {"version": 1, "start_s": -2.0, "end_s": 3.0, "x_field": "time", "y_field": "shear_stress", "n_events": n}
    assert "4 events extracted" in view.status_var.get()
    # reopening restores the window
    v2 = ExtractEventsView(app, 0)
    assert v2.start_var.get() == "-2" and v2.end_var.get() == "+3"
    v2.on_close()


def test_extract_asks_before_replacing(app, view, monkeypatch):
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: False)
    before = app.data_manager.get_data("runs/[0]/events")
    assert view.extract() == 0
    assert app.data_manager.get_data("runs/[0]/events") is before


def test_bad_window_is_reported(app, view):
    view.start_var.set("abc")
    assert view.extract(confirm=False) == 0 and "numbers" in view.status_var.get()
    view.start_var.set("3")
    view.end_var.set("1")
    assert view.extract(confirm=False) == 0 and "before" in view.status_var.get()


def test_without_picks(app):
    run = app.data_manager.get_data("runs/[0]")
    del run["event_indices"]
    v = ExtractEventsView(app, 0)
    assert "pick events first" in v.status_var.get()
    assert v.extract_button.instate(["disabled"]) and v.extract(confirm=False) == 0
    v.on_close()


def test_registered_on_event_indices(app):
    ctx = app.context_at(app.find_item("runs/[0]/event_indices"))
    action = [a for a in actions_for(ctx.kind) if a.label == "Extract Events"][0]
    app.run_action(action, ctx)
    v = app.child_windows[-1]
    assert isinstance(v, ExtractEventsView) and v.run_idx == 0
    v.on_close()
