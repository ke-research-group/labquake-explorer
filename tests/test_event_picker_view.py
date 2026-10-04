import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import EventPickerView


def open_from(app, path, label="Pick Events"):
    item = app.find_item(path)
    app.data_tree.selection_set(item)
    ctx = app.context_at(item)
    action = [a for a in actions_for(ctx.kind) if a.label == label][0]
    app.run_action(action, ctx)
    return app.child_windows[-1]


@pytest.fixture
def view(app):
    v = EventPickerView(app, 0)
    yield v
    v.on_close()


def test_defaults(app, view):
    truth = app.truth[0]
    n = len(truth.event_indices)
    assert view.title() == "Pick Events - run00"
    assert view.x_combo.get() == "time" and view.y_combo.get() == "shear_stress"
    assert view.start_var.get() == "-5" and view.end_var.get() == "5"
    assert view.picks == sorted(int(i) for i in truth.event_indices)
    assert list(view.event_combo["values"]) == [str(i) for i in range(n)]
    assert view.event_combo.get() == "0"
    assert "time" not in view.y_combo["values"] and "index" in view.x_combo["values"]
    assert view.ax.get_xlabel() == "time"
    assert len(view.spans) == n == len(view.picker.markers)
    assert not view.extract_button.instate(["disabled"])
    assert view.status_var.get().startswith(f"{n} picks")


def test_opened_from_array_run_and_event_indices(app):
    v = open_from(app, "runs/[0]/friction")
    assert isinstance(v, EventPickerView) and v.y_combo.get() == "friction"
    v.on_close()
    for path in ("runs/[0]", "runs/[0]/event_indices"):
        v = open_from(app, path)
        assert isinstance(v, EventPickerView) and v.run_idx == 0 and v.y_combo.get() == "shear_stress"
        v.on_close()


def test_picks_stay_sorted_and_follow_edits(app, view):
    truth = app.truth[0]
    saved = list(truth.event_indices)
    first = view.picks[0]
    view.picker.add_point(5)                            # earlier than every existing pick
    assert view.picks[0] == 5 and view.event_combo.get() == "0"
    assert len(view.spans) == len(view.picks) == len(view.picker.markers)
    assert "unsaved" in view.status_var.get()
    view.picker.remove_point(0)
    assert view.picks[0] == first and "unsaved" not in view.status_var.get()
    # a drag ends with a "move" notification carrying the new sample
    view.picker.picks[-1] = 1
    view.picker.on_release(None)                        # nothing moved: no notification
    assert view.event_combo.get() == "0"
    view.on_picks_changed("move", 1)
    assert view.picks[0] == 1 and view.event_combo.get() == "0"
    assert app.data_manager.get_data("runs/[0]/event_indices") == saved


def test_zoom_survives_a_replot(app, view):
    t_event = app.truth[0].event_times[1]
    view.event_combo.set("1")
    view.on_event_selected()
    lo, hi = view.ax.get_xlim()
    assert lo == pytest.approx(t_event - 15.0, abs=0.01) and hi == pytest.approx(t_event + 15.0, abs=0.01)
    view.picker.add_point(view.picks[1] + 3)            # replots: the zoom must stay
    assert view.ax.get_xlim() == (lo, hi)
    view.x_combo.set("index")
    view.plot()                                         # a new X axis autoscales again
    assert view.ax.get_xlim() != (lo, hi)
    view.show_whole_run()
    lo, hi = view.ax.get_xlim()
    assert lo < 1 and hi > len(app.truth[0].time) - 2


def test_spinbox_ticks_redraw_the_windows(app, view):
    t = app.truth[0].time
    idx = view.picks[0]
    view.start_entry.event_generate("<<Increment>>")
    view.end_entry.event_generate("<<Decrement>>")
    assert view.start_var.get() == "-4.5" and view.end_var.get() == "4.5"
    assert view.window_s() == (-4.5, 4.5)
    span = view.spans[0]
    assert span.get_x() == pytest.approx(t[idx] - 4.5, abs=2e-3)
    assert span.get_x() + span.get_width() == pytest.approx(t[idx] + 4.5, abs=2e-3)


def test_prev_next_buttons_step_the_event(app, view):
    n = len(view.picks)
    view.prev_button.invoke()
    assert view.event_combo.get() == "0"                # clamped at the first event
    view.next_button.invoke()
    assert view.event_combo.get() == "1"
    lo, hi = view.ax.get_xlim()
    assert lo < app.truth[0].event_times[1] < hi and hi - lo == pytest.approx(30.0, abs=0.05)
    for _ in range(n + 2):
        view.next_button.invoke()
    assert view.event_combo.get() == str(n - 1)         # clamped at the last event


def test_extract_saves_picks_and_uses_the_window(app, view):
    truth = app.truth[0]
    view.start_var.set("-2")
    view.end_var.set("3")
    view.picker.add_point(5)
    n = view.extract(confirm=False)
    assert n == len(truth.event_indices) + 1
    assert app.data_manager.get_data("runs/[0]/event_indices") == view.picks
    events = app.data_manager.get_data("runs/[0]/events")
    assert len(events) == n
    e = events[2]                                       # the second original event
    assert e["event_time"] == pytest.approx(truth.event_times[1])
    assert e["time"][0] == pytest.approx(truth.event_times[1] - 2.0, abs=2e-3)
    assert e["time"][-1] == pytest.approx(truth.event_times[1] + 3.0, abs=2e-3)
    assert e["shear_stress"].shape == e["time"].shape
    saved = app.data_manager.get_data("runs/[0]/event_window")
    assert saved == {"version": 1, "start_s": -2.0, "end_s": 3.0, "x_field": "time",
                     "y_field": "shear_stress", "n_events": n}
    assert f"{n} events extracted" in view.status_var.get()
    v2 = EventPickerView(app, 0)                        # reopening restores the window
    assert v2.start_var.get() == "-2" and v2.end_var.get() == "3"
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
    v = EventPickerView(app, 0)
    assert v.picks == [] and v.event_combo.get() == ""
    assert v.extract_button.instate(["disabled"]) and v.extract(confirm=False) == 0
    v.picker.add_point(100)
    assert not v.extract_button.instate(["disabled"]) and v.event_combo.get() == "0"
    v.save_picks()
    assert app.data_manager.get_data("runs/[0]/event_indices") == [100]
    v.on_close()


def test_nearest_index_from_the_curve(app, view):
    t = app.truth[0].time
    y = app.truth[0].shear_stress
    k = 1234
    assert view.picker.nearest_index(t[k], y[k]) == k
