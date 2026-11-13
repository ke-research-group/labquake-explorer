"""Headless-Tk tests for the Auto windows box of the Slip Distribution view
(analysis/fit_windows.py; added on the slip-auto-windows branch)."""
import numpy as np
import pytest

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.analysis import fit_windows as fw
from labquake_explorer.ui.views import SlipDistributionView
from labquake_explorer.ui.views import slip_distribution_view as sdv
from tests.synthetic import SLIP_X_MM, add_slip_array

EVENT = 1


@pytest.fixture
def slip_app(app):
    """The synthetic experiment with an 8-sensor slip array on every event; the
    last sensor of every event slips little and creeps fast (its creep is close
    to its coseismic rate)."""
    run = app.data_manager.get_data("runs/[0]")
    run["event_extraction"].update({"start_s": -5.0, "end_s": 5.0})
    slips = np.maximum(0.0, 100.0 * (1.0 - ((np.asarray(SLIP_X_MM) - 279.0) / 300.0) ** 2))
    slips[-1] = 1.0
    app.slip_truth = [add_slip_array(event, slips_um=slips, creep_um_s=30.0, noise_um=0.05, seed=j)
                      for j, event in enumerate(run["events"])]
    app.refresh_tree()
    return app


@pytest.fixture
def view(slip_app):
    v = SlipDistributionView(slip_app, 0, EVENT)
    yield v
    if v.winfo_exists():
        v.on_close()


def test_apply_to_this_event_places_every_sensors_windows(slip_app, view):
    truth = slip_app.slip_truth[EVENT]
    assert view.same_windows_var.get() is True
    assert view.apply_auto_windows()
    assert view.same_windows_var.get() is False
    t_i = np.array([s.t_shift_s for s in view.result.sensors])
    after = np.array(view.fit_after)
    before = np.array(view.fit_before)
    assert np.all(after[:, 0] > t_i) and np.all(before[:, 1] < t_i)
    np.testing.assert_allclose(after[:, 1] - after[:, 0], fw.DEFAULT_WIDTH_S)
    slips = np.array([s.coseismic_slip for s in view.result.sensors])
    np.testing.assert_allclose(slips[:-1], truth["slips_um"][:-1], atol=0.3)
    # the creeping sensor is flagged, in the table and the status line
    assert fw.FLAG_CREEP in view.auto_info[-1].flags
    assert all(fw.FLAG_CREEP not in w.flags for w in view.auto_info[:-1])
    assert fw.FLAG_CREEP in view.table.item("7", "values")[-1]
    assert fw.FLAG_CREEP in view.status_var.get()


def test_the_record_keeps_how_the_windows_were_placed(slip_app, view):
    assert view.apply_auto_windows()
    view.auto_creep_var.set("20")
    assert view.save()                                    # the settings used, not the ones typed since
    auto = view.event["slip_analysis"][sdv.AUTO_KEY]
    assert auto["version"] == sdv.AUTO_VERSION and auto["applied"] == [True] * 8
    assert auto["settings"]["creep_ratio"] == pytest.approx(fw.DEFAULT_CREEP_RATIO)
    assert auto["settings"]["background_factor"] == pytest.approx(fw.DEFAULT_BACKGROUND_FACTOR)
    assert set(auto["kind"]) <= set(fw.KINDS) and fw.FLAG_CREEP in auto["flags"][-1]
    assert auto["creep_ratio"][-1] > fw.DEFAULT_CREEP_RATIO > auto["creep_ratio"][0]
    # a window placed by hand afterwards: that sensor is no longer an auto one
    view.set_sensor(2)
    before, after = view.fit_before[2], view.fit_after[2]
    view.set_windows(before, (after[0] + 0.05, after[1] + 0.05))
    assert view.auto_info[2] is None and view.auto_info[3] is not None
    assert view.record()[sdv.AUTO_KEY]["applied"][2] is False
    assert "differs from the saved record" in view.status_var.get()


def test_auto_windows_survive_an_hdf5_round_trip(slip_app, view, tmp_path, app_from_file):
    assert view.apply_auto_windows()
    assert view.save()
    path = tmp_path / "exp.h5"
    slip_app.data_manager.save_file(path)
    v2 = SlipDistributionView(app_from_file(path), 0, EVENT)
    try:
        assert [w is not None for w in v2.auto_info] == [True] * 8
        assert [w.kind for w in v2.auto_info] == [w.kind for w in view.auto_info]
        assert fw.FLAG_CREEP in v2.auto_info[-1].flags
        assert v2.auto_settings["background_factor"] == pytest.approx(fw.DEFAULT_BACKGROUND_FACTOR)
        assert "matches the saved record" in v2.status_var.get()
    finally:
        v2.on_close()


def test_apply_to_events_keeps_hand_placed_records_and_lists_what_to_check(slip_app, view):
    view.set_event(2)
    assert view.save()                                    # event 2: windows placed by hand (the defaults)
    view.set_event(EVENT)
    view.auto_events_var.set("0-3")
    out = view.apply_auto_to_events(confirm=False)
    assert out["done"] == [0, 1, 3] and out["kept"] == [2] and out["failed"] == []
    assert view.event_idx == EVENT                        # back where it started
    run = slip_app.data_manager.get_data("runs/[0]")
    assert run[sdv.RUN_KEY] == {"version": sdv.RUN_VERSION, "events": [0, 1, 2, 3]}
    for k in (0, 1, 3):
        assert sdv.AUTO_KEY in run["events"][k]["slip_analysis"]
    assert sdv.AUTO_KEY not in run["events"][2]["slip_analysis"]
    # the creeping sensor of every event done is listed; a click opens it
    assert [(k, i) for k, i, _ in out["check"]] == [(0, 7), (1, 7), (3, 7)]
    assert len(view.auto_check_table.get_children()) == 3
    view.auto_check_table.selection_set("3:7")
    view.on_auto_check_selected()
    assert view.event_idx == 3 and view.sensor == 7
    # not keeping: the hand-placed event is done too; a blank entry uses the run's list
    view.auto_events_var.set("")
    view.auto_keep_var.set(False)
    out = view.apply_auto_to_events(confirm=False)
    assert out["done"] == [0, 1, 2, 3] and out["kept"] == []


def test_the_run_is_named_by_its_name_and_the_list_scrolls_sideways(slip_app, view, monkeypatch):
    run = slip_app.data_manager.get_data("runs/[0]")
    run.pop("name", None)
    assert view.run_label() == "runs/[0]"
    run["name"] = "run7"
    assert view.run_label() == "run7 (runs/[0])"
    asked = []
    monkeypatch.setattr(sdv.messagebox, "askokcancel", lambda title, message, **k: asked.append(message) or False)
    view.auto_events_var.set("0")
    assert view.apply_auto_to_events() is None
    assert "1 event(s) of run7 (runs/[0])" in asked[0]
    # the list of sensors to check keeps its width; the long "why" is reached by scrolling
    table = view.auto_check_table
    assert table.column("why", "width") >= sdv.AUTO_WHY_WIDTH_PX
    assert str(table.cget("xscrollcommand"))
    view.update_idletasks()
    assert sdv.AUTO_CHECK_WIDTH_PX <= table.master.winfo_reqwidth() < table.column("why", "width")


def test_event_arrows_step_through_the_events(slip_app, view):
    n = view.n_events()
    assert view.event_idx == EVENT
    view.prev_event_button.invoke()
    assert view.event_idx == 0 and view.event_combobox.get() == "0"
    view.prev_event_button.invoke()                       # nothing before the first
    assert view.event_idx == 0
    view.next_event_button.invoke()
    view.next_event_button.invoke()
    assert view.event_idx == 2
    # an event without a slip array is skipped
    del slip_app.data_manager.get_data("runs/[0]/events/[3]")["slip"]
    view.step_event(+1)
    assert view.event_idx == (4 if n > 4 else 2)
    # with a list of chosen events, the arrows step through those
    slip_app.data_manager.get_data("runs/[0]")[sdv.RUN_KEY] = {"version": 1, "events": [0, 2]}
    view.set_event(1)
    view.step_event(+1)
    assert view.event_idx == 2
    view.step_event(+1)
    assert view.event_idx == 2
    view.step_event(-1)
    assert view.event_idx == 0


def test_apply_to_events_entries_and_refusals(slip_app, view, monkeypatch):
    n = view.n_events()
    view.auto_events_var.set("")
    assert view.auto_event_list() == (list(range(n)), False)
    view.auto_events_var.set("2, 0-1; 1")
    assert view.auto_event_list() == ([0, 1, 2], True)
    for bad in ("x", f"0-{n}", "1-"):
        view.auto_events_var.set(bad)
        with pytest.raises(ValueError):
            view.auto_event_list()
        assert view.apply_auto_to_events(confirm=False) is None
        assert "auto windows: events" in view.status_var.get()
    view.auto_events_var.set("0")
    monkeypatch.setattr(sdv.messagebox, "askokcancel", lambda *a, **k: False)
    assert view.apply_auto_to_events() is None            # cancelled: nothing saved
    assert "slip_analysis" not in slip_app.data_manager.get_data("runs/[0]/events/[0]")
    view.auto_width_var.set("0")
    assert not view.apply_auto_windows() and "width must be > 0" in view.status_var.get()
    view.auto_width_var.set("0.6")
    view.select_t_i_method(cs.METHOD_BURST)
    assert not view.apply_auto_windows() and "Burst" in view.status_var.get()
