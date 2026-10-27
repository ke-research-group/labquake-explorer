"""Headless-Tk tests for the Slip Distribution view (coseismic slip per sensor)."""
import math

import numpy as np
import pytest

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import SlipDistributionView
from labquake_explorer.ui.views.slip_distribution_view import slip_fields
from tests.synthetic import add_slip_array

EVENT = 1


@pytest.fixture
def slip_app(app):
    """The synthetic experiment with an 8-sensor slip array on every event and
    an extraction window of +-5 s recorded for the run."""
    run = app.data_manager.get_data("runs/[0]")
    run["event_extraction"].update({"start_s": -5.0, "end_s": 5.0})
    truths = []
    for j, event in enumerate(run["events"]):
        truths.append(add_slip_array(event, noise_um=0.05, seed=j))
    app.slip_truth = truths
    app.refresh_tree()
    return app


@pytest.fixture
def view(slip_app):
    v = SlipDistributionView(slip_app, 0, EVENT)
    yield v
    if v.winfo_exists():
        v.on_close()


def test_menu_and_defaults(slip_app, view):
    truth = slip_app.slip_truth[EVENT]
    item = slip_app.find_item(f"runs/[0]/events/[{EVENT}]")
    ctx = slip_app.context_at(item)
    assert "Slip Distribution" in [a.label for a in actions_for(ctx.kind)]
    assert view.title() == f"Slip Distribution - Event {EVENT}"
    assert view.field == "slip" and list(view.field_combobox["values"]) == ["slip"]
    assert len(view.channels) == 8 and view.x_mm[0] == 31.0
    assert (view.start_s, view.end_s) == (-5.0, 5.0)
    assert all(m == cs.METHOD_MAX_RATE for m in view.methods)
    assert view.fit_before[0] == (-3.75, -1.25) and view.fit_after[0] == (1.25, 3.75)
    assert view.same_windows_var.get() is True
    assert (view.reference_var.get(), view.smoothing_var.get(), view.search_var.get(), view.snr_var.get()) == \
        ("500", "2.5", "50", "5")
    slips = [s.coseismic_slip for s in view.result.sensors]
    np.testing.assert_allclose(slips, truth["slips_um"], atol=0.2)
    assert len(view.table.get_children()) == 8
    assert "t_i from" in view.table["columns"] and "t_i - event (ms)" in view.table["columns"]
    assert view.method_combobox.get() == "max slip rate"
    assert "not saved" in view.status_var.get() and "reference at +" in view.status_var.get()
    assert set(view.handles) == {"t_i", "before_start", "before_end", "after_start", "after_end"}
    # the rate axes sit behind the trace axes (which get the mouse) and keep their label on the right
    assert view.ax_rate.get_zorder() < view.ax_trace.get_zorder() and not view.ax_trace.patch.get_visible()
    assert view.ax_rate.yaxis.get_label_position() == "right"


def test_default_windows_follow_the_extraction_window(slip_app):
    run = slip_app.data_manager.get_data("runs/[0]")
    run["event_extraction"].update({"start_s": -2.0, "end_s": 6.0})
    v = SlipDistributionView(slip_app, 0, EVENT)
    try:
        assert (v.start_s, v.end_s) == (-2.0, 6.0)
        assert v.fit_before[0] == (-1.5, -0.5) and v.fit_after[0] == (1.5, 4.5)
    finally:
        v.on_close()
    del run["event_extraction"]["start_s"], run["event_extraction"]["end_s"]
    v = SlipDistributionView(slip_app, 0, EVENT)               # no window recorded: the event's time axis
    try:
        t_rel = v.time - v.t_event
        dt = cs.sampling_interval(v.time)
        assert v.start_s == pytest.approx(t_rel[0]) and v.end_s == pytest.approx(t_rel[-1] + dt)
        assert v.fit_before[0] == pytest.approx((0.75 * v.start_s, 0.25 * v.start_s))
        assert v.fit_after[0] == pytest.approx((0.25 * v.end_s, 0.75 * v.end_s))
    finally:
        v.on_close()


def test_event_without_slip_is_refused(app):
    assert slip_fields(app.data_manager.get_data("runs/[0]/events/[0]")) == []
    with pytest.raises(ValueError, match="no slip channel array"):
        SlipDistributionView(app, 0, 0)


def test_switching_to_an_event_without_slip_is_refused(slip_app, view, no_dialogs):
    del slip_app.data_manager.get_data("runs/[0]/events/[2]")["slip"]
    view.set_event(2)
    assert view.event_idx == EVENT and any("No slip data" in str(c) for c in no_dialogs)


def test_a_failed_switch_keeps_nothing_of_the_previous_event(slip_app, view):
    view.smoothing_var.set("-1")                          # typed, never applied
    view.set_event(2)
    assert view.event_idx == 2 and view.result is None
    assert view.table.get_children() == () and not view.ax_trace.lines
    assert "smoothing" in view.status_var.get()
    assert not view.save() and "slip_analysis" not in view.event
    view.smoothing_var.set("2.5")                         # the same text as before still recomputes
    view.schedule(view.on_settings_edited, now=True)
    assert view.result is not None and view.result.event_time == view.t_event
    assert view.save() and "slip_analysis" in view.event


def test_sensor_navigation_and_table(slip_app, view):
    view.step_sensor(+1)
    assert view.sensor == 1 and view.table.selection() == ("1",)
    view.step_sensor(-5)
    assert view.sensor == 0
    view.table.selection_set("6")
    view.on_table_select()
    assert view.sensor == 6 and view.sensor_combobox.get().startswith("7: slip_7")
    assert view.ax_trace.get_title().find("slip_7") >= 0


def test_typing_event_time_and_windows(slip_app, view):
    truth = slip_app.slip_truth[EVENT]
    view.set_sensor(4)
    view.t_shift_var.set(f"{(truth['t_i'][4] - view.t_event) * 1e3 + 5.0:.3f}")
    view.schedule(view.on_t_shift_edited, now=True)
    s = view.result.sensors[4]
    assert s.method == cs.METHOD_MANUAL and view.method_combobox.get() == "manual"
    assert s.t_shift_s == pytest.approx(truth["t_i"][4] - view.t_event + 0.005, abs=1e-9)
    # a window edit applies to every sensor while "same windows" is on
    view.before_start_var.set("-2")
    view.before_end_var.set("-0.5")
    view.schedule(view.on_window_edited, now=True)
    assert all(w == (-2.0, -0.5) for w in view.fit_before)
    # ... and only to the selected sensor when it is off
    view.same_windows_var.set(False)
    view.after_end_var.set("2")
    view.schedule(view.on_window_edited, now=True)
    assert view.fit_after[4] == (1.25, 2.0) and view.fit_after[0] == (1.25, 3.75)
    # turning it back on copies the selected sensor's windows to all
    view.same_windows_var.set(True)
    view.on_same_windows_toggled()
    assert all(w == (1.25, 2.0) for w in view.fit_after)
    # bad values are reported, nothing changes
    view.before_end_var.set("0.5")
    view.schedule(view.on_window_edited, now=True)
    assert "before window must end" in view.status_var.get() and view.fit_before[0] == (-2.0, -0.5)
    view.before_end_var.set("abc")
    view.schedule(view.on_window_edited, now=True)
    assert "enter a number" in view.status_var.get()


def test_unchanged_entry_text_is_not_an_edit(slip_app, view):
    view.set_sensor(3)
    pick = view.result.sensors[3].t_shift_s
    view.schedule(view.on_t_shift_edited, now=True)       # the rounded text read back, e.g. on focus-out
    view.schedule(view.on_window_edited, now=True)
    assert view.methods[3] == cs.METHOD_MAX_RATE and view.result.sensors[3].t_shift_s == pick


def test_a_typed_value_is_applied_before_save_and_navigation(slip_app, view):
    view.set_sensor(2)
    view.t_shift_var.set("12.5")
    view.schedule(view.on_t_shift_edited)                 # still waiting for the keys to settle
    assert view.methods[2] == cs.METHOD_MAX_RATE
    assert view.save()
    rec = view.event["slip_analysis"]
    assert rec["time_method"][2] == "manual" and rec["t_shift_s"][2] == pytest.approx(0.0125)
    view.t_shift_var.set("13")
    view.schedule(view.on_t_shift_edited)
    view.step_sensor(+1)                                  # goes to sensor 2, then the view moves on
    assert view.sensor == 3 and view.t_shift[2] == pytest.approx(0.013) and view.methods[3] == cs.METHOD_MAX_RATE
    view.before_end_var.set("abc")                        # a bad value in an entry is never saved past
    assert not view.save() and "not saved" in view.status_var.get()
    assert view.event["slip_analysis"]["t_shift_s"][2] == pytest.approx(0.0125)


def test_closing_cancels_a_pending_recompute(slip_app, view):
    view.t_shift_var.set("5")
    view.schedule(view.on_t_shift_edited)
    after_id = view._compute_after_id
    assert after_id is not None
    view.on_close()
    assert after_id not in view.tk.splitlist(view.tk.call("after", "info"))


def test_method_switch_and_repick(slip_app, view):
    view.set_sensor(2)
    pick = view.result.sensors[2].t_shift_s
    view.method_combobox.set("manual")
    view.on_method_selected()
    assert view.methods[2] == cs.METHOD_MANUAL and view.result.sensors[2].t_shift_s == pytest.approx(pick)
    view.repick_all()
    assert all(m == cs.METHOD_MAX_RATE for m in view.methods)
    view.search_var.set("10")
    view.schedule(view.on_settings_edited, now=True)
    assert view.result.search_halfwidth_s == pytest.approx(0.010)
    view.smoothing_var.set("-1")
    view.schedule(view.on_settings_edited, now=True)
    assert "smoothing" in view.status_var.get()


def test_dragging_event_time_and_windows(slip_app, view):
    view.set_sensor(3)
    s = view.result.sensors[3]
    ti = s.t_shift_s
    view.begin_drag("t_i", ti)
    state = view.drag_to(ti + 0.010)
    assert state["t_i"] == pytest.approx(ti + 0.010)
    assert float(view.t_shift_var.get()) == pytest.approx((ti + 0.010) * 1e3, abs=1e-3)
    assert view.handles["before_end"].get_xdata()[0] == pytest.approx(ti + 0.010 - 1.25)   # windows follow t_i
    view.end_drag()
    assert view.methods[3] == cs.METHOD_MANUAL and view.result.sensors[3].t_shift_s == pytest.approx(ti + 0.010)
    # a window edge cannot cross t_i, the body slides as one
    ti = view.result.sensors[3].t_shift_s
    view.begin_drag("before_end", ti - 1.25)
    state = view.drag_to(ti + 0.5)
    assert state["before"][1] == 0.0
    view.end_drag()
    assert all(w[1] == 0.0 for w in view.fit_before)
    view.begin_drag("after", ti + 2.0)
    state = view.drag_to(ti + 2.0 - 5.0)
    assert state["after"][0] == 0.0 and state["after"][1] == pytest.approx(3.75 - 1.25)
    view.end_drag()
    assert view.fit_after[0] == (0.0, pytest.approx(2.5))
    # a click without a move changes nothing
    before = list(view.fit_before)
    view.begin_drag("before_start", -3.0)
    view.end_drag()
    assert view.fit_before == before


def test_mouse_events_grab_handles_and_bodies(slip_app, view):
    from matplotlib.backend_bases import MouseEvent
    view.set_sensor(4)
    view.canvas.draw()
    ax = view.ax_trace
    y_mid = float(np.mean(ax.get_ylim()))

    def mouse(name, x, dx_px=0.0):
        px, py = ax.transData.transform((x, y_mid))
        return MouseEvent(name, view.canvas, px + dx_px, py, button=1)

    ti = view.result.sensors[4].t_shift_s
    press = mouse("button_press_event", ti, dx_px=4)       # a few pixels off still grabs t_i
    assert press.inaxes is ax                              # not the rate axes on top of it
    view.on_press(press)
    assert view._drag["what"] == "t_i"
    view.on_motion(mouse("motion_notify_event", ti + 0.5))
    view.on_release(mouse("button_release_event", ti + 0.5))
    assert view._drag is None and view.methods[4] == cs.METHOD_MANUAL
    assert view.t_shift[4] == pytest.approx(ti + 0.5, abs=0.02)
    ti = view.t_shift[4]
    view.on_press(mouse("button_press_event", ti + 1.25, dx_px=-3))
    assert view._drag["what"] == "after_start"
    view.on_release(mouse("button_release_event", ti + 1.25))
    view.on_press(mouse("button_press_event", ti - 2.0))
    assert view._drag["what"] == "before"
    view.on_release(mouse("button_release_event", ti - 2.0))
    view.on_press(mouse("button_press_event", ti + 0.6))   # between the windows: nothing
    assert view._drag is None
    view.toolbar.zoom()
    view.on_press(mouse("button_press_event", ti))
    assert view._drag is None
    view.toolbar.zoom()


# ------------------------------------------------------- real Tk mouse events
def shown(view):
    view.deiconify()
    view.update()
    view.canvas.draw()
    view.update()
    if not view.canvas.get_tk_widget().winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    return view.canvas.get_tk_widget()


def tk_xy(view, xd, yd):
    px, py = view.ax_trace.transData.transform((xd, yd))
    return int(round(px)), int(round(view.canvas.figure.bbox.height - py))


def tk_drag(widget, x0, y0, x1, y1):
    widget.event_generate("<Button-1>", x=x0, y=y0)
    widget.update()
    for f in (0.3, 0.6, 1.0):
        widget.event_generate("<Motion>", x=int(x0 + f * (x1 - x0)), y=int(y0 + f * (y1 - y0)), state=0x100)
        widget.update()
    widget.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    widget.update()


def test_real_mouse_drags_event_time_edges_and_windows(slip_app, view):
    view.set_sensor(4)
    widget = shown(view)
    y = float(np.mean(view.ax_trace.get_ylim()))
    ti = view.result.sensors[4].t_shift_s
    view.after_end_var.set("3")                          # typed, then straight to the mouse
    view.schedule(view.on_window_edited)
    view.entries["after end"].focus_force()
    view.update()
    tk_drag(widget, *tk_xy(view, ti, y), *tk_xy(view, ti + 0.5, y))
    assert view.methods[4] == cs.METHOD_MANUAL and view.t_shift[4] == pytest.approx(ti + 0.5, abs=0.05)
    assert all(w == (1.25, 3.0) for w in view.fit_after)
    ti = view.t_shift[4]
    tk_drag(widget, *tk_xy(view, ti - 3.75, y), *tk_xy(view, ti - 3.0, y))
    assert all(w[0] == pytest.approx(-3.0, abs=0.05) and w[1] == -1.25 for w in view.fit_before)
    tk_drag(widget, *tk_xy(view, ti + 2.0, y), *tk_xy(view, ti + 1.5, y))
    assert all(w == (pytest.approx(0.75, abs=0.05), pytest.approx(2.5, abs=0.05)) for w in view.fit_after)
    assert view.methods[4] == cs.METHOD_MANUAL and view.t_shift[4] == ti


def test_save_and_reopen_restore_everything(slip_app, view):
    view.set_sensor(5)
    view.set_event_time(5, view.result.sensors[5].t_shift_s + 0.003)
    view.same_windows_var.set(False)
    view.set_windows((-3.0, -1.0), (1.0, 3.0))
    view.snr_var.set("3")
    view.schedule(view.on_settings_edited, now=True)
    assert view.save()
    rec = slip_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/slip_analysis")
    assert rec is view.event["slip_analysis"]
    assert rec["version"] == 1 and rec["field"] == "slip" and rec["unit"] == "um" and rec["same_windows"] is False
    assert rec["time_method"][5] == "manual" and rec["fit_before_s"][5] == [-3.0, -1.0]
    assert rec["fit_before_s"][0] == [-3.75, -1.25] and rec["rate_snr"] == 3.0
    assert "matches the saved record" in view.status_var.get()
    view.on_close()
    v2 = SlipDistributionView(slip_app, 0, EVENT)
    try:
        assert v2.methods[5] == "manual" and v2.t_shift[5] == pytest.approx(rec["t_shift_s"][5])
        assert v2.fit_before[5] == (-3.0, -1.0) and v2.same_windows_var.get() is False and v2.snr_var.get() == "3"
        assert "matches the saved record" in v2.status_var.get()
        v2.repick_all()
        assert "differs from the saved record" in v2.status_var.get()
    finally:
        v2.on_close()


def test_save_survives_an_hdf5_round_trip(slip_app, view, tmp_path, app_from_file):
    view.set_event_time(0, 0.031)
    assert view.save()
    path = tmp_path / "exp.h5"
    slip_app.data_manager.save_file(path)
    app2 = app_from_file(path)
    v2 = SlipDistributionView(app2, 0, EVENT)
    try:
        assert v2.methods[0] == "manual" and v2.t_shift[0] == pytest.approx(0.031)
        assert "matches the saved record" in v2.status_var.get()
        slips = [s.coseismic_slip for s in v2.result.sensors]
        np.testing.assert_allclose(slips, view.record()["coseismic_slip_um"])
    finally:
        v2.on_close()


def test_zoom_buttons(slip_app, view):
    view.set_sensor(4)
    view.zoom_to_event_time()
    lo, hi = view.ax_trace.get_xlim()
    ti = view.result.sensors[4].t_shift_s
    assert lo == pytest.approx(ti - 0.2) and hi == pytest.approx(ti + 0.2)
    view.step_sensor(+1)                                   # the zoom survives a sensor change
    assert view.ax_trace.get_xlim() == (lo, hi)
    view.show_whole_window()
    lo, hi = view.ax_trace.get_xlim()
    assert lo == pytest.approx(-5.0, abs=0.01) and hi > 4.9
    assert math.isfinite(view.ax_rate.get_ylim()[1])
