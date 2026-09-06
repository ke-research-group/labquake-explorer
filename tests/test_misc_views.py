"""Tests for the path/data-based views: SlopeAnalyzerView, IndexPickerView,
PointsSelectorView and SimplePlotView, plus the main-window actions that open them."""
import numpy as np
import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import (
    IndexPickerView, PointsSelectorView, SimplePlotView, SlopeAnalyzerView,
)


def run_menu_action(app, path, label):
    """Select ``path`` in the tree and run the context-menu action ``label`` on it."""
    item = app.find_item(path)
    assert item is not None, path
    app.data_tree.selection_set(item)
    ctx = app.context_at(item)
    action = [a for a in actions_for(ctx.kind) if a.label == label][0]
    app.run_action(action, ctx)
    return ctx


# ---------------------------------------------------------------- SlopeAnalyzerView
def test_slope_analyzer_lists_siblings_and_computes_slope(app):
    view = SlopeAnalyzerView(app, item_y="runs/[0]/shear_stress")
    try:
        assert view.title() == "Slope Analyzer"
        assert view in app.child_windows
        names = list(view.data_y_combo["values"])
        assert names == list(view.data_x_combo["values"])
        for expected in ("time", "shear_stress", "displacement", "LP_displacement",
                         "LP_velocity", "normal_stress", "friction"):
            assert expected in names
        assert "name" not in names
        assert view.data_y_combo.get() == "shear_stress"
        assert view.data_x_combo.get() == ""
        assert view.base_path == "runs/[0]"
        assert len(view.picked_idx) == 2
        assert len(view.markers) == 2
        assert [m.get_label() for m in view.markers] == ["0", "1"]
        slope = float(view.slope_textbox.get())
        assert np.isfinite(slope)
        assert slope == view.slope
        # X defaults to the sample index
        n = len(app.truth[0].time)
        assert len(view.data_x) == n and view.data_x[-1] == n - 1
    finally:
        view.on_close()
    assert view not in app.child_windows


def test_slope_analyzer_loading_rate_vs_time(app):
    truth = app.truth[0]
    view = SlopeAnalyzerView(app, item_y="runs/[0]/shear_stress")
    try:
        view.data_x_combo.set("time")
        view.data_x_selected(None)
        assert view.item_x == "time"
        assert view.ax.get_xlabel() == "time"
        # move both picks into the loading phase between the first two events
        i0, i1 = truth.event_indices[0] + 200, truth.event_indices[1] - 200
        view.picked_idx = [i0, i1]
        view.plot_picked_points()
        slope = float(view.slope_textbox.get())
        assert slope == pytest.approx(truth.loading_rate, rel=1e-6)
        assert view.slope_line is not None
        xs, ys = view.slope_line.get_data()
        assert list(xs) == [truth.time[i0], truth.time[i1]]
        assert list(ys) == [truth.shear_stress[i0], truth.shear_stress[i1]]
        # dragging-path update keeps the line and the readout in sync
        view.picked_idx[1] = truth.event_indices[1] - 100
        view.update_overlays()
        view.update_readout()
        assert float(view.slope_textbox.get()) == pytest.approx(truth.loading_rate, rel=1e-6)
        assert view.slope_line.get_data()[0][1] == truth.time[view.picked_idx[1]]
    finally:
        view.on_close()


def test_slope_analyzer_item_x_in_constructor(app):
    truth = app.truth[0]
    view = SlopeAnalyzerView(app, item_y="runs/[0]/shear_stress", item_x="LP_displacement")
    try:
        assert view.data_x_combo.get() == "LP_displacement"
        view.picked_idx = [truth.event_indices[0] + 200, truth.event_indices[1] - 200]
        view.plot_picked_points()
        assert float(view.slope_textbox.get()) == pytest.approx(truth.stiffness_lp, rel=1e-6)
    finally:
        view.on_close()


def test_slope_analyzer_switch_y_clamps_picks(app):
    view = SlopeAnalyzerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    try:
        assert view.base_path == "runs/[0]/events/[1]"
        names = list(view.data_y_combo["values"])
        assert "shear_stress" in names and "time" in names
        assert "event_time" not in names  # scalar, not an array
        n = len(app.data_manager.get_data("runs/[0]/events/[1]/shear_stress"))
        assert view.picked_idx == [int(n / 3), int(n / 3 * 2)]
        view.data_y_combo.set("displacement")
        view.data_y_selected(None)
        assert view.item_y == "displacement"
        assert view.ax.get_ylabel() == "displacement"
        assert np.isfinite(float(view.slope_textbox.get()))
    finally:
        view.on_close()


# ------------------------------------------------------------------ IndexPickerView
def test_index_picker_reports_indices(app):
    view = IndexPickerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    try:
        assert view.title() == "Index Picker"
        assert view in app.child_windows
        assert view.data_y_combo.get() == "shear_stress"
        n = len(app.data_manager.get_data("runs/[0]/events/[1]/shear_stress"))
        assert view.picked_idx == [int(n / 3), int(n / 3 * 2)]
        assert view.index_textbox.get() == str(view.picked_idx)
        assert view.clipboard_get() == str(view.picked_idx)
        view.picked_idx = [10, 20]
        view.plot_picked_points()
        assert view.index_textbox.get() == "[10, 20]"
        assert [m.center for m in view.markers] == [(view.data_x[10], view.data_y[10]),
                                                    (view.data_x[20], view.data_y[20])]
        # drag path: nearest sample to a point on the curve is that sample
        assert view.nearest_index(view.data_x[15], view.data_y[15]) == 15
    finally:
        view.on_close()
    assert view not in app.child_windows


def test_index_picker_with_time_axis(app):
    view = IndexPickerView(app, item_y="runs/[0]/shear_stress", item_x="time")
    try:
        assert view.data_x_combo.get() == "time"
        assert np.array_equal(view.data_x, app.truth[0].time)
    finally:
        view.on_close()


def test_index_picker_opens_via_action(app):
    run_menu_action(app, "runs/[0]/events/[0]/displacement", "Pick Indices")
    views = [w for w in app.child_windows if isinstance(w, IndexPickerView)]
    assert len(views) == 1
    assert views[0].item_y == "displacement"
    assert views[0].base_path == "runs/[0]/events/[0]"
    views[0].on_close()

    run_menu_action(app, "runs/[0]/friction", "Extract Slopes")
    views = [w for w in app.child_windows if isinstance(w, SlopeAnalyzerView)]
    assert len(views) == 1
    assert views[0].item_y == "friction"
    views[0].on_close()


# ---------------------------------------------------------------- PointsSelectorView
def test_points_selector_save_sorts_and_calls_back(app):
    x = np.linspace(0, 2 * np.pi, 500)
    y = np.sin(x)
    received = []
    view = PointsSelectorView(app, x, y, [300, 100, 200], add_remove_enabled=True,
                              callback=received.append, xlabel="x", ylabel="sin",
                              title="demo")
    try:
        assert view.title() == "Points Selector"
        assert view in app.child_windows
        assert hasattr(view, "save_button")
        assert view.ax.get_xlabel() == "x" and view.ax.get_ylabel() == "sin"
        assert view.ax.get_title() == "demo"
        assert len(view.markers) == 3
        view.save()
        assert received == [[100, 200, 300]]
        # the callback gets a copy, never the live list
        assert received[0] == view.picked_idx and received[0] is not view.picked_idx
        # markers follow the sorted order
        assert [m.get_label() for m in view.markers] == ["0", "1", "2"]
        assert [m.center[0] for m in view.markers] == [x[100], x[200], x[300]]
    finally:
        view.on_close()
    assert view not in app.child_windows


def test_points_selector_add_remove_renumbers(app):
    x = np.arange(100.0)
    y = x ** 2
    view = PointsSelectorView(app, x, y, [10, 20, 30], add_remove_enabled=True)
    try:
        assert not hasattr(view, "save_button")
        assert view.add_point(55) == 3
        assert view.picked_idx == [10, 20, 30, 55]
        assert [m.get_label() for m in view.markers] == ["0", "1", "2", "3"]
        view.remove_point(1)
        assert view.picked_idx == [10, 30, 55]
        assert [m.get_label() for m in view.markers] == ["0", "1", "2"]
        # every marker still sits on the sample its label points to
        for marker in view.markers:
            idx = view.picked_idx[int(marker.get_label())]
            assert marker.center == (x[idx], y[idx])
        assert len(view.ax.patches) == 3
        assert view.nearest_index(42.2, 42.2 ** 2) == 42
    finally:
        view.on_close()


def test_points_selector_edits_after_save_stay_local(app):
    x = np.arange(50.0)
    y = x
    received = []
    view = PointsSelectorView(app, x, y, [10, 20], add_remove_enabled=True,
                              callback=received.append)
    try:
        view.save()
        saved = received[-1]
        assert saved == [10, 20]
        # a drag, an add and a removal in the still-open window must not
        # reach what the callback already stored
        view.picked_idx[0] = 5
        view.add_point(40)
        view.remove_point(1)
        assert view.picked_idx == [5, 40]
        assert saved == [10, 20]
        # the next Save delivers the new state, again as a fresh copy
        view.save()
        assert received[-1] == [5, 40]
        assert received[-1] is not received[-2]
        assert received[-1] is not view.picked_idx
        assert saved == [10, 20]
    finally:
        view.on_close()


def test_points_selector_does_not_mutate_input(app):
    x = np.arange(10.0)
    y = x
    picks = [np.int64(3), np.int64(7)]
    view = PointsSelectorView(app, x, y, picks)
    try:
        assert view.picked_idx == [3, 7]
        assert all(type(i) is int for i in view.picked_idx)
        view.picked_idx[0] = 1
        assert picks[0] == 3
    finally:
        view.on_close()


def test_pick_events_action_preloads_event_indices(app):
    truth = app.truth[0]
    ctx = run_menu_action(app, "runs/[0]/shear_stress", "Pick Events")
    assert ctx.key == "shear_stress"
    views = [w for w in app.child_windows if isinstance(w, PointsSelectorView)]
    assert len(views) == 1
    view = views[0]
    assert view.picked_idx == list(truth.event_indices)
    assert view.add_remove_enabled is True
    assert view.callback is not None
    assert view.ax.get_ylabel() == "shear_stress"
    assert view.ax.get_title() == "runs/[0]/shear_stress"
    assert np.array_equal(view.y_values, truth.shear_stress)
    # saving writes back through the callback into the data manager
    view.add_point(5)
    view.save()
    stored = app.data_manager.get_data("runs/[0]/event_indices")
    assert stored == [5] + list(truth.event_indices)
    # the stored list is not aliased with the view's: later edits in the
    # still-open window do not change the data until Save is pressed again
    assert stored is not view.picked_idx
    view.remove_point(0)
    view.add_point(7)
    assert app.data_manager.get_data("runs/[0]/event_indices") == [5] + list(truth.event_indices)
    view.save()
    assert app.data_manager.get_data("runs/[0]/event_indices") == [7] + list(truth.event_indices)
    view.on_close()


def test_min_max_action_picks_argmax_argmin(app):
    path = "runs/[0]/events/[1]/shear_stress"
    run_menu_action(app, path, "Min/Max")
    views = [w for w in app.child_windows if isinstance(w, PointsSelectorView)]
    assert len(views) == 1
    view = views[0]
    y = app.data_manager.get_data(path)
    assert view.picked_idx == [int(np.argmax(y)), int(np.argmin(y))]
    assert view.add_remove_enabled is False
    assert view.callback is None
    assert len(view.markers) == 2
    view.on_close()


def test_min_max_action_ignores_nan_samples(app, no_dialogs):
    path = "runs/[0]/events/[1]/shear_stress"
    y = app.data_manager.get_data(path)
    y[:5] = np.nan                                   # np.argmin/argmax would return 0
    run_menu_action(app, path, "Min/Max")
    view = [w for w in app.child_windows if isinstance(w, PointsSelectorView)][-1]
    assert view.picked_idx == [int(np.nanargmax(y)), int(np.nanargmin(y))]
    assert 0 not in view.picked_idx
    view.on_close()
    # all-NaN: a warning, no window, no exception
    y[:] = np.nan
    before = len(app.child_windows)
    run_menu_action(app, path, "Min/Max")
    assert len(app.child_windows) == before
    assert any("no finite samples" in str(call) for call in no_dialogs)


def test_nearest_index_skips_nan_samples(app):
    """Dragging a marker onto a NaN gap must snap to a finite neighbour, never
    the NaN sample (an Ellipse centred at (x, nan) is invisible and can never
    be picked up again)."""
    x = np.arange(100.0)
    y = x.copy()
    y[40:50] = np.nan
    view = PointsSelectorView(app, x, y, [10, 20], add_remove_enabled=True)
    try:
        idx = view.nearest_index(45.0, 45.0)
        assert idx in (39, 50)
        assert np.isfinite(view.y_values[idx])
        assert view.add_point(view.nearest_index(44.0, 44.0)) == 2
        assert np.isfinite(view.y_values[view.picked_idx[-1]])
    finally:
        view.on_close()
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    event["shear_stress"][100:200] = np.nan
    view = IndexPickerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    try:
        idx = view.nearest_index(view.data_x[150], float(np.nanmean(view.data_y)))
        assert not 100 <= idx < 200
        assert np.isfinite(view.data_y[idx])
    finally:
        view.on_close()


# ------------------------------------------------------------------- SimplePlotView
def test_simple_plot_view(app):
    view = SimplePlotView(app)
    try:
        assert view.title() == "Simple Plot"
        assert view in app.child_windows
        assert view.ax is view.figure.axes[0]
        view.ax.plot(app.truth[0].shear_stress)
        view.canvas.draw()
        assert view.toolbar is not None
    finally:
        view.on_close()
    assert view not in app.child_windows


def test_simple_plot_view_close_drops_duplicate_registrations(app):
    view = SimplePlotView(app)
    assert app.child_windows.count(view) == 1
    # a caller that appends the already-registered view a second time
    app.child_windows.append(view)
    assert app.child_windows.count(view) == 2
    view.on_close()
    assert view not in app.child_windows
    assert not view.winfo_exists()


def test_double_click_array_opens_simple_plot_and_close_unregisters(app):
    path = "runs/[0]/shear_stress"
    item = app.find_item(path)
    app.data_tree.selection_set(item)
    before = list(app.child_windows)
    app.on_double_click(None)
    new = [w for w in app.child_windows if w not in before]
    assert len(new) >= 1
    view = new[0]
    assert isinstance(view, SimplePlotView)
    assert all(w is view for w in new)
    assert view.ax.get_ylabel() == "shear_stress"
    assert view.ax.get_title() == "runs[0]/shear_stress"  # main window strips '/[' -> '['
    line, = view.ax.get_lines()
    assert np.array_equal(line.get_ydata(), app.truth[0].shear_stress)
    view.on_close()
    # no stale (destroyed) entry survives, however many times the main
    # window registered the view
    assert view not in app.child_windows
    assert app.child_windows == before
