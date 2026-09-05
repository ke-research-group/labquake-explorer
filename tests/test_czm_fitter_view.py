import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.ui.views import CZMFitterView

EXPECTED_KEYS = {"Cf", "y", "Xc", "Gc", "x_min", "x_tip", "x_max", "x_lim_min", "x_lim_max", "strain_gauge"}


@pytest.fixture
def view(app_with_strain):
    v = CZMFitterView(app_with_strain, 0, 1)
    yield v
    try:
        v.on_close()
    except Exception:
        pass


def positions(v):
    return [float(line.get_xdata()[0]) for line in v.vlines]


def mouse_event(v, name, xdata, button=1):
    """A matplotlib MouseEvent on the top axes at data x ``xdata``."""
    ax = v.axs[0]
    ydata = float(np.mean(ax.get_ylim()))
    x, y = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, v.canvas, x, y, button=button)


def dispatch(v, name, xdata, button=1):
    """Send a MouseEvent through the canvas callback registry (mpl_connect path)."""
    v.canvas.callbacks.process(name, mouse_event(v, name, xdata, button=button))


def test_open_lists_gauges(app_with_strain, view):
    assert view.event_idx == 1
    assert view.title() == "Cohesive Zone Model Fitting - Event 1"
    assert view in app_with_strain.child_windows
    assert view.num_gauges == 16
    assert list(view.gauge_combobox["values"]) == [str(i) for i in range(16)]
    assert view.strain_gauge.get() == 6
    assert view.event_combobox.get() == "1"
    # three draggable marker pairs on the two shared axes
    assert len(view.vlines) == 3 and len(view.vlines_twin) == 3
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx(positions(view))
    assert view.axs[0].get_shared_x_axes().joined(view.axs[0], view.axs[1])
    assert view.axs[0].get_xlim() == pytest.approx((-0.1, 0.1))
    # default Cf falls back to 10 when the event has no rupture_speed
    assert view.Cf.get() == 10
    assert view.y.get() == pytest.approx(8e-3)


def test_switch_event_reloads(app_with_strain, view):
    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Cohesive Zone Model Fitting - Event 2"
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"
    assert len(view.vlines) == 3
    assert view.figure._suptitle.get_text().endswith("event2")


def test_save_writes_dict(app_with_strain, view):
    view.Cf.set(1000.0)
    view.strain_gauge.set(3)
    view.save_parameters()
    saved = app_with_strain.data_manager.get_data("runs/[0]/events/[1]/czm_parms")
    assert isinstance(saved, dict)
    assert set(saved) == EXPECTED_KEYS
    assert saved is view.event["czm_parms"]
    assert saved["Cf"] == 1000.0
    assert saved["strain_gauge"] == 3
    assert saved["x_min"] < saved["x_tip"] < saved["x_max"]
    assert (saved["x_lim_min"], saved["x_lim_max"]) == pytest.approx((-0.1, 0.1))
    # the tree shows the new key
    tree = app_with_strain.data_tree
    labels = []

    def walk(item):
        for child in tree.get_children(item):
            labels.append(tree.item(child, "text"))
            walk(child)
    walk("")
    assert any(label.startswith("czm_parms") for label in labels)


def test_reopen_restores_parameters_and_lines(app_with_strain, view):
    view.Cf.set(1000.0)
    view.y.set(5e-3)
    view.Xc.set(2.5)
    view.Gc.set(3.0)
    view.move_line(0, -0.006)
    view.move_line(1, -0.001)
    view.move_line(2, 0.004)
    view.axs[0].set_xlim(-0.02, 0.02)
    view.save_parameters()
    saved = view.event["czm_parms"]
    assert saved["x_min"] == pytest.approx(-0.006)
    assert saved["x_tip"] == pytest.approx(-0.001)
    assert saved["x_max"] == pytest.approx(0.004)

    view.set_event(2)
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    assert view.Cf.get() == 10

    view.set_event(1)
    assert view.Cf.get() == 1000.0
    assert view.y.get() == pytest.approx(5e-3)
    assert view.Xc.get() == pytest.approx(2.5)
    assert view.Gc.get() == pytest.approx(3.0)
    assert positions(view) == pytest.approx([-0.006, -0.001, 0.004])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx([-0.006, -0.001, 0.004])
    assert view.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))

    # a fresh window on the same event also restores everything
    other = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert other.Cf.get() == 1000.0
        assert positions(other) == pytest.approx([-0.006, -0.001, 0.004])
        assert other.strain_gauge.get() == saved["strain_gauge"]
    finally:
        other.on_close()


def test_legacy_list_is_read(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[3]")
    event["czm_parms"] = [500.0, 5e-3, 2.0, 3.0, -0.004, 0.0, -0.02, 0.02]
    v = CZMFitterView(app_with_strain, 0, 3)
    try:
        assert v.Cf.get() == 500.0
        assert v.y.get() == pytest.approx(5e-3)
        assert v.Xc.get() == 2.0
        assert v.Gc.get() == 3.0
        assert positions(v) == pytest.approx([-0.004, 0.0, 0.004])
        assert v.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))
        assert v.strain_gauge.get() == 6
        # loading does not rewrite the stored value; saving does
        assert isinstance(event["czm_parms"], list)
        v.save_parameters()
        assert isinstance(event["czm_parms"], dict)
        assert set(event["czm_parms"]) == EXPECTED_KEYS
        assert event["czm_parms"]["x_max"] == pytest.approx(0.004)
    finally:
        v.on_close()


def test_filter_toggle_and_update(app_with_strain, view):
    assert view.filter_button["text"] == "Filter Off"
    view.toggle_filter()
    assert view.filtering is True
    assert view.filter_button["text"] == "Filter On"
    view.filter_window.set(50)  # even -> corrected to odd
    view.update_plot()
    assert view.filter_window.get() == 51
    view.gauge_combobox.set("2")
    view.update_plot()
    assert view.strain_gauge.get() == 2
    view.toggle_filter()
    assert view.filtering is False
    assert view.filter_button["text"] == "Filter Off"
    assert view.validate_filter_window("51") is True
    assert view.validate_filter_window("50") is False
    assert view.validate_filter_window("") is True
    assert view.validate_filter_window("abc") is False


def test_fit_improves_objective(app_with_strain, view):
    view.Cf.set(1000.0)
    initial_guess = [view.Gc.get(), view.Xc.get()]
    objective = view.build_fit_objective()
    assert objective is not None
    initial_fun = objective(initial_guess)
    assert np.isfinite(initial_fun) and initial_fun > 0

    result = view.fit_parameters()
    assert result is not None
    assert result.success, result.message
    assert result.initial_fun == pytest.approx(initial_fun)
    # the optimizer actually did something: lower objective, parameters moved
    assert result.fun < initial_fun
    assert result.fun == pytest.approx(objective(result.x))
    assert (view.Gc.get(), view.Xc.get()) != tuple(initial_guess)
    assert view.Gc.get() == pytest.approx(result.x[0])
    assert view.Xc.get() == pytest.approx(result.x[1])
    assert view.Gc.get() >= 1e-6 and view.Xc.get() >= 1e-6  # within the bounds
    # the plotted CZM curve follows the fitted parameters
    assert any(line.get_label() == "CZM" for line in view.axs[0].get_lines())

    # fitting again with the filter on also succeeds and does not get worse
    view.toggle_filter()
    result2 = view.fit_parameters()
    assert result2 is not None and result2.success
    assert result2.fun <= result2.initial_fun + 1e-9 * max(1.0, abs(result2.initial_fun))


def test_fit_without_samples_returns_none(app_with_strain, view):
    # tip and max lines beyond the end of the strain record -> no samples in window
    t_end = float(view._time()[-1])
    view.move_line(1, t_end + 1.0)
    view.move_line(2, t_end + 2.0)
    gc, xc = view.Gc.get(), view.Xc.get()
    assert view.build_fit_objective() is None
    assert view.fit_parameters() is None
    assert (view.Gc.get(), view.Xc.get()) == (gc, xc)


def test_drag_line_with_mouse_events(app_with_strain, view):
    view.canvas.draw()
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])

    # press far from any line: nothing is grabbed, motion does nothing
    dispatch(view, "button_press_event", 0.025)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    dispatch(view, "button_release_event", 0.03)

    # right button does not grab either
    dispatch(view, "button_press_event", 0.0, button=3)
    assert view.drag_active is False
    dispatch(view, "button_release_event", 0.0, button=3)

    # left press on the tip line grabs it; motion drags both marker pairs
    dispatch(view, "button_press_event", 0.0)
    assert view.drag_active is True and view.active_line_idx == 1
    dispatch(view, "motion_notify_event", 0.02)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx([-0.05, 0.02, 0.05])
    assert view.line_positions == pytest.approx([-0.05, 0.02, 0.05])

    # while the toolbar is in pan mode the drag is ignored
    view.toolbar.pan()
    assert view.toolbar_active()
    dispatch(view, "motion_notify_event", 0.04)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    view.toolbar.pan()
    assert not view.toolbar_active()

    # ... and resumes once pan is switched off
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])

    # release ends the drag; further motion does nothing
    dispatch(view, "button_release_event", 0.03)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.06)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])

    # a press while pan mode is on is ignored too
    view.toolbar.pan()
    dispatch(view, "button_press_event", 0.03)
    assert view.drag_active is False
    view.toolbar.pan()

    # the dragged position is what gets saved
    view.save_parameters()
    assert view.event["czm_parms"]["x_tip"] == pytest.approx(0.03)


def test_close_unregisters(app_with_strain):
    v = CZMFitterView(app_with_strain, 0, 0)
    assert v in app_with_strain.child_windows
    v.on_close()
    assert v not in app_with_strain.child_windows
