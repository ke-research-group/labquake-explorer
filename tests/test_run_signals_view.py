import numpy as np
import pytest
from matplotlib.lines import Line2D

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.run_signals_view import (
    RunSignalsView, event_positions, normalize01, signal_candidates,
)


@pytest.fixture
def view(app):
    v = RunSignalsView(app, 0)
    yield v
    if v in app.child_windows:
        v.on_close()


def n_events(app):
    return len(app.data_manager.get_data("runs/[0]/events"))


# ---------------------------------------------------------------- pure helpers
def test_signal_candidates_are_aligned_numeric_arrays():
    n = 10
    run = {
        "name": "run00",
        "time": np.arange(n, dtype=float),
        "shear_stress": np.ones(n),
        "counts": np.arange(n),                      # integer dtype ok
        "with_nan": np.r_[np.nan, np.ones(n - 1)],   # NaN allowed
        "short": np.ones(n - 1),                     # wrong length
        "matrix": np.ones((n, 2)),                   # not 1-D
        "flags": np.ones(n, dtype=bool),             # bool excluded
        "event_indices": [2, 5],
        "events": [{"event_time": 2.0}, {"event_time": 5.0}],
        "nested": {"time": np.arange(n)},
    }
    assert signal_candidates(run) == ["time", "shear_stress", "counts", "with_nan"]
    assert signal_candidates({"name": "x"}) == []
    assert signal_candidates({"time": np.array([])}) == []


def test_normalize01_scales_finite_values_and_keeps_nan():
    y = np.array([2.0, np.nan, 4.0, 6.0])
    out = normalize01(y)
    assert np.isnan(out[1])
    assert out[[0, 2, 3]] == pytest.approx([0.0, 0.5, 1.0])
    assert np.all(normalize01(np.full(5, 3.0)) == 0.0)
    assert np.all(np.isnan(normalize01(np.full(3, np.nan))))


def test_event_positions_prefers_event_times_and_maps_to_x():
    t = np.linspace(0.0, 9.0, 10)
    lp = 10.0 * t
    run = {"time": t, "event_indices": [3, 7],
           "events": [{"event_time": 3.02}, {"event_time": 6.98}]}
    assert event_positions(run, t) == pytest.approx([3.0, 7.0])
    assert event_positions(run, lp) == pytest.approx([30.0, 70.0])
    # fall back to event_indices when there are no events
    run_no_events = {"time": t, "event_indices": [1, 8]}
    assert event_positions(run_no_events, lp) == pytest.approx([10.0, 80.0])
    assert event_positions({"time": t}, t).size == 0


# ----------------------------------------------------------------------- view
def test_open_lists_candidates_and_plots_default(app, view):
    assert view.title() == "Run Signals - run00"
    assert view in app.child_windows
    listed = list(view.signal_listbox.get(0, "end"))
    assert listed == view.candidates
    for expected in ("shear_stress", "displacement", "LP_displacement", "LP_velocity",
                     "normal_stress", "friction"):
        assert expected in listed
    for excluded in ("events", "event_indices", "name", "time"):
        assert excluded not in listed
    x_values = list(view.x_combo["values"])
    assert x_values[:2] == ["time", "index"]
    assert "LP_displacement" in x_values
    assert view.x_combo.get() == "time"
    assert view.normalize_var.get() is False
    assert view.mark_events_var.get() is True
    # default selection plots shear_stress against time with event markers
    assert view.selected_signals() == ["shear_stress"]
    assert [l.get_label() for l in view.signal_lines] == ["shear_stress"]
    assert len(view.event_lines) == n_events(app)
    assert view.ax.get_xlabel() == "time"
    assert view.ax.get_ylabel() == "shear_stress"


def test_registered_action_opens_view(app):
    ctx = app.context_at(app.find_item("runs/[0]"))
    assert ctx.kind == RUN
    action = [a for a in actions_for(ctx.kind) if a.label == "Plot Run Signals"][0]
    before = list(app.child_windows)
    app.run_action(action, ctx)
    opened = [w for w in app.child_windows if w not in before]
    assert len(opened) == 1 and isinstance(opened[0], RunSignalsView)
    assert opened[0].run_idx == 0
    assert RunSignalsView.result_key is None
    opened[0].on_close()


def test_plot_two_signals_with_event_markers(app, view):
    run = app.data_manager.get_data("runs/[0]")
    view.select_signals(["shear_stress", "displacement"])
    view.plot_button.invoke()
    assert len(view.signal_lines) == 2
    assert all(isinstance(l, Line2D) for l in view.signal_lines)
    assert [l.get_label() for l in view.signal_lines] == ["shear_stress", "displacement"]
    assert len(view.event_lines) == n_events(app)
    assert len(view.ax.lines) == 2 + n_events(app)
    marker_x = sorted(l.get_xdata()[0] for l in view.event_lines)
    assert marker_x == pytest.approx([e["event_time"] for e in run["events"]])
    for line in view.event_lines:
        assert line.get_linestyle() == ":"
    np.testing.assert_array_equal(view.signal_lines[0].get_ydata(), run["shear_stress"])
    np.testing.assert_array_equal(view.signal_lines[1].get_xdata(), run["time"])
    legend = view.ax.get_legend()
    assert legend is not None
    assert legend._get_loc() == 2  # matplotlib code for 'upper left'
    assert [t.get_text() for t in legend.get_texts()] == ["shear_stress", "displacement"]


def test_normalize_puts_every_line_in_unit_range(app, view):
    view.select_signals(["shear_stress", "displacement", "LP_displacement"])
    view.normalize_var.set(True)
    view.plot_button.invoke()
    assert len(view.signal_lines) == 3
    for line in view.signal_lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        assert y.size > 0
        assert y.min() == pytest.approx(0.0) and y.max() == pytest.approx(1.0)
    assert "normalized" in view.ax.get_ylabel()


def test_x_axis_lp_displacement(app, view):
    run = app.data_manager.get_data("runs/[0]")
    view.select_signals(["shear_stress"])
    view.x_combo.set("LP_displacement")
    view.plot_button.invoke()
    np.testing.assert_array_equal(view.signal_lines[0].get_xdata(), run["LP_displacement"])
    assert view.ax.get_xlabel() == "LP_displacement"
    expected = run["LP_displacement"][np.asarray(run["event_indices"])]
    marker_x = sorted(l.get_xdata()[0] for l in view.event_lines)
    assert marker_x == pytest.approx(sorted(expected))
    # sample index axis
    view.x_combo.set("index")
    view.plot_button.invoke()
    np.testing.assert_array_equal(view.signal_lines[0].get_xdata(), np.arange(len(run["time"])))
    assert view.ax.get_xlabel() == "sample index"


def test_mark_events_off_draws_no_markers(app, view):
    view.select_signals(["shear_stress"])
    view.mark_events_var.set(False)
    view.plot_button.invoke()
    assert view.event_lines == []
    assert len(view.ax.lines) == 1


def test_no_selection_plots_nothing(app, view):
    view.select_signals([])
    view.plot_button.invoke()
    assert view.signal_lines == []
    assert view.ax.get_legend() is None
    assert "Select" in view.status_var.get()


def test_zoom_is_preserved_across_replots(app, view):
    view.select_signals(["shear_stress"])
    view.plot_button.invoke()
    auto_x, auto_y = view.ax.get_xlim(), view.ax.get_ylim()
    view.ax.set_xlim(10.0, 20.0)
    view.ax.set_ylim(4.0, 6.0)
    # re-plot with another signal added: both zooms survive
    view.select_signals(["shear_stress", "normal_stress"])
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    assert view.ax.get_ylim() == pytest.approx((4.0, 6.0))
    # toggling normalize changes the y quantity: y resets, x zoom kept
    view.normalize_var.set(True)
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    assert view.ax.get_ylim() != pytest.approx((4.0, 6.0))
    # changing the x quantity resets x
    view.x_combo.set("LP_displacement")
    view.plot_button.invoke()
    assert view.ax.get_xlim() != pytest.approx((10.0, 20.0))
    # an un-zoomed plot re-autoscales
    view.x_combo.set("time")
    view.normalize_var.set(False)
    view.select_signals(["shear_stress"])
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    assert view.ax.get_ylim() == pytest.approx(auto_y)


def test_arrays_with_nan_are_listed_and_plotted(app, view):
    run = app.data_manager.get_data("runs/[0]")
    gappy = np.array(run["shear_stress"], dtype=float)
    gappy[100:200] = np.nan
    run["gappy"] = gappy
    view.refresh_candidates()
    assert "gappy" in view.candidates
    assert "gappy" in list(view.x_combo["values"])
    view.select_signals(["gappy"])
    view.normalize_var.set(True)
    view.plot_button.invoke()
    y = np.asarray(view.signal_lines[0].get_ydata(), dtype=float)
    assert np.isnan(y[150])
    finite = y[np.isfinite(y)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    del run["gappy"]


def test_refresh_keeps_selection(app, view):
    view.select_signals(["displacement", "friction"])
    view.x_combo.set("LP_displacement")
    view.refresh_candidates()
    assert view.selected_signals() == ["displacement", "friction"]
    assert view.x_combo.get() == "LP_displacement"


def test_close_unregisters(app):
    v = RunSignalsView(app, 0)
    assert v in app.child_windows
    v.on_close()
    assert v not in app.child_windows
