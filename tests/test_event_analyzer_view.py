import numpy as np
import pytest

from labquake_explorer.ui.views import EventAnalyzerView


@pytest.fixture
def view(app):
    v = EventAnalyzerView(app, 0, 1)
    yield v
    v.on_close()


def test_open_and_defaults(app, view):
    truth = app.truth[0]
    assert view.event_idx == 1
    assert view.title() == "Event Analyzer - Event 1"
    assert view.data_x_combo.get() == "displacement"
    assert view.data_y_combo.get() == "shear_stress"
    assert len(view.picked_idx) == 6
    assert view in app.child_windows
    # default loading range lies inside the loading phase: slope = d(tau)/d(displacement)
    k = float(view.loading_slope_text.get())
    assert k == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert float(view.stress_drop_text.get()) > 0


def test_switch_event_reloads(app, view):
    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Event Analyzer - Event 2"
    assert view.event is app.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"


def test_save_writes_event_analysis(app, view):
    view.save_event()
    saved = app.data_manager.get_data("runs/[0]/events/[1]/event_analysis")
    assert saved is view.event["event_analysis"]
    assert set(saved) >= {"loading_indices", "unloading_indices", "rupture_start_index",
                          "rupture_end_index", "loading_stiffness", "unloading_stiffness",
                          "stress_drop", "displacement"}
    assert saved["loading_indices"] == view.picked_idx[:2]
    # reopening restores the saved picks
    view.picked_idx[4] = 3
    view.set_event(1)
    assert view.picked_idx == [saved["loading_indices"][0], saved["loading_indices"][1],
                               saved["unloading_indices"][0], saved["unloading_indices"][1],
                               saved["rupture_start_index"], saved["rupture_end_index"]]


def test_close_unregisters(app):
    v = EventAnalyzerView(app, 0, 0)
    assert v in app.child_windows
    v.on_close()
    assert v not in app.child_windows
