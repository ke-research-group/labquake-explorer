"""Views address rows of channel arrays ('slip/slip_3') like plain fields."""
import numpy as np
import pytest

from labquake_explorer.data.channels import channel_array, positions_table
from labquake_explorer.ui.views import EventAnalyzerView, InterEventView, RunSignalsView


@pytest.fixture
def app_with_arrays(app):
    """The synthetic experiment with its displacement moved into a 'slip' channel
    array (two sensors) and the volt-like channels into 'raw_data'."""
    run = app.data_manager.get_data("runs/[0]")
    d = run["displacement"]
    run["slip"] = channel_array(np.vstack([d, 2.0 * d]), ["slip_1", "slip_2"], unit="um",
                                positions=positions_table(["slip_1", "slip_2"], {"slip_1": (50.0, 0.0, 0.0), "slip_2": (450.0, 0.0, 0.0)}))
    run["raw_data"] = channel_array(np.vstack([run["LP_displacement"] / 100.0]), ["lvdt"], unit="V")
    for event in run["events"]:
        n = event["time"].size
        i0 = int(np.searchsorted(run["time"], event["time"][0]))
        event["slip"] = {k: (v[:, i0:i0 + n] if k == "data" else v) for k, v in run["slip"].items()}
    app.refresh_tree()
    return app


def test_event_analyzer_lists_and_uses_rows(app_with_arrays):
    v = EventAnalyzerView(app_with_arrays, 0, 1)
    fields = list(v.data_x_combo["values"])
    assert "slip/slip_1" in fields and "slip/slip_2" in fields and "displacement" in fields
    v.data_x_combo.set("slip/slip_2")
    v.data_selected()
    assert v.item_x == "slip/slip_2" and v.data_x.shape == v.data_y.shape
    np.testing.assert_array_equal(v.data_x, v.event["slip"]["data"][1])
    v.update_analysis()
    assert v.result["x_field"] == "slip/slip_2"
    # the row is read back by path when the picks are applied to every event
    assert v.apply_to_all_events(confirm=False) == 4
    assert app_with_arrays.data_manager.get_data("runs/[0]/events/[0]/event_analysis")["x_field"] == "slip/slip_2"
    v.on_close()


def test_run_signals_offers_rows(app_with_arrays):
    v = RunSignalsView(app_with_arrays, 0)
    assert "slip/slip_1" in v.candidates and "raw_data/lvdt" in v.candidates
    v.select_signals(["slip/slip_1", "raw_data/lvdt"])
    v.plot()
    assert len(v.signal_lines) == 2
    v.on_close()


def test_interevent_uses_rows(app_with_arrays):
    v = InterEventView(app_with_arrays, 0)
    assert "slip/slip_2" in v.slip_combo["values"]
    v.slip_combo.set("slip/slip_2")
    v.compute()
    truth = app_with_arrays.truth[0]
    expected = 2.0 * (truth.slip + truth.creep_fraction * truth.lp_velocity * 12.0)
    assert v.result["slip_field"] == "slip/slip_2"
    assert v.result["slip_per_cycle"][1:] == pytest.approx([expected] * (len(v.result["slip_per_cycle"]) - 1), rel=1e-6)
    v.on_close()


def test_tree_label_for_channel_array(app_with_arrays):
    label = app_with_arrays.format_tree_label("slip", app_with_arrays.data_manager.get_data("runs/[0]/slip"))
    assert label.startswith("slip: 2 channels x ") and label.endswith(" um")
