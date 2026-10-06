import numpy as np
import pytest

from labquake_explorer.ui.context import resolve_context
from labquake_explorer.ui.views import CZMFitterView, DynamicStrainArrivalPickerView

N_CHANNELS = 16
OLD_LAYOUT_ENABLED = [i % 2 == 0 for i in range(N_CHANNELS)]
OLD_LAYOUT_FITTING = [i in (6, 8, 10, 12) for i in range(N_CHANNELS)]


@pytest.fixture
def view(app_with_strain):
    v = DynamicStrainArrivalPickerView(app_with_strain, 0, 1)
    yield v
    v.on_close()


def _speed_from_picks(view):
    """Independent recomputation of Cf from the picked indices of the fitting channels."""
    strain = view.event["strain"]
    t = strain["original"]["time"]
    fit = [i for i, f in enumerate(view.fitting_channels) if f]
    x = np.array([t[view.picked_idx[i]] for i in fit])
    y = np.array([strain["locations"][i] for i in fit], dtype=float)
    slope = np.polyfit(y, x, 1)[0]
    return -1e-3 / slope


def _move_all_picks(view, idx):
    """Put every marker on sample ``idx`` (as a sequence of drags would)."""
    view.picked_idx = [idx] * len(view.picked_idx)
    view.draw_markers()
    view.update_fitted_line()


def test_open_and_defaults(app_with_strain, view):
    assert view.event_idx == 1
    assert view.title() == "Pick Arrivals - Event 1"
    assert view in app_with_strain.child_windows
    assert view.event_combobox.get() == "1"
    # 'p0001' -> experiment 1 -> old (paired gauge) layout
    assert view.exp_number() == 1
    assert view.enabled_channels == OLD_LAYOUT_ENABLED
    assert view.fitting_channels == OLD_LAYOUT_FITTING
    truth_locs = list(view.event["strain_truth"]["locations_mm"])
    assert view.event["strain"]["locations"] == pytest.approx(truth_locs)
    assert len(view.picked_idx) == N_CHANNELS
    assert sum(line is not None for line in view.lines) == 8
    assert len(view.fitting_markers) == 4
    assert len(view.not_fitting_markers) == 4
    assert len(view.enabled_channels_mb.items) == N_CHANNELS
    assert [v.get() for v in view.fitting_channels_mb.items] == [int(f) for f in OLD_LAYOUT_FITTING]


def test_from_context(app_with_strain):
    ctx = resolve_context("runs/[0]/events/[2]")
    v = DynamicStrainArrivalPickerView.from_context(app_with_strain, ctx)
    try:
        assert v.event_idx == 2
        assert v.run_idx == 0
    finally:
        v.on_close()


def test_default_identical_picks_report_na(app_with_strain, view):
    # every marker starts at the middle sample -> no propagation -> no speed, no crash
    assert len(set(view.picked_idx)) == 1
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    view.save()
    # a degenerate fit saves the picks but no rupture_speed at all (not nan)
    assert "rupture_speed" not in view.event
    assert "rupture_speed" not in app_with_strain.data_manager.get_data("runs/[0]/events/[1]")
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/strain/original/picked_idx") == view.picked_idx


def test_identical_picks_away_from_zero_report_na(view):
    # polyfit through four identical times at a non-zero relative time gives a slope of
    # ~1e-21 rather than 0; that must still be reported as no speed, not ~1e18 m/s
    t = view.event["strain"]["original"]["time"]
    target = 3 * len(t) // 4
    assert t[target] - view.event["event_time"] != 0
    _move_all_picks(view, target)
    assert all(m.center[0] == pytest.approx(t[target] - view.event["event_time"]) for m in view.fitting_markers)
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    assert view.fitted_line is None
    # and the pure fit helper agrees
    locs = [10.5, 22.5, 34.5, 46.5]
    assert DynamicStrainArrivalPickerView.fit_arrival_line(locs, [5e-3] * 4) is None
    assert DynamicStrainArrivalPickerView.fit_arrival_line(locs, [1e-3, 1e-3, 1e-3, 1e-3 + 1e-12]) is not None
    assert DynamicStrainArrivalPickerView.fit_arrival_line([10.5] * 4, [1e-3, 2e-3, 3e-3, 4e-3]) is None
    assert DynamicStrainArrivalPickerView.fit_arrival_line([10.5], [1e-3]) is None


def test_degenerate_save_removes_stale_speed_and_czm_falls_back(app_with_strain, view):
    dm = app_with_strain.data_manager
    view.magic()
    view.save()
    good = dm.get_data("runs/[0]/events/[1]/rupture_speed")
    assert np.isfinite(good)

    # saving degenerate picks must not leave a nan or the stale value behind
    _move_all_picks(view, 3 * len(view.event["strain"]["original"]["time"]) // 4)
    assert np.isnan(view.rupture_speed)
    view.save()
    assert "rupture_speed" not in dm.get_data("runs/[0]/events/[1]")
    assert dm.get_data("runs/[0]/events/[1]/strain/original/picked_idx") == view.picked_idx

    # the CZM fitter opens with its default Cf rather than nan
    czm = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert czm.Cf.get() == pytest.approx(10)
    finally:
        czm.on_close()

    # a well-posed save afterwards restores the key and the CZM default
    view.magic()
    view.save()
    assert dm.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(good)
    czm = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert czm.Cf.get() == pytest.approx(abs(good))
    finally:
        czm.on_close()


def test_magic_then_save(app_with_strain, view):
    view.magic()
    strain = view.event["strain"]
    raw = strain["original"]["raw"]
    t = strain["original"]["time"]

    # Magic picks the extreme of the plotted (sign-flipped, scaled) trace: a sample on
    # the saturated plateau of each step (raw at its maximum, up to floating-point
    # tie-breaking among plateau samples), not the onset itself.
    for i, enabled in enumerate(view.enabled_channels):
        if enabled:
            pick = view.picked_idx[i]
            assert raw[i][pick] == pytest.approx(raw[i].max(), rel=1e-9)
            assert pick == int(np.argmin(view.lines[i].get_data()[1]))
    # all picks trail the true arrivals by a near-constant lag, so Cf is consistent
    truth = view.event["strain_truth"]
    arrival_idx = [int(np.argmin(np.abs(t - a))) for a in truth["arrival_times"]]
    lags = [view.picked_idx[i] - arrival_idx[i] for i, f in enumerate(view.fitting_channels) if f]
    assert all(lag > 0 for lag in lags)
    assert max(lags) - min(lags) <= 2

    expected = _speed_from_picks(view)
    assert np.isfinite(view.rupture_speed)
    assert view.rupture_speed == pytest.approx(expected, rel=1e-9)
    # sample quantization of the picks limits the accuracy; magnitude must still be right
    assert 0.5 * truth["rupture_speed"] < abs(view.rupture_speed) < 2.0 * truth["rupture_speed"]
    assert view.cf_label.cget("text") == f"Cf = {view.rupture_speed:.2f} m/s"
    assert view.fitted_line is not None

    view.save()
    dm = app_with_strain.data_manager
    base = "runs/[0]/events/[1]"
    assert dm.get_data(f"{base}/rupture_speed") == pytest.approx(expected, rel=1e-9)
    assert view.event["rupture_speed"] == dm.get_data(f"{base}/rupture_speed")
    assert dm.get_data(f"{base}/strain/enabled_channels") == OLD_LAYOUT_ENABLED
    assert dm.get_data(f"{base}/strain/fitting_channels") == OLD_LAYOUT_FITTING
    assert dm.get_data(f"{base}/strain/original/picked_idx") == view.picked_idx
    arrival_time = dm.get_data(f"{base}/strain/original/rupture_arrival_time")
    np.testing.assert_allclose(arrival_time, t[view.picked_idx])
    # nothing leaked to the top level of the event other than rupture_speed
    assert "enabled_channels" not in view.event
    assert "picked_idx" not in view.event


def test_switch_event_reloads_channel_state(app_with_strain, view):
    view.magic()
    view.fitting_channels_mb.items[0].set(1)
    view.fitting_channels_changed()
    assert view.fitting_channels[0] is True
    assert len(view.fitting_markers) == 5
    view.save()
    saved_picks = list(view.picked_idx)
    saved_speed = view.rupture_speed
    saved_fitting = list(view.fitting_channels)

    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Pick Arrivals - Event 2"
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"
    # a fresh event comes up with defaults again
    assert view.fitting_channels == OLD_LAYOUT_FITTING
    assert len(set(view.picked_idx)) == 1
    assert np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    assert len(view.fitting_markers) == 4

    # going back restores what was saved
    view.set_event(1)
    assert view.picked_idx == saved_picks
    assert view.fitting_channels == saved_fitting
    assert [v.get() for v in view.fitting_channels_mb.items] == [int(f) for f in saved_fitting]
    assert view.rupture_speed == pytest.approx(saved_speed)


def test_switch_to_event_without_strain_is_refused(app_with_strain, view, no_dialogs):
    view.magic()
    picks = list(view.picked_idx)
    speed = view.rupture_speed
    event_2 = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    event_2.pop("strain")

    # programmatic switch: nothing changes, the user is warned instead
    view.set_event(2)
    assert view.event_idx == 1
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[1]")
    assert view.title() == "Pick Arrivals - Event 1"
    assert view.event_combobox.get() == "1"
    assert view.picked_idx == picks
    assert view.rupture_speed == pytest.approx(speed)
    assert len(no_dialogs) == 1 and no_dialogs[0][0] == "No strain data"

    # the same through the combobox: it snaps back to the current event
    view.event_combobox.set("2")
    view.on_event_selected()
    assert view.event_idx == 1
    assert view.event_combobox.get() == "1"
    assert len(no_dialogs) == 2

    # the view is still fully usable
    view.save()
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/rupture_speed") == pytest.approx(speed)
    assert "rupture_speed" not in event_2

    # an event that does have strain data still loads
    view.set_event(3)
    assert view.event_idx == 3
    assert view.title() == "Pick Arrivals - Event 3"


def test_open_on_event_without_strain_raises_and_unregisters(app_with_strain):
    app_with_strain.data_manager.get_data("runs/[0]/events/[0]").pop("strain")
    with pytest.raises(ValueError, match="no strain data"):
        DynamicStrainArrivalPickerView(app_with_strain, 0, 0)
    assert not any(isinstance(w, DynamicStrainArrivalPickerView) for w in app_with_strain.child_windows)


def test_unsaved_channel_edits_do_not_leak_into_data(app_with_strain, view):
    view.save()
    view.enabled_channels_mb.items[1].set(1)
    view.enabled_channels_changed()
    assert view.enabled_channels[1] is True
    assert len(view.not_fitting_markers) == 5
    assert app_with_strain.data_manager.get_data("runs/[0]/events/[1]/strain/enabled_channels")[1] is False


def test_filter_toggle_keeps_picks(view):
    view.magic()
    picks = list(view.picked_idx)
    view.toggle_filter()
    assert view.filtering
    assert view.filter_toggle.cget("text") == "Filter On"
    assert view.picked_idx == picks
    assert np.isfinite(view.rupture_speed)
    view.toggle_filter()
    assert not view.filtering
    assert view.filter_toggle.cget("text") == "Filter Off"


def test_filter_window_is_used_as_typed(view):
    from scipy import signal

    raw = view.event["strain"]["original"]["raw"]
    channel = next(i for i, f in enumerate(view.fitting_channels) if f)

    # an even window is used as typed: the spinbox keeps showing it
    view.filter_window_length.set("50")
    view.toggle_filter()
    assert view.filter_window() == 50
    assert view.filter_window_length.get() == "50"
    x, y = view.lines[channel].get_data()
    filtered = signal.savgol_filter(raw[channel], 50, 2)
    span = filtered.max() - filtered.min()
    expected = filtered * (-12 / span) + view.event["strain"]["locations"][channel]
    np.testing.assert_allclose(y, expected)

    # a value savgol cannot use is clamped, and the widget shows what was applied
    view.filter_window_length.set("1")
    view.on_filter_window_length_box_changed()
    assert view.filter_window() == 3
    assert view.filter_window_length.get() == "3"

    # garbage falls back to the default; the widget is corrected while filtering
    view.filter_window_length.set("abc")
    view.on_filter_window_length_box_changed()
    assert view.filter_window_length.get() == "51"

    # with the filter off the spinbox is left alone
    view.toggle_filter()
    view.filter_window_length.set("50")
    view.on_filter_window_length_box_changed()
    assert view.filter_window_length.get() == "50"
    x, y = view.lines[channel].get_data()
    span = raw[channel].max() - raw[channel].min()
    np.testing.assert_allclose(y, raw[channel] * (-12 / span) + view.event["strain"]["locations"][channel])


class _FakeMouse:
    def __init__(self, x, y):
        self.xdata = x
        self.ydata = y


class _FakePick:
    def __init__(self, artist, x, y):
        self.artist = artist
        self.mouseevent = _FakeMouse(x, y)


def test_drag_moves_pick_and_is_ignored_in_toolbar_mode(view):
    marker = view.fitting_markers[0]
    channel = int(marker.get_label())
    x, y = view.lines[channel].get_data()
    target = len(x) // 4
    cx, cy = marker.center

    # toolbar in pan/zoom: pick and press are ignored
    view.toolbar.mode = "pan/zoom"
    try:
        view.on_press(_FakeMouse(cx, cy))
        view.on_pick(_FakePick(marker, cx, cy))
        assert view.current_artist is None
        assert not view.currently_dragging
        view.on_motion(_FakeMouse(x[target], y[target]))
        assert view.picked_idx[channel] != target
    finally:
        view.toolbar.mode = ""

    # normal mode: drag snaps the marker to the nearest sample
    view.on_press(_FakeMouse(cx, cy))
    view.on_pick(_FakePick(marker, cx, cy))
    assert view.current_artist is marker
    view.on_motion(_FakeMouse(x[target], y[target]))
    assert view.picked_idx[channel] == target
    assert marker.center == pytest.approx((x[target], y[target]))
    assert np.isfinite(view.rupture_speed)
    view.on_release(None)
    assert view.current_artist is None
    assert not view.currently_dragging


def test_close_unregisters(app_with_strain):
    v = DynamicStrainArrivalPickerView(app_with_strain, 0, 0)
    assert v in app_with_strain.child_windows
    v.on_close()
    assert v not in app_with_strain.child_windows
