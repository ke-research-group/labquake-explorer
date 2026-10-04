import numpy as np
import pytest

from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data import sources as S
from tests.synthetic import write_ni_npz, write_tpc5


def pulse(t0, width=1e-4, amp=0.1):
    return lambda t: amp * np.exp(-((t - t0) / width) ** 2)


ELSYS_SIGNALS = {
    "A1 (0.A1)": lambda t: pulse(0.5)(t) + pulse(1.4)(t),
    "A5 (0.A5)": lambda t: 2.0 + 0.0 * t,
    "A7 (0.A7)": lambda t: 0.1 * t,
}
ELSYS_MAP = {"A1": "pzt_1", "A5": "pressure_1", "A7": "lvdt"}


@pytest.fixture
def elsys(tmp_path):
    path = tmp_path / "t0001_01_8MPa_run1.tpc5"
    write_tpc5(path, ELSYS_SIGNALS, trigger_times=(0.5, 1.4), volt_range=12.0)
    return path


def test_tpc5_source_fields_blocks_and_history(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.0)
    assert src.fields == ["pzt_1", "pressure_1", "lvdt"]
    assert src.channel_numbers == [1, 2, 3] and src.channel_names[0] == "A1 (0.A1)"
    assert src.continuous.block == 1 and [b.block for b in src.trigger_blocks] == [2, 3]
    assert src.trigger_times() == pytest.approx([2.5, 3.4])
    t, d = src.time_history()
    assert d.shape == (3, 4000) and d.dtype == np.float32
    assert t[0] == pytest.approx(2.0) and t[-1] == pytest.approx(2.0 + 3999 / 2000)
    assert d[1].mean() == pytest.approx(2.0, abs=1e-3)
    t2, d2 = src.time_history(decimation=4)
    assert d2.shape == (3, 1000) and t2[0] == pytest.approx(2.0 + 1.5 / 2000)


def test_tpc5_read_window_on_run_clock(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.0)
    win = src.read_window(2.499, 2.503)
    assert win is not None and win.block == 2 and win.sample_rate == 200_000.0
    assert win.fields == ["pzt_1", "pressure_1", "lvdt"]
    assert win.time[0] == pytest.approx(2.499, abs=1e-5) and win.time[-1] == pytest.approx(2.503, abs=1e-5)
    assert win.time[np.argmax(win.data[0])] == pytest.approx(2.5, abs=2e-5)   # pulse at the trigger, shifted
    # clipped to the block, subset of fields, and None away from every trigger block
    win = src.read_window(2.0, 3.0, ["lvdt"])
    assert win.fields == ["lvdt"] and win.time[0] == pytest.approx(2.498, abs=1e-5) and win.time[-1] == pytest.approx(2.508, abs=1e-5)
    assert src.read_window(3.0, 3.01) is None
    with pytest.raises(KeyError):
        src.read_window(2.499, 2.503, ["nope"])


def test_tpc5_event_window_uses_source_settings(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, event_fields=["pzt_1"], event_window_s=(0.001, 0.002))
    win = src.event_window(0.5, pre=5.0, post=5.0)
    assert win.fields == ["pzt_1"]
    assert win.time.size == pytest.approx(0.003 * 200_000 + 1, abs=1)
    assert win.data.dtype == np.float32


def test_tpc5_reference_round_trip(elsys, tmp_path):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.389, event_fields=["pzt_1"],
                            event_window_s=(0.1, 0.5), meta={"clock_offset_method": "trigger"})
    ref = src.to_reference(tmp_path)
    assert ref["format"] == "tpc5" and ref["filename"] == elsys.name and ref["time_offset"] == 2.389
    assert ref["blocks"]["trigger_time_run"] == pytest.approx([2.889, 3.789])
    assert ref["clock_offset_method"] == "trigger"
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [{"name": "run1", "time": np.arange(3.0), "strain": ref}]}
    out = tmp_path / "exp.h5"
    dm.save_file(out)
    back = DataManager()
    back.load_file(out)
    run = back.get_data("runs/[0]")
    assert S.reference_format(run["strain"]) == "tpc5" and S.run_sources(run) == {"strain": run["strain"]}
    again = S.open_source(run["strain"], tmp_path, run)
    assert isinstance(again, S.Tpc5Source)
    assert again.fields == src.fields and again.time_offset == 2.389 and again.event_fields == ["pzt_1"]
    assert again.event_window_s == (0.1, 0.5) and again.meta["clock_offset_method"] == "trigger"
    assert [b.block for b in again.trigger_blocks] == [2, 3] and again.continuous.n_samples == 4000
    a = src.read_window(2.888, 2.892)
    b = again.read_window(2.888, 2.892)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_allclose(a.time, b.time)


def test_ni_source(tmp_path):
    path = tmp_path / "T0001-raw-run1-20260101_000000.npz"
    write_ni_npz(path, {"n": lambda t: 3.5 + 0 * t, "s": lambda t: 0.7 + 0 * t, "e": lambda t: -0.5 - 2.0 * t},
                 fs=10_000.0, duration=2.0, trigger_sample_index=12_000)
    src = S.NINpzSource.open(path, {"ai0": "pressure_1", "ai1": "pressure_2", "ai2": "eddy_1"},
                             decimation=5, time_offset=0.0, meta={"input_range_v": 10.0})
    assert src.fields == ["pressure_1", "pressure_2", "eddy_1"]
    assert src.trigger_times() == pytest.approx([1.2])
    t, d = src.time_history()
    assert d.shape == (3, 4000) and t[1] - t[0] == pytest.approx(5 / 10_000)
    win = src.read_window(1.0, 1.001)
    assert win.sample_rate == 10_000.0 and win.time.size == 11 and win.fields[2] == "eddy_1"
    assert win.data[2, 0] == pytest.approx(-2.5, abs=1e-6)
    assert src.read_window(5.0, 5.1) is None
    ref = src.to_reference(tmp_path)
    assert ref["format"] == "ni_npz" and ref["decimation"] == 5 and ref["input_range_v"] == 10.0
    assert ref["trigger_sample_index"] == 12_000 and ref["sample_rate"] == 10_000.0
    again = S.open_source(ref, tmp_path)
    assert isinstance(again, S.NINpzSource) and again.fields == src.fields and again.decimation == 5
    assert again.meta["input_range_v"] == 10.0
    np.testing.assert_array_equal(again.read_window(1.0, 1.001).data, win.data)


def test_legacy_strain_source(tmp_path):
    path = tmp_path / "legacy.tpc5"
    write_tpc5(path, {"ch1": lambda t: 0.1 * t, "ch2": lambda t: 0.0 * t}, fs_continuous=1000.0,
               duration=2.0, trigger_times=())
    run = {"time": 10.0 + np.arange(0, 2.0, 0.01),
           "strain": {"filename": "legacy.tpc5", "time_offset": 0.25, "time": np.arange(0, 2.0, 0.1),
                      "raw": np.zeros((2, 20)), "filename_downsampled": ""}}
    assert S.reference_format(run["strain"]) == "tpc5_legacy"
    src = S.open_source(run["strain"], tmp_path, run)
    assert isinstance(src, S.LegacyTpc5Source) and src.fields == ["ch1", "ch2"] and src.sample_rate == 1000.0
    win = src.read_window(10.75, 10.76)       # file time 0.5 .. 0.51
    assert win.time[0] == pytest.approx(10.75) and win.time.size == 11
    assert win.data[0, 0] == pytest.approx(0.05, abs=2e-4)
    with pytest.raises(ValueError):
        S.open_source(run["strain"], tmp_path)   # needs the run for its clock


def test_register_source_and_block_mean():
    with pytest.raises(ValueError):
        S.register_source(type("Bad", (S.Source,), {}))
    x = np.arange(10.0)
    np.testing.assert_array_equal(S.block_mean(x, 3), [1.0, 4.0, 7.0])
    np.testing.assert_array_equal(S.block_mean(x, 1), x)
    assert S.block_mean(np.arange(12.0).reshape(2, 6), 2).shape == (2, 3)
    assert S.reference_format({"a": 1}) is None and S.reference_format(3) is None
