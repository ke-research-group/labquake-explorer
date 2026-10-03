import numpy as np
import pytest

from labquake_explorer.utils.ni_npz import open_ni_npz
from tests.synthetic import write_ni_npz


SIGNALS = {
    "normal": lambda t: 3.5 + 0.0 * t,
    "shear": lambda t: 0.7 + 0.01 * np.sin(2 * np.pi * 3.0 * t),
    "eddy": lambda t: -0.5 - 2.0 * t,
}


@pytest.fixture(params=[False, True], ids=["stored", "compressed"])
def rec(tmp_path, request):
    path = tmp_path / "T0000-raw-run1-20260101_000000.npz"
    truth = write_ni_npz(path, SIGNALS, fs=10_000.0, duration=2.0, trigger_sample_index=12_345,
                         compressed=request.param)
    return open_ni_npz(path), truth, request.param


def test_metadata(rec):
    r, truth, compressed = rec
    assert r.sample_rate == 10_000.0 and r.n_samples == 20_000 and r.duration == pytest.approx(2.0)
    assert r.channels == ["ai0", "ai1", "ai2"] and r.labels == ["normal", "shear", "eddy"]
    assert r.dtype == np.float64
    assert r.trigger_sample_index == 12_345 and r.trigger_time == pytest.approx(1.2345)
    d = r.as_dict()
    assert d["format"] == "ni_npz" and d["channels"] == ["ai0", "ai1", "ai2"] and d["trigger_time"] == pytest.approx(1.2345)
    assert (d["member_offsets"]["ai0"] is None) == compressed


def test_channel_access_is_lazy_and_exact(rec):
    r, truth, compressed = rec
    ch = r.channel("ai2")
    assert isinstance(ch, np.memmap) != compressed
    np.testing.assert_array_equal(np.asarray(ch[100:110]), truth["ai2"][100:110])
    np.testing.assert_array_equal(np.asarray(r.channel(0)[:5]), truth["ai0"][:5])
    with pytest.raises(KeyError):
        r.channel("ai9")
    with pytest.raises(IndexError):
        r.channel(7)


def test_read_and_window(rec):
    r, truth, _ = rec
    t, d = r.read(["shear", "ai2"] if False else [1, "ai2"], 1000, 1010)
    assert d.shape == (2, 10) and t[0] == pytest.approx(0.1)
    np.testing.assert_array_equal(d[0], truth["ai1"][1000:1010])
    t, d = r.read_window(0.5, 0.5005)
    assert d.shape == (3, 6) and t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(0.5005)
    assert r.sample_at(-1.0) == 0 and r.sample_at(99.0) == r.n_samples - 1
    assert r.contains(1.0) and not r.contains(2.5)
    t_all, d_all = r.read(stop=None)
    assert d_all.shape == (3, 20_000)


def test_decimate_is_block_mean(rec):
    r, truth, _ = rec
    t, d = r.decimate(100, channels=["ai1", "ai2"], chunk_samples=700)   # chunk not a multiple of n
    assert d.shape == (2, 200) and d.dtype == np.float32
    expected = truth["ai1"][:20_000].reshape(200, 100).mean(axis=1)
    np.testing.assert_allclose(d[0], expected, rtol=1e-6)
    assert t[0] == pytest.approx((99 / 2) / 10_000.0) and t[1] - t[0] == pytest.approx(0.01)
    t1, d1 = r.decimate(1, channels=[0])
    np.testing.assert_array_equal(d1[0], truth["ai0"].astype(np.float32))
    with pytest.raises(ValueError):
        r.decimate(0)


def test_missing_trigger_and_metadata(tmp_path):
    path = tmp_path / "a.npz"
    write_ni_npz(path, {"x": lambda t: t}, fs=1000.0, duration=0.1)
    r = open_ni_npz(path)
    assert r.trigger_sample_index is None and r.trigger_time is None
    np.savez(tmp_path / "b.npz", ai0=np.zeros(5))
    with pytest.raises(ValueError):
        open_ni_npz(tmp_path / "b.npz")
    np.savez(tmp_path / "c.npz", ai0=np.zeros(5), ai1=np.zeros(6), sample_rate=np.array(1.0))
    with pytest.raises(ValueError):
        open_ni_npz(tmp_path / "c.npz")
    with pytest.raises(FileNotFoundError):
        open_ni_npz(tmp_path / "missing.npz")
