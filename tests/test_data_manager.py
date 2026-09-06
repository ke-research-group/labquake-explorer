import numpy as np
import pytest

from labquake_explorer.data.data_manager import DataManager
from tests.synthetic import make_experiment


@pytest.fixture
def dm():
    data, _ = make_experiment(recurrence=12.0)
    manager = DataManager()
    manager.data = data
    return manager


def test_get_data_paths(dm):
    assert dm.get_data("") is dm.data
    assert dm.get_data("name") == "p0001"
    assert dm.get_data("runs/[0]/events/[1]") is dm.data["runs"][0]["events"][1]
    assert dm.get_data("runs\\[0]\\name") == "run00"
    with pytest.raises(KeyError):
        dm.get_data("runs/[0]/nope")
    with pytest.raises(IndexError):
        dm.get_data("runs/[9]")


def test_set_data_add_key_creates_parents(dm):
    dm.set_data("runs/[0]/events/[1]/event_analysis", {"a": 1}, True)
    assert dm.get_data("runs/[0]/events/[1]/event_analysis") == {"a": 1}
    dm.set_data("runs/[0]/derived/deep/value", 3.0, add_key=True)
    assert dm.get_data("runs/[0]/derived/deep/value") == 3.0
    with pytest.raises(KeyError):
        dm.set_data("runs/[0]/missing/value", 1.0)
    with pytest.raises(ValueError):
        dm.set_data("", 1.0)


def test_delete_data(dm):
    dm.set_data("runs/[0]/tmp", 1, add_key=True)
    dm.delete_data("runs/[0]/tmp")
    with pytest.raises(KeyError):
        dm.get_data("runs/[0]/tmp")


def test_no_data_loaded():
    manager = DataManager()
    with pytest.raises(ValueError):
        manager.get_data("name")
    with pytest.raises(ValueError):
        manager.set_data("name", "x")


def test_npz_round_trip(tmp_path, dm):
    path = tmp_path / "exp.npz"
    dm.set_data("runs/[0]/events/[0]/event_analysis", {"version": 2, "stress_drop": 0.4}, True)
    dm.save_file(path)
    other = DataManager()
    other.load_file(path)
    assert other.get_data("runs/[0]/events/[0]/event_analysis")["stress_drop"] == 0.4
    np.testing.assert_array_equal(other.get_data("runs/[0]/shear_stress"), dm.get_data("runs/[0]/shear_stress"))


def _assert_equivalent(a, b, path=""):
    """Recursively compare loaded data with the original (arrays vs lists allowed)."""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            _assert_equivalent(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple, np.ndarray)) and not isinstance(a, str):
        if len(a) and isinstance(a[0], dict):
            assert isinstance(b, list), path
            assert len(a) == len(b), path
            for x, y in zip(a, b):
                _assert_equivalent(x, y, path)
        else:
            x = np.asarray(a)
            y = np.asarray(b)
            if x.dtype.kind in "US":
                assert [str(v) for v in np.ravel(x)] == [str(v) for v in np.ravel(y)], path
            else:
                np.testing.assert_allclose(x.astype(float), y.astype(float), equal_nan=True, err_msg=path)
    elif isinstance(a, str):
        assert a == b, path
    elif isinstance(a, (bool, np.bool_)):
        assert int(a) == int(b), path
    else:
        np.testing.assert_allclose(float(a), float(b), equal_nan=True, err_msg=path)


def test_hdf5_round_trip_of_nested_results(tmp_path, dm):
    nan = float("nan")
    dm.set_data("runs/[0]/events/[0]/event_analysis", {
        "version": 2, "x_field": "displacement", "fit_method": "ols",
        "loading_indices": [10, 20], "post_indices": [30, 40],
        "loading_window": [-4.0, -1.0], "stress_drop": 0.4, "displacement_trend": nan,
        "loading_fit": {"slope": 1.5, "valid": True, "reason": ""},
    }, True)
    dm.set_data("runs/[0]/events/[1]/pzt_spectrum", {
        "version": 1,
        "channels": {"ch0": {"channel": 0, "pre_ms": 0.5, "n_fixed": True,
                             "spectrum": {"f_binned": [1e3, 2e3], "snr": [5.0, nan], "counts": [3, 4]},
                             "fit": {"band_used": [1e3, 2e3], "at_bounds": {"ln_fc": False}, "converged": True},
                             "warnings": ["near node", "kR < 3"], "source": None}},
    }, True)
    dm.set_data("runs/[0]/interevent", {"version": 1, "recurrence": [nan, 12.0, 12.0], "lp_field": "LP_displacement"}, True)
    dm.set_data("runs/[0]/events/[2]/czm_parms", [1.0, 2.0, 3.0, 4.0, -0.1, 0.0, -0.2, 0.2], True)
    original = dm.data

    path = tmp_path / "exp.h5"
    dm.save_file(path)
    other = DataManager()
    other.load_file(path)

    assert isinstance(other.get_data("runs"), list) and isinstance(other.get_data("runs/[0]/events"), list)
    assert len(other.get_data("runs/[0]/events")) == len(original["runs"][0]["events"])
    loaded = other.get_data("runs/[0]/events/[0]/event_analysis")
    assert loaded["version"] == 2 and loaded["x_field"] == "displacement"
    assert list(loaded["loading_indices"]) == [10, 20]
    assert loaded["loading_fit"]["valid"] == 1
    rec = other.get_data("runs/[0]/events/[1]/pzt_spectrum/channels/ch0")
    assert list(rec["fit"]["band_used"]) == [1e3, 2e3]
    assert list(rec["warnings"]) == ["near node", "kR < 3"]
    assert "source" not in rec  # None cannot be stored in HDF5
    np.testing.assert_allclose(rec["spectrum"]["snr"], [5.0, nan], equal_nan=True)
    assert list(other.get_data("runs/[0]/events/[2]/czm_parms")) == [1.0, 2.0, 3.0, 4.0, -0.1, 0.0, -0.2, 0.2]

    # everything else matches the original
    del original["runs"][0]["events"][1]["pzt_spectrum"]["channels"]["ch0"]["source"]
    _assert_equivalent(original, other.data)

    # lists survive deletion by index after a load
    other.delete_data("runs/[0]/events/[0]")
    assert len(other.get_data("runs/[0]/events")) == len(original["runs"][0]["events"]) - 1


def test_hdf5_non_contiguous_digit_keys_stay_a_dict(tmp_path, dm):
    dm.set_data("runs/[0]/legacy", {"0": {"a": 1}, "2": {"a": 3}}, True)
    path = tmp_path / "exp.h5"
    dm.save_file(path)
    other = DataManager()
    other.load_file(path)
    assert other.get_data("runs/[0]/legacy") == {"0": {"a": 1}, "2": {"a": 3}}
