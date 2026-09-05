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
