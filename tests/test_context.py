import numpy as np
import pytest

from labquake_explorer.ui import context as C
from labquake_explorer.ui.context import resolve_context, split_path


def test_split_path_handles_separators_and_empties():
    assert split_path("runs/[0]/events/[3]/shear_stress") == ["runs", "[0]", "events", "[3]", "shear_stress"]
    assert split_path("runs\\[0]\\events") == ["runs", "[0]", "events"]
    assert split_path("") == []


@pytest.mark.parametrize(
    "path,value,kind,run_idx,event_idx",
    [
        ("", None, C.ROOT, None, None),
        ("name", "p5993", C.STRING, None, None),
        ("time", np.arange(10), C.ARRAY, None, None),
        ("runs", None, C.OTHER, None, None),
        ("runs/[2]", {}, C.RUN, 2, None),
        ("runs/[2]/shear_stress", np.arange(10), C.RUN_ARRAY, 2, None),
        ("runs/[2]/event_indices", [1, 2, 3], C.EVENT_INDICES, 2, None),
        ("runs/[2]/events", [{}, {}], C.EVENTS, 2, None),
        ("runs/[2]/name", "run02", C.STRING, 2, None),
        ("runs/[2]/strain", {}, C.OTHER, 2, None),
        ("runs/[2]/events/[5]", {}, C.EVENT, 2, 5),
        ("runs/[2]/events/[5]/shear_stress", np.arange(10), C.EVENT_ARRAY, 2, 5),
        ("runs/[2]/events/[5]/event_time", 3.5, C.EVENT, 2, 5),
        ("runs/[2]/events/[5]/czm_parms", {}, C.EVENT, 2, 5),
        ("runs/[2]/events/[5]/strain/original/raw", np.zeros((16, 10)), C.OTHER, 2, 5),
    ],
)
def test_resolve_context_kinds(path, value, kind, run_idx, event_idx):
    ctx = resolve_context(path, value)
    assert ctx.kind == kind
    assert ctx.run_idx == run_idx
    assert ctx.event_idx == event_idx
    assert ctx.path == path


def test_context_paths():
    ctx = resolve_context("runs/[1]/events/[4]/displacement", np.arange(5))
    assert ctx.key == "displacement"
    assert ctx.run_path == "runs/[1]"
    assert ctx.event_path == "runs/[1]/events/[4]"
    assert ctx.parent_path == "runs/[1]/events/[4]"
    assert resolve_context("runs/[1]", {}).event_path is None


def test_single_element_array_is_not_an_array():
    assert resolve_context("runs/[0]/scalar", np.array([1.0])).kind == C.OTHER
    assert resolve_context("runs/[0]/events/[0]/scalar", np.array([1.0])).kind == C.EVENT


def test_list_of_dicts_is_not_an_array():
    assert resolve_context("runs/[0]/events", [{"a": 1}, {"b": 2}]).kind == C.EVENTS
    assert resolve_context("runs/[0]/things", [{"a": 1}, {"b": 2}]).kind == C.OTHER
