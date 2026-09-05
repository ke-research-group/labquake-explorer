import tkinter as tk

import pytest

from labquake_explorer.ui.views.base import BaseView, EventView, RunView


class Broken(EventView):
    window_title = "Broken"

    def build_ui(self):
        raise RuntimeError("boom")


class BrokenRun(RunView):
    window_title = "BrokenRun"

    def on_run_loaded(self):
        raise KeyError("missing")


class Simple(EventView):
    window_title = "Simple"
    result_key = "simple"
    loads = 0

    def build_ui(self):
        self.build_event_selector(self, row=0, column=0)
        self.make_figure(figsize=(3, 2), figure_kwargs={"layout": "constrained"}, row=1, column=0)

    def on_event_loaded(self):
        self.loads += 1


def test_failed_construction_leaves_no_child(app):
    with pytest.raises(RuntimeError):
        Broken(app, 0, 0)
    assert app.child_windows == []
    with pytest.raises(KeyError):
        BrokenRun(app, 0)
    assert app.child_windows == []


def test_event_view_lifecycle(app):
    v = Simple(app, 0, 1)
    assert v.title() == "Simple - Event 1"
    assert v.loads == 1
    assert v.figure.get_layout_engine() is not None
    assert v.event_combobox.get() == "1" and len(v.event_combobox["values"]) == 4
    v.event_combobox.set("3")
    v.on_event_selected()
    assert v.event_idx == 3 and v.loads == 2 and v.title() == "Simple - Event 3"
    v.save_results({"version": 1, "x": 2})
    assert app.data_manager.get_data("runs/[0]/events/[3]/simple") == {"version": 1, "x": 2}
    assert v.load_results() == {"version": 1, "x": 2}
    v.save_field("strain/original/note", "hi")
    assert app.data_manager.get_data("runs/[0]/events/[3]/strain/original/note") == "hi"
    assert v.figure_title() == "p0001 run00 event3"
    v.on_close()
    assert v not in app.child_windows


def test_unregister_removes_duplicates(app):
    v = BaseView(app, title="x")
    app.child_windows.append(v)
    assert app.child_windows.count(v) == 2
    v.on_close()
    assert v not in app.child_windows


def test_run_view_lifecycle(app):
    class R(RunView):
        window_title = "R"
        result_key = "r"

    v = R(app, 0)
    assert v.title() == "R - run00"
    assert v.run is app.data_manager.get_data("runs/[0]")
    v.save_results({"version": 1})
    assert app.data_manager.get_data("runs/[0]/r") == {"version": 1}
    assert v.load_results() == {"version": 1}
    v.on_close()
    assert v not in app.child_windows
