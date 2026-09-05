import numpy as np

from labquake_explorer.ui import context as C
from labquake_explorer.ui.views import EventAnalyzerView


def labels(app, path):
    item = app.find_item(path)
    assert item is not None, path
    ctx = app.context_at(item)
    menu = app.build_context_menu(ctx)
    if menu is None:
        return ctx, []
    n = menu.index("end")
    return ctx, [menu.entrycget(i, "label") for i in range(n + 1)]


def test_find_item_and_context(app):
    ctx, _ = labels(app, "runs/[0]/events/[1]/shear_stress")
    assert ctx.kind == C.EVENT_ARRAY
    assert (ctx.run_idx, ctx.event_idx, ctx.key) == (0, 1, "shear_stress")


def test_context_menus_from_registry(app):
    assert labels(app, "runs/[0]/shear_stress")[1] == ["Pick Events", "Pick Indices", "Extract Slopes"]
    assert labels(app, "runs/[0]/event_indices")[1] == ["Extract Events"]
    assert labels(app, "runs/[0]/events/[0]")[1] == ["Analyze Event", "Pick Arrivals", "Fit Cohesive Zone Model"]
    assert labels(app, "runs/[0]/events/[0]/event_time")[1] == ["Analyze Event", "Pick Arrivals", "Fit Cohesive Zone Model"]
    assert labels(app, "runs/[0]/events/[0]/shear_stress")[1] == ["Pick Indices", "Extract Slopes", "Min/Max"]
    assert labels(app, "name")[1] == ["Edit String"]
    assert labels(app, "runs/[0]/name")[1] == ["Edit String"]
    assert labels(app, "runs")[1] == []
    assert "Inter-event Metrics" in labels(app, "runs/[0]")[1]


def test_run_action_opens_registered_view(app):
    ctx, menu_labels = labels(app, "runs/[0]/events/[1]")
    from labquake_explorer.ui.actions import actions_for
    action = [a for a in actions_for(ctx.kind) if a.label == "Analyze Event"][0]
    app.run_action(action, ctx)
    assert len(app.child_windows) == 1
    view = app.child_windows[0]
    assert isinstance(view, EventAnalyzerView)
    assert (view.run_idx, view.event_idx) == (0, 1)
    view.destroy()


def test_double_click_run_opens_signal_overlay(app):
    from labquake_explorer.ui.views import RunSignalsView
    item = app.find_item("runs/[0]")
    app.data_tree.selection_set(item)
    app.on_double_click(None)
    assert isinstance(app.child_windows[-1], RunSignalsView)
    assert "Plot Run Signals" in labels(app, "runs/[0]")[1]
    app.child_windows[-1].on_close()


def test_double_click_array_opens_simple_plot_once(app):
    from labquake_explorer.ui.views import SimplePlotView
    item = app.find_item("runs/[0]/shear_stress")
    app.data_tree.selection_set(item)
    app.on_double_click(None)
    views = [w for w in app.child_windows if isinstance(w, SimplePlotView)]
    assert len(views) == 1 and app.child_windows.count(views[0]) == 1
    views[0].on_close()
    assert views[0] not in app.child_windows
