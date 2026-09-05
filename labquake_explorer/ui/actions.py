"""Registry of context-menu actions.

An action is a label plus a callable ``run(app, ctx)`` that applies to one or
more tree-node kinds (see :mod:`labquake_explorer.ui.context`).  The main
window builds its right-click menus from this registry, so adding a view or a
command means registering it here, not editing the main window.

Register a plain function::

    @register_action("Extract Events", kinds=[EVENT_INDICES])
    def extract_events(app, ctx): ...

Register a view class (it must provide ``from_context(app, ctx)``)::

    @register_view("Analyze Event", kinds=[EVENT])
    class EventAnalyzerView(EventView): ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from labquake_explorer.ui.context import KINDS, TreeContext

ActionCallable = Callable[[Any, TreeContext], Any]


@dataclass(frozen=True)
class Action:
    label: str
    kinds: frozenset
    run: ActionCallable
    order: int
    name: str


_REGISTRY: list[Action] = []


def _check_kinds(kinds: Iterable[str]) -> frozenset:
    kinds = frozenset(kinds)
    unknown = kinds - set(KINDS)
    if unknown:
        raise ValueError(f"unknown tree kinds: {sorted(unknown)}")
    if not kinds:
        raise ValueError("an action needs at least one kind")
    return kinds


def add_action(action: Action) -> Action:
    """Add or replace (by name) an action."""
    for i, existing in enumerate(_REGISTRY):
        if existing.name == action.name:
            _REGISTRY[i] = action
            return action
    _REGISTRY.append(action)
    return action


def register_action(label: str, kinds: Iterable[str], order: int = 100):
    """Decorator registering ``func(app, ctx)`` as a menu action."""
    kinds = _check_kinds(kinds)

    def decorator(func: ActionCallable) -> ActionCallable:
        add_action(Action(label=label, kinds=kinds, run=func, order=order, name=func.__qualname__))
        return func

    return decorator


def register_view(label: str, kinds: Iterable[str], order: int = 100):
    """Decorator registering a view class; opening it constructs ``cls.from_context(app, ctx)``."""
    kinds = _check_kinds(kinds)

    def decorator(cls):
        if not hasattr(cls, "from_context"):
            raise TypeError(f"{cls.__name__} needs a from_context(app, ctx) classmethod")

        def run(app, ctx):
            view = cls.from_context(app, ctx)
            if view is not None and hasattr(app, "register_child"):
                app.register_child(view)
            return view

        add_action(Action(label=label, kinds=kinds, run=run, order=order, name=cls.__qualname__))
        cls.menu_label = label
        cls.menu_kinds = kinds
        return cls

    return decorator


def actions_for(kind: str) -> list[Action]:
    """Actions applicable to a tree kind, in menu order."""
    matching = [(a.order, i, a) for i, a in enumerate(_REGISTRY) if kind in a.kinds]
    return [a for _, _, a in sorted(matching, key=lambda t: (t[0], t[1]))]


def all_actions() -> list[Action]:
    return list(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
