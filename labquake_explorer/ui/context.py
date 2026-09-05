"""Resolve a data-tree path into a typed context for menus and views."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# Tree-node kinds. Actions and views declare which kinds they apply to.
ROOT = "root"
RUN = "run"                      # runs/[i]
RUN_ARRAY = "run_array"          # runs/[i]/<array>
EVENT_INDICES = "event_indices"  # runs/[i]/event_indices
EVENTS = "events"                # runs/[i]/events
EVENT = "event"                  # runs/[i]/events/[j] or a non-array child of it
EVENT_ARRAY = "event_array"      # runs/[i]/events/[j]/<array>
ARRAY = "array"                  # top-level array
STRING = "string"                # any string value
OTHER = "other"

KINDS = (ROOT, RUN, RUN_ARRAY, EVENT_INDICES, EVENTS, EVENT, EVENT_ARRAY, ARRAY, STRING, OTHER)


@dataclass(frozen=True)
class TreeContext:
    """What the user right-clicked on, resolved once from the tree path."""
    path: str
    kind: str
    key: str
    run_idx: Optional[int] = None
    event_idx: Optional[int] = None

    @property
    def run_path(self) -> Optional[str]:
        if self.run_idx is None:
            return None
        return f"runs/[{self.run_idx}]"

    @property
    def event_path(self) -> Optional[str]:
        if self.run_idx is None or self.event_idx is None:
            return None
        return f"runs/[{self.run_idx}]/events/[{self.event_idx}]"

    @property
    def parent_path(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""


def split_path(path: str) -> list[str]:
    """Split a tree path such as 'runs/[0]/events/[3]/shear_stress' into parts."""
    return [p for p in path.replace("\\", "/").split("/") if p]


def _index(part: str) -> Optional[int]:
    if len(part) >= 3 and part[0] == "[" and part[-1] == "]":
        try:
            return int(part[1:-1])
        except ValueError:
            return None
    return None


def _is_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > 1
    if isinstance(value, list):
        return len(value) > 1 and not any(isinstance(v, dict) for v in value)
    return False


def resolve_context(path: str, value: Any = None) -> TreeContext:
    """Classify a tree path (and optionally the value found there).

    The path grammar mirrors the data layout: ``runs/[i]`` is a run,
    ``runs/[i]/events/[j]`` is an event, and everything below an event or a
    run is classified by its value type.  ``value`` is the object at ``path``
    (as returned by ``DataManager.get_data``); pass it when available so that
    arrays and strings are recognized.
    """
    parts = split_path(path)
    key = parts[-1] if parts else ""
    run_idx: Optional[int] = None
    event_idx: Optional[int] = None

    if len(parts) >= 2 and parts[0] == "runs":
        run_idx = _index(parts[1])
    if run_idx is not None and len(parts) >= 4 and parts[2] == "events":
        event_idx = _index(parts[3])

    if not parts:
        kind = ROOT
    elif isinstance(value, str):
        kind = STRING
    elif run_idx is not None and len(parts) == 2:
        kind = RUN
    elif run_idx is not None and len(parts) == 3:
        if key == "event_indices":
            kind = EVENT_INDICES
        elif key == "events":
            kind = EVENTS
        elif _is_array(value):
            kind = RUN_ARRAY
        else:
            kind = OTHER
    elif event_idx is not None and len(parts) == 4:
        kind = EVENT
    elif event_idx is not None and len(parts) == 5:
        kind = EVENT_ARRAY if _is_array(value) else EVENT
    elif len(parts) == 1 and _is_array(value):
        kind = ARRAY
    else:
        kind = OTHER

    return TreeContext(path=path, kind=kind, key=key, run_idx=run_idx, event_idx=event_idx)
