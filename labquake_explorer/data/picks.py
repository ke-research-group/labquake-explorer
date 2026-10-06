"""Where a run keeps its picked event samples.

``run['event_extraction']`` is the Pick Events form's record: ``event_indices``
(the picks, sample indices on the run's time axis), ``start_s`` and ``end_s``
(the window around each pick), ``x_field`` and ``y_field`` (the series the
form showed) and ``n_events`` (how many events the last extraction wrote).
Older files kept the picks at the run's top level as ``event_indices``;
:func:`picked_indices` reads both, and the form moves them into the record
the next time it saves.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

EXTRACTION_KEY = "event_extraction"
LEGACY_INDICES_KEY = "event_indices"


def extraction_record(run) -> dict:
    """The run's ``event_extraction`` record, or ``{}``."""
    value = run.get(EXTRACTION_KEY) if isinstance(run, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def picked_indices(run) -> list:
    """The run's picks as ints: from the extraction record, else the legacy
    top-level ``event_indices``, else ``[]``."""
    if not isinstance(run, Mapping):
        return []
    raw = extraction_record(run).get(LEGACY_INDICES_KEY)
    if raw is None:
        raw = run.get(LEGACY_INDICES_KEY)
    if raw is None:
        return []
    try:
        return [int(i) for i in np.asarray(raw).ravel()]
    except (TypeError, ValueError):
        return []


__all__ = ["EXTRACTION_KEY", "LEGACY_INDICES_KEY", "extraction_record", "picked_indices"]
