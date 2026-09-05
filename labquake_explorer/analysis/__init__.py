"""Pure numerical analysis for Labquake Explorer.

Functions here take arrays in and return dataclasses or dicts; they know
nothing about Tk, the data tree, or persistence.  Views call them and decide
what to display and save.
"""
from labquake_explorer.analysis.fitting import LinearFit, linear_fit
from labquake_explorer.analysis.event_metrics import (
    EventPicks, TrendDrop, analyze_event, trend_drop, RESULT_VERSION,
)

__all__ = [
    "LinearFit", "linear_fit",
    "EventPicks", "TrendDrop", "analyze_event", "trend_drop", "RESULT_VERSION",
]
