"""Per-event metrics from picked sample ranges.

The inputs are two aligned series of the event (``x`` and ``y``, any fields
the user chose) and six sample indices picked on them:

* ``loading_indices``   (i0, i1): ``loading_slope``   = dY/dX over the range, by least squares
* ``unloading_indices`` (i2, i3): ``unloading_slope`` = dY/dX over the range
* ``delta_indices``     (i4, i5): ``delta_x`` = X[i5] - X[i4] and ``delta_y`` = Y[i5] - Y[i4]

Nothing is interpreted here.  With X = fault slip and Y = shear stress the
slopes are stiffnesses, ``delta_x`` the coseismic slip and ``-delta_y`` the
stress drop; on other fields they are whatever those fields make them.  The
result records ``x_field`` and ``y_field`` so a reader can tell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from labquake_explorer.analysis.fitting import linear_fit

RESULT_VERSION = 3
RANGES = ("loading", "unloading", "delta")


@dataclass(frozen=True)
class EventPicks:
    """Sample indices picked on an event slice: two inclusive ranges and the
    two samples that are differenced."""
    loading: tuple[int, int]
    unloading: tuple[int, int]
    delta: tuple[int, int]              # (start, end)

    @classmethod
    def from_list(cls, idx: Sequence[int]) -> "EventPicks":
        idx = [int(i) for i in idx]
        if len(idx) < 6:
            raise ValueError("expected six picked indices")
        return cls((idx[0], idx[1]), (idx[2], idx[3]), (idx[4], idx[5]))     # a legacy post range (6, 7) is dropped

    def to_list(self) -> list[int]:
        return [*self.loading, *self.unloading, *self.delta]

    @staticmethod
    def defaults(n: int) -> "EventPicks":
        """Default marker positions as fractions of the slice length."""
        f = lambda p: min(max(int(n * p), 0), max(n - 1, 0))
        return EventPicks(loading=(f(0.25), f(0.35)), unloading=(f(0.5), f(0.6)), delta=(f(0.4), f(0.7)))

    def clipped(self, n: int) -> "EventPicks":
        c = lambda i: min(max(int(i), 0), max(n - 1, 0))
        return EventPicks((c(self.loading[0]), c(self.loading[1])),
                          (c(self.unloading[0]), c(self.unloading[1])),
                          (c(self.delta[0]), c(self.delta[1])))

    def pairs(self) -> dict:
        return {"loading": self.loading, "unloading": self.unloading, "delta": self.delta}


def _slice(pair: tuple[int, int]) -> slice:
    a, b = sorted(int(i) for i in pair)
    return slice(a, b + 1)


def analyze_event(x, y, picks: EventPicks, x_field: str = "", y_field: str = "") -> dict:
    """The EventAnalyzerView result dict (schema version 3) for ``y`` against ``x``.

    JSON-like (floats, ints, lists) and safe to store under
    ``event['event_analysis']``.  Degenerate ranges give NaN slopes, never an
    exception.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length")
    n = x.size
    if n == 0:
        raise ValueError("empty event")
    picks = picks.clipped(n)
    i4, i5 = picks.delta
    return {
        "version": RESULT_VERSION,
        "x_field": x_field,
        "y_field": y_field,
        "loading_indices": list(picks.loading),
        "unloading_indices": list(picks.unloading),
        "delta_indices": list(picks.delta),
        "loading_slope": linear_fit(x[_slice(picks.loading)], y[_slice(picks.loading)]).slope,
        "unloading_slope": linear_fit(x[_slice(picks.unloading)], y[_slice(picks.unloading)]).slope,
        "delta_x": float(x[i5] - x[i4]),
        "delta_y": float(y[i5] - y[i4]),
    }


def picks_from_result(result, n: int) -> Optional[EventPicks]:
    """Recover the picks of a saved result (version 1, 2 or 3), or None when
    the record has none or they fall outside a slice of ``n`` samples."""
    if not isinstance(result, dict):
        return None
    try:
        if "delta_indices" in result:
            delta = (int(result["delta_indices"][0]), int(result["delta_indices"][1]))
        else:
            delta = (int(result["rupture_start_index"]), int(result["rupture_end_index"]))
        picks = EventPicks((int(result["loading_indices"][0]), int(result["loading_indices"][1])),
                           (int(result["unloading_indices"][0]), int(result["unloading_indices"][1])), delta)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if any(i >= n or i < 0 for i in picks.to_list()):
        return None
    return picks


def windows_from_picks(t_rel, picks: EventPicks) -> dict:
    """The picks as times relative to the event, ``{range: (t_a, t_b)}``, for
    placing the same ranges on another event of the run."""
    t_rel = np.asarray(t_rel, dtype=float)
    if t_rel.size == 0:
        raise ValueError("empty event")
    picks = picks.clipped(t_rel.size)
    return {name: (float(t_rel[a]), float(t_rel[b])) for name, (a, b) in picks.pairs().items()}


def picks_from_windows(t_rel, windows: Mapping[str, Sequence[float]]) -> EventPicks:
    """Relative-time ranges (from :func:`windows_from_picks`) back to sample
    picks on ``t_rel``; each endpoint snaps to the nearest sample."""
    t_rel = np.asarray(t_rel, dtype=float)
    if t_rel.size == 0:
        raise ValueError("empty event")

    def pair(window):
        return (int(np.argmin(np.abs(t_rel - float(window[0])))), int(np.argmin(np.abs(t_rel - float(window[1])))))

    return EventPicks(pair(windows["loading"]), pair(windows["unloading"]), pair(windows["delta"]))


__all__ = ["RESULT_VERSION", "RANGES", "EventPicks", "analyze_event", "picks_from_result",
           "windows_from_picks", "picks_from_windows"]
