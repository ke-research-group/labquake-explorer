"""Per-event mechanical metrics from picked sample ranges.

All inputs are the event's own arrays (time, X such as displacement, Y such
as shear stress) and integer sample indices picked by the user.  Sign
conventions, documented once here and stored with every result:

* ``stress_drop``       = Y(rupture start) - Y(rupture end)   (drop positive)
* ``displacement``      = X(rupture end)   - X(rupture start) (slip positive)
* ``stress_drop_trend`` = pre-trend(t_ref) - post-trend(t_ref), where the pre
  trend is a line fitted to Y(t) over the loading range and the post trend a
  line fitted over the post range; both are extrapolated to ``t_ref``
  (default: the event time).  This removes interseismic loading and
  post-seismic relaxation from the drop estimate.
* ``displacement_trend`` = post-trend_X(t_ref) - pre-trend_X(t_ref), the same
  construction applied to X (creep-corrected coseismic slip).
* Stiffnesses are signed slopes dY/dX of a straight-line fit over the
  loading and unloading ranges.

Times stored in the result are relative to the event time so the analysis can
be reproduced after re-extracting events with a different window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from labquake_explorer.analysis.fitting import LinearFit, linear_fit, _invalid

RESULT_VERSION = 2


@dataclass(frozen=True)
class EventPicks:
    """Sample indices picked on an event slice (inclusive ranges)."""
    loading: tuple[int, int]
    unloading: tuple[int, int]
    rupture: tuple[int, int]            # (start, end)
    post: Optional[tuple[int, int]] = None

    @classmethod
    def from_list(cls, idx: Sequence[int]) -> "EventPicks":
        idx = [int(i) for i in idx]
        if len(idx) not in (6, 8):
            raise ValueError("expected 6 or 8 picked indices")
        post = (idx[6], idx[7]) if len(idx) == 8 else None
        return cls((idx[0], idx[1]), (idx[2], idx[3]), (idx[4], idx[5]), post)

    def to_list(self) -> list[int]:
        out = [*self.loading, *self.unloading, *self.rupture]
        if self.post is not None:
            out += [*self.post]
        return out

    @staticmethod
    def defaults(n: int, with_post: bool = True) -> "EventPicks":
        """Default marker positions as fractions of the slice length."""
        f = lambda p: min(max(int(n * p), 0), max(n - 1, 0))
        return EventPicks(
            loading=(f(0.25), f(0.35)),
            unloading=(f(0.5), f(0.6)),
            rupture=(f(0.4), f(0.7)),
            post=(f(0.8), f(0.9)) if with_post else None,
        )

    def clipped(self, n: int) -> "EventPicks":
        c = lambda i: min(max(int(i), 0), max(n - 1, 0))
        return EventPicks(
            (c(self.loading[0]), c(self.loading[1])),
            (c(self.unloading[0]), c(self.unloading[1])),
            (c(self.rupture[0]), c(self.rupture[1])),
            None if self.post is None else (c(self.post[0]), c(self.post[1])),
        )


def _slice(pair: tuple[int, int]) -> slice:
    a, b = sorted(int(i) for i in pair)
    return slice(a, b + 1)


@dataclass(frozen=True)
class TrendDrop:
    pre: LinearFit
    post: LinearFit
    t_ref: float
    drop: float            # pre(t_ref) - post(t_ref)
    valid: bool

    @property
    def pre_at_ref(self) -> float:
        return float(self.pre(self.t_ref)) if self.pre.valid else float("nan")

    @property
    def post_at_ref(self) -> float:
        return float(self.post(self.t_ref)) if self.post.valid else float("nan")


def trend_drop(t, y, pre: slice | tuple[int, int], post: slice | tuple[int, int],
               t_ref: float = 0.0, method: str = "ols") -> TrendDrop:
    """Fit lines to ``y(t)`` over two index ranges and difference them at ``t_ref``.

    ``t`` should already be relative to the reference (so ``t_ref=0`` is the
    event time).  ``pre``/``post`` are inclusive index pairs or slices.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    pre_s = pre if isinstance(pre, slice) else _slice(pre)
    post_s = post if isinstance(post, slice) else _slice(post)
    pre_fit = linear_fit(t[pre_s], y[pre_s], method)
    post_fit = linear_fit(t[post_s], y[post_s], method)
    valid = pre_fit.valid and post_fit.valid
    drop = float(pre_fit(t_ref) - post_fit(t_ref)) if valid else float("nan")
    return TrendDrop(pre_fit, post_fit, float(t_ref), drop, valid)


def analyze_event(time, x, y, event_time: float, picks: EventPicks,
                  method: str = "ols", x_field: str = "", y_field: str = "") -> dict:
    """Compute the EventAnalyzerView result dict (schema version 2).

    ``time``, ``x``, ``y`` are the event-slice arrays; ``event_time`` is the
    absolute event time used as the trend reference.  The returned dict is
    JSON-like (floats, ints, lists, small nested dicts) and safe to store
    under ``event['event_analysis']``.
    """
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = time.size
    if not (x.size == y.size == n):
        raise ValueError("time, x and y must have the same length")
    if n == 0:
        raise ValueError("empty event")
    picks = picks.clipped(n)
    t_rel = time - float(event_time)

    loading = linear_fit(x[_slice(picks.loading)], y[_slice(picks.loading)], method)
    unloading = linear_fit(x[_slice(picks.unloading)], y[_slice(picks.unloading)], method)

    i4, i5 = picks.rupture
    stress_drop = float(y[i4] - y[i5])
    displacement = float(x[i5] - x[i4])

    if picks.post is not None:
        td_y = trend_drop(t_rel, y, picks.loading, picks.post, 0.0, method)
        td_x = trend_drop(t_rel, x, picks.loading, picks.post, 0.0, method)
    else:
        nan_fit = _invalid(method, 0, "no post range")
        td_y = TrendDrop(nan_fit, nan_fit, 0.0, float("nan"), False)
        td_x = td_y

    def window(pair):
        a, b = sorted(pair)
        return [float(t_rel[a]), float(t_rel[b])]

    result = {
        "version": RESULT_VERSION,
        "x_field": x_field,
        "y_field": y_field,
        "fit_method": method,
        "event_time": float(event_time),
        # picks (sample indices on the event slice) and the same as times
        # relative to event_time
        "loading_indices": list(picks.loading),
        "unloading_indices": list(picks.unloading),
        "rupture_start_index": int(picks.rupture[0]),
        "rupture_end_index": int(picks.rupture[1]),
        "post_indices": list(picks.post) if picks.post is not None else None,
        "loading_window": window(picks.loading),
        "unloading_window": window(picks.unloading),
        "rupture_window": window(picks.rupture),
        "post_window": window(picks.post) if picks.post is not None else None,
        # stiffness: slope of Y vs X
        "loading_stiffness": loading.slope,
        "loading_fit": loading.as_dict(),
        "unloading_stiffness": unloading.slope,
        "unloading_fit": unloading.as_dict(),
        # peak-to-trough on picked samples
        "stress_drop": stress_drop,
        "displacement": displacement,
        # trend-extrapolated (pre = loading range, post = post range) at t_ref = event_time
        "stress_drop_trend": td_y.drop,
        "displacement_trend": -td_x.drop if td_x.valid else float("nan"),
        "pre_trend": td_y.pre.as_dict(),
        "post_trend": td_y.post.as_dict(),
        "pre_trend_x": td_x.pre.as_dict(),
        "post_trend_x": td_x.post.as_dict(),
        "sign_convention": "stress drop positive; displacement (slip) positive",
    }
    return result


def picks_from_result(result: dict, n: int) -> Optional[EventPicks]:
    """Recover picks from a saved result (version 1 or 2), or None."""
    if not isinstance(result, dict):
        return None
    try:
        idx = [
            result["loading_indices"][0], result["loading_indices"][1],
            result["unloading_indices"][0], result["unloading_indices"][1],
            result["rupture_start_index"], result["rupture_end_index"],
        ]
    except (KeyError, IndexError, TypeError):
        return None
    post = result.get("post_indices")
    if post is not None and len(post) == 2:
        idx += [post[0], post[1]]
    picks = EventPicks.from_list(idx)
    if picks.post is None:
        picks = EventPicks(picks.loading, picks.unloading, picks.rupture,
                           EventPicks.defaults(n).post)
    if any(i >= n or i < 0 for i in picks.to_list()):
        return None
    return picks
