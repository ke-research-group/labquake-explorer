"""Putting two recorders on one clock."""
from __future__ import annotations

from typing import Optional

import numpy as np

from labquake_explorer.data.sources import NINpzSource, Tpc5Source, block_mean


def offset_from_trigger(ni: NINpzSource, elsys: Tpc5Source) -> float:
    """``time_offset`` for the Elsys file so that its first trigger block lands on
    the NI sample that received the trigger pulse: ``t_ni = t_elsys + offset``.

    Both sources are taken on their own file clocks (their ``time_offset`` is
    ignored).
    """
    t_ni = ni.record.trigger_time
    if t_ni is None:
        raise ValueError(f"{ni.path.name}: no trigger_sample_index")
    if not elsys.trigger_blocks:
        raise ValueError(f"{elsys.path.name}: no trigger blocks")
    return float(t_ni - elsys.trigger_blocks[0].trigger_time)


def largest_step(t: np.ndarray, y: np.ndarray, t_center: float, half: float = 1.0, width: float = 0.02):
    """Time and size of the largest change over ``width`` seconds within ``t_center +- half``."""
    m = (t >= t_center - half) & (t <= t_center + half)
    tt, yy = t[m], y[m]
    if tt.size < 10:
        return float("nan"), float("nan")
    dt = float(tt[1] - tt[0])
    w = max(1, int(round(width / dt)))
    d = yy[w:] - yy[:-w]
    i = int(np.argmax(np.abs(d)))
    return float(tt[i] + 0.5 * w * dt), float(d[i])


def slip_step_table(ni: NINpzSource, trigger_times_run, field: str, decimation: int = 250,
                    half: float = 1.0, width: float = 0.02, step_min_v: float = 0.05) -> list:
    """For every trigger time (run clock) the largest step of one NI field nearby.

    Returns one dict per trigger: ``t_run``, ``slip_step`` (bool), ``step_t_run``,
    ``lag_ms`` (step after trigger) and ``dV``.  Slip should follow the trigger by
    tens of milliseconds; a negative lag or no step at all flags a misalignment
    or a trigger without slip.
    """
    rows = []
    for t_trig in trigger_times_run:
        t_trig = float(t_trig)
        win = ni.read_window(t_trig - half, t_trig + half, [field], dtype=np.float64)
        if win is None or win.time.size < 2 * decimation:
            rows.append({"t_run": t_trig, "slip_step": False, "step_t_run": float("nan"),
                         "lag_ms": float("nan"), "dV": float("nan"), "note": "outside the NI record"})
            continue
        td = block_mean(win.time, decimation)
        yd = block_mean(win.data[0], decimation)
        t_step, dv = largest_step(td, yd, t_trig, half, width)
        found = bool(np.isfinite(dv) and abs(dv) >= step_min_v)
        rows.append({"t_run": t_trig, "slip_step": found,
                     "step_t_run": t_step if found else float("nan"),
                     "lag_ms": 1e3 * (t_step - t_trig) if found else float("nan"),
                     "dV": dv, "note": ""})
    return rows


__all__ = ["offset_from_trigger", "largest_step", "slip_step_table"]
