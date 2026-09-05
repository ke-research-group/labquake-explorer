"""Run-level metrics between consecutive events.

Given the run time axis, the event times, and run-level signals, compute for
each event the value of every signal shortly after the event and the change
of that value since the previous event ("per cycle").  Typical use: load-point
advance per cycle from ``LP_displacement`` and fault slip per cycle from
``displacement``; their difference over a cycle is interseismic creep once the
coseismic slip of the event is subtracted.

Sampling is a mean over ``[t + delay, t + delay + width]`` seconds.  A window
that is not fully inside the run gives NaN (never a clamped endpoint value).
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

RESULT_VERSION = 1


def sample_after(time, signal, t: float, delay: float = 0.0, width: float = 0.0) -> float:
    """Mean of ``signal`` over ``[t+delay, t+delay+width]``; NaN if outside the run.

    ``time`` must be non-decreasing.  The window is converted to a fixed
    number of samples ``max(1, round(width / dt))`` starting at the first
    sample at or after ``t + delay`` (within half a sample), so consecutive
    events are averaged over identical sample counts.  With ``width == 0`` the
    single sample at or after ``t + delay`` is returned.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.shape != signal.shape or time.ndim != 1 or time.size == 0:
        raise ValueError("time and signal must be equal-length 1-D arrays")
    if width < 0 or not np.isfinite(t) or not np.isfinite(delay) or not np.isfinite(width):
        raise ValueError("t, delay and width must be finite and width >= 0")
    n = time.size
    start = t + delay
    if n == 1:
        return float(signal[0]) if start == time[0] else float("nan")
    i0 = int(np.searchsorted(time, start, side="left"))
    # local sample spacing near the window start
    j = min(max(i0, 1), n - 1)
    dt = time[j] - time[j - 1]
    if not dt > 0:
        finite = np.diff(time)
        finite = finite[finite > 0]
        if finite.size == 0:
            return float("nan")
        dt = float(np.median(finite))
    # accept a start within half a sample before the first sample
    if i0 > 0 and time[i0 - 1] >= start - 0.5 * dt:
        i0 -= 1
    if i0 >= n or time[i0] < start - 0.5 * dt or time[i0] > start + 0.5 * dt + width:
        return float("nan")
    count = max(1, int(round(width / dt)) + (1 if width > 0 else 0))
    i1 = i0 + count
    if i1 > n:
        return float("nan")
    values = signal[i0:i1]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def interevent_metrics(time, event_times, signals: Mapping[str, np.ndarray],
                       delay: float = 0.05, width: float = 0.05) -> dict:
    """Per-event recurrence and per-cycle change of each signal.

    Returns a dict with ``event_times`` (sorted as given), ``recurrence``
    (t_i - t_{i-1}, NaN for the first event), and for every signal ``name``:
    ``f"{name}_after"`` (value sampled after each event) and
    ``f"{name}_per_cycle"`` (difference of consecutive after-values, NaN for
    the first event).  Event times must be increasing.
    """
    time = np.asarray(time, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    if event_times.ndim != 1:
        raise ValueError("event_times must be 1-D")
    if event_times.size > 1 and np.any(np.diff(event_times) <= 0):
        raise ValueError("event_times must be strictly increasing")
    n = event_times.size
    result: dict = {
        "version": RESULT_VERSION,
        "delay_s": float(delay),
        "width_s": float(width),
        "event_times": event_times.tolist(),
        "recurrence": [float("nan")] + np.diff(event_times).tolist() if n else [],
    }
    for name, signal in signals.items():
        after = np.array([sample_after(time, signal, t, delay, width) for t in event_times])
        per_cycle = np.full(n, np.nan)
        if n > 1:
            per_cycle[1:] = after[1:] - after[:-1]
        result[f"{name}_after"] = after.tolist()
        result[f"{name}_per_cycle"] = per_cycle.tolist()
    return result


def creep_per_cycle(slip_per_cycle, coseismic_slip) -> np.ndarray:
    """Interseismic creep = fault slip over the cycle - coseismic slip of the event ending it."""
    slip_per_cycle = np.asarray(slip_per_cycle, dtype=float)
    coseismic_slip = np.asarray(coseismic_slip, dtype=float)
    if slip_per_cycle.shape != coseismic_slip.shape:
        raise ValueError("slip_per_cycle and coseismic_slip must have the same length")
    return slip_per_cycle - coseismic_slip


def event_times_from_run(run: Mapping) -> Optional[np.ndarray]:
    """Event times of a run dict: from ``events[*].event_time``, else ``time[event_indices]``."""
    events = run.get("events")
    if events is not None and len(events) > 0:
        times = []
        for event in events:
            try:
                times.append(float(event["event_time"]))
            except (KeyError, TypeError, ValueError):
                return None
        return np.asarray(times)
    indices = run.get("event_indices")
    if indices is not None and len(indices) > 0 and "time" in run:
        return np.asarray(run["time"], dtype=float)[np.asarray(indices, dtype=int)]
    return None
