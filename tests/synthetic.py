"""Synthetic experiments with known answers for tests.

The mechanical data mimic a biaxial stick-slip run: shear stress rises
linearly with load-point displacement at a known stiffness, drops by a known
amount at each event, and the on-fault displacement jumps by a known slip.
Every generated quantity is returned so tests can assert against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StickSlipRun:
    time: np.ndarray
    shear_stress: np.ndarray
    displacement: np.ndarray
    LP_displacement: np.ndarray
    LP_velocity: np.ndarray
    normal_stress: np.ndarray
    event_indices: list[int]
    event_times: np.ndarray
    stiffness: float          # d(tau)/d(LP_displacement) while locked, MPa/um
    stress_drop: float        # MPa, same for every event
    slip: float               # um, same for every event
    lp_velocity: float        # um/s
    post_slope: float         # MPa/s, stress rate after each event (healing/relaxation)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "time": self.time,
            "shear_stress": self.shear_stress,
            "displacement": self.displacement,
            "LP_displacement": self.LP_displacement,
            "LP_velocity": self.LP_velocity,
            "normal_stress": self.normal_stress,
            "friction": self.shear_stress / self.normal_stress,
            "event_indices": list(self.event_indices),
        }
        d.update(self.extra)
        return d


def make_stick_slip_run(
    n_events: int = 4,
    recurrence: float = 4.0,
    fs: float = 1000.0,
    stiffness: float = 0.01,
    stress_drop: float = 0.4,
    slip: float = 30.0,
    lp_velocity: float = 10.0,
    post_slope: float = 0.0,
    tau0: float = 5.0,
    normal_stress: float = 10.0,
    noise: float = 0.0,
    seed: int = 0,
    lead: float = 2.0,
) -> StickSlipRun:
    """Build a sawtooth stick-slip run.

    Between events the fault is locked: LP advances at ``lp_velocity``, on-fault
    displacement is constant, shear stress rises at ``stiffness * lp_velocity``
    per second.  At each event the stress drops by ``stress_drop`` in one
    sample and displacement jumps by ``slip``.  ``post_slope`` adds a linear
    stress trend for ``recurrence/4`` seconds after each event so that the
    pre/post trends differ.
    """
    rng = np.random.default_rng(seed)
    duration = lead + n_events * recurrence + lead
    n = int(round(duration * fs)) + 1
    time = np.arange(n) / fs
    lp = lp_velocity * time
    event_times = lead + recurrence * np.arange(1, n_events + 1)
    event_indices = [int(round(t * fs)) for t in event_times]
    event_times = time[event_indices]

    tau = tau0 + stiffness * lp
    disp = np.zeros(n)
    for k, idx in enumerate(event_indices):
        tau[idx:] -= stress_drop
        disp[idx:] += slip
        if post_slope:
            span = int(round(recurrence / 4 * fs))
            seg = slice(idx, min(idx + span, n))
            t_rel = time[seg] - time[idx]
            tau[seg] += post_slope * t_rel
            # keep continuity after the segment ends
            if seg.stop < n:
                tau[seg.stop:] += post_slope * t_rel[-1]
    if noise:
        tau = tau + rng.normal(0.0, noise, n)
    sigma_n = np.full(n, normal_stress)
    return StickSlipRun(
        time=time,
        shear_stress=tau,
        displacement=disp,
        LP_displacement=lp,
        LP_velocity=np.full(n, lp_velocity),
        normal_stress=sigma_n,
        event_indices=event_indices,
        event_times=event_times,
        stiffness=stiffness,
        stress_drop=stress_drop,
        slip=slip,
        lp_velocity=lp_velocity,
        post_slope=post_slope,
    )


def extract_events(run: dict, window: float) -> list[dict]:
    """Slice a run dict into event dicts the way EventProcessor does (no strain)."""
    events = []
    time = run["time"]
    for idx in run["event_indices"]:
        event_time = time[idx]
        beg = int(np.argmin(np.abs(event_time - window - time)))
        end = int(np.argmin(np.abs(event_time + window - time)))
        event = {"event_time": event_time, "time": time[beg:end]}
        for key, value in run.items():
            if key in ("time", "events", "event_indices"):
                continue
            if isinstance(value, np.ndarray) and value.shape == time.shape:
                event[key] = value[beg:end]
        events.append(event)
    return events


def make_experiment(name: str = "p0001", n_runs: int = 1, window: float = 5.0, **kwargs) -> tuple[dict, list[StickSlipRun]]:
    """Return (experiment dict, [StickSlipRun ...]) with events already extracted."""
    runs = []
    truth = []
    for i in range(n_runs):
        ssr = make_stick_slip_run(seed=i, **kwargs)
        run = ssr.as_dict()
        run["name"] = f"run{i:02d}"
        run["events"] = extract_events(run, window)
        runs.append(run)
        truth.append(ssr)
    return {"name": name, "runs": runs}, truth
