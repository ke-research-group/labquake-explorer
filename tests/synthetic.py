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
    creep_fraction: float     # interseismic fault slip rate / lp_velocity
    extra: dict = field(default_factory=dict)

    @property
    def loading_rate(self) -> float:
        """d(tau)/dt while locked, MPa/s."""
        return self.stiffness * self.lp_velocity * (1.0 - self.creep_fraction)

    @property
    def stiffness_lp(self) -> float:
        """d(tau)/d(LP_displacement) while locked, MPa/um."""
        return self.stiffness * (1.0 - self.creep_fraction)

    @property
    def stiffness_fault(self) -> float:
        """d(tau)/d(displacement) while locked (creep only), MPa/um."""
        if self.creep_fraction == 0:
            return float("inf")
        return self.stiffness * (1.0 - self.creep_fraction) / self.creep_fraction

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
    creep_fraction: float = 0.05,
    tau0: float = 5.0,
    normal_stress: float = 10.0,
    noise: float = 0.0,
    seed: int = 0,
    lead: float = 2.0,
) -> StickSlipRun:
    """Build a sawtooth stick-slip run.

    Between events the fault creeps at ``creep_fraction * lp_velocity`` while
    LP advances at ``lp_velocity``; shear stress follows
    ``tau0 + stiffness * (LP_displacement - displacement)`` so it rises at
    ``stiffness * lp_velocity * (1 - creep_fraction)`` per second.  At each
    event displacement jumps by ``slip`` and stress drops by ``stress_drop``
    in one sample (the drop is imposed, so it need not equal stiffness*slip).  ``post_slope`` adds a linear
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

    disp = creep_fraction * lp_velocity * time
    tau = tau0 + stiffness * (lp - disp)
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
        creep_fraction=creep_fraction,
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


# ----------------------------------------------------------------------------
# strain-gauge array (for CZM fitter / arrival picker tests)
# ----------------------------------------------------------------------------
def add_strain(
    event: dict,
    n_channels: int = 16,
    fs: float = 2.0e5,
    half_window: float = 0.01,
    rupture_speed: float = 1000.0,
    rise_time: float = 2.0e-5,
    amplitude: float = 0.5,
    locations_mm: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """Attach a synthetic ``strain`` block to an event dict and return it.

    Each channel shows a smoothed step (tanh) arriving at
    ``event_time + (location - location[0]) / rupture_speed``; channel 14 has
    the opposite sign so it can stand in for Eyy.  Layout matches what
    EventProcessor produces: ``strain.original.time/raw`` at full rate and a
    100x downsampled ``strain.time/raw``.
    """
    rng = np.random.default_rng(seed)
    if locations_mm is None:
        locations_mm = 10.5 + 12.0 * (np.arange(n_channels) // 2)
    locations_mm = np.asarray(locations_mm, dtype=float)
    event_time = float(event["event_time"])
    n = int(round(2 * half_window * fs)) + 1
    tt = event_time - half_window + np.arange(n) / fs
    arrivals = event_time + (locations_mm - locations_mm[0]) * 1e-3 / rupture_speed
    raw = np.zeros((n_channels, n))
    for i in range(n_channels):
        sign = -1.0 if i == 14 else 1.0
        raw[i] = sign * amplitude * 0.5 * (1.0 + np.tanh((tt - arrivals[i]) / rise_time))
        if noise:
            raw[i] += rng.normal(0.0, noise, n)
    event["strain"] = {
        "filename": "synthetic.tpc5",
        "filename_downsampled": "",
        "time": tt[::100].copy(),
        "raw": raw[:, ::100].copy(),
        "original": {"time": tt, "raw": raw},
    }
    event["strain_truth"] = {
        "arrival_times": arrivals,
        "locations_mm": locations_mm,
        "rupture_speed": rupture_speed,
    }
    return event["strain"]


def make_experiment_with_strain(**kwargs) -> tuple[dict, list[StickSlipRun]]:
    data, truth = make_experiment(**kwargs)
    for run in data["runs"]:
        for event in run["events"]:
            add_strain(event)
    return data, truth
