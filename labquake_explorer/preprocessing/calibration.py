"""Volt-to-physical conversions applied to a run dict.

A :class:`Calibration` is an ordered list of steps.  Each step reads fields of
the run (volts, or fields produced by earlier steps), writes new fields, and
updates ``run['units']``.  ``Calibration.apply`` also stores
``run['calibration']``, a JSON-like description of every step, so a saved
experiment says how its physical channels were derived.

Steps provided here:

* :class:`Linear` -- ``field = factor * source + offset``.
* :class:`EddySlip` -- eddy-current voltages to slip in micrometres, zeroed at
  the start of the run, plus the ``displacement`` field the explorer uses.
* :class:`Friction` -- ``shear_stress / normal_stress`` once both are in MPa.

A new kind of conversion is a class with ``apply(run, units)`` and
``describe()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np


class Step:
    def apply(self, run: dict, units: dict) -> None:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


@dataclass
class Linear(Step):
    """``field (unit) = factor * run[source] + offset``; skipped when ``source`` is absent."""
    field: str
    source: str
    factor: float
    offset: float = 0.0
    unit: str = ""

    def apply(self, run, units):
        if self.source not in run:
            return
        run[self.field] = (self.factor * np.asarray(run[self.source], dtype=np.float64) + self.offset).astype(np.float32)
        units[self.field] = self.unit

    def describe(self):
        return {"kind": "linear", "field": self.field, "source": self.source, "factor": float(self.factor),
                "offset": float(self.offset), "unit": self.unit,
                "formula": f"{self.field} ({self.unit}) = {self.factor:.6g} * {self.source} + {self.offset:.6g}"}


@dataclass
class EddySlip(Step):
    """Slip in micrometres from eddy-current sensor voltages.

    The lab calibrates each sensor as ``d (mm) = slope * V + intercept``.  The
    intercept belongs to the calibration jig, so instead every slip channel is
    zeroed on the mean of the first ``zero_window_s`` of the run (the sensors are
    zeroed after the run-in in the experiment as well).  Sensors without their
    own slope use ``default_slope_mm_per_v`` (the mean of the known ones when
    None).  ``displacement`` is the mean of all slip channels (``'mean'``), one of
    them (e.g. ``'slip_5'``) or not written (None).
    """
    slopes_mm_per_v: Mapping[str, float]
    default_slope_mm_per_v: Optional[float] = None
    zero_window_s: float = 0.5
    source_prefix: str = "eddy_"
    out_prefix: str = "slip_"
    displacement: Optional[str] = "mean"

    def slope_for(self, source: str) -> float:
        if source in self.slopes_mm_per_v:
            return float(self.slopes_mm_per_v[source])
        if self.default_slope_mm_per_v is not None:
            return float(self.default_slope_mm_per_v)
        if not self.slopes_mm_per_v:
            raise ValueError("EddySlip needs at least one slope or a default")
        return float(np.mean(list(self.slopes_mm_per_v.values())))

    def sources(self, run) -> list:
        return sorted((k for k in run if k.startswith(self.source_prefix) and k[len(self.source_prefix):].isdigit()),
                      key=lambda k: int(k[len(self.source_prefix):]))

    def apply(self, run, units):
        sources = self.sources(run)
        if not sources:
            return
        t = np.asarray(run["time"], dtype=np.float64)
        dt = float(np.median(np.diff(t[: min(t.size, 2000)]))) if t.size > 1 else 1.0
        n0 = max(1, int(round(self.zero_window_s / dt)))
        slips = {}
        for src in sources:
            v = np.asarray(run[src], dtype=np.float64)
            out = self.out_prefix + src[len(self.source_prefix):]
            s = 1000.0 * self.slope_for(src) * (v - np.nanmean(v[:n0]))
            run[out] = s.astype(np.float32)
            units[out] = "um"
            slips[out] = s
        if self.displacement == "mean":
            run["displacement"] = np.mean(list(slips.values()), axis=0).astype(np.float32)
            units["displacement"] = "um"
        elif self.displacement:
            if self.displacement not in slips:
                raise KeyError(f"displacement source {self.displacement!r} is not one of {list(slips)}")
            run["displacement"] = np.asarray(slips[self.displacement], dtype=np.float32)
            units["displacement"] = "um"

    def describe(self):
        return {"kind": "eddy_slip", "slopes_mm_per_v": {k: float(v) for k, v in self.slopes_mm_per_v.items()},
                "default_slope_mm_per_v": self.slope_for("__default__") if (self.slopes_mm_per_v or self.default_slope_mm_per_v is not None) else None,
                "zero_window_s": float(self.zero_window_s), "source_prefix": self.source_prefix,
                "out_prefix": self.out_prefix, "displacement": self.displacement,
                "formula": f"{self.out_prefix}k (um) = 1000 * slope_k * (V - mean(V over the first {self.zero_window_s:g} s))"}


@dataclass
class Friction(Step):
    """``friction = shear / normal`` when both carry the same non-volt unit."""
    shear: str = "shear_stress"
    normal: str = "normal_stress"
    field: str = "friction"

    def apply(self, run, units):
        if self.shear in run and self.normal in run and units.get(self.shear) == units.get(self.normal) not in (None, "", "V"):
            with np.errstate(divide="ignore", invalid="ignore"):
                run[self.field] = (np.asarray(run[self.shear], dtype=np.float64)
                                   / np.asarray(run[self.normal], dtype=np.float64)).astype(np.float32)
            units[self.field] = ""

    def describe(self):
        return {"kind": "friction", "field": self.field, "formula": f"{self.field} = {self.shear} / {self.normal}"}


class Calibration:
    """An ordered list of steps applied to a run dict."""

    def __init__(self, *steps: Step, note: str = ""):
        self.steps = list(steps)
        self.note = note

    def add(self, *steps: Step) -> "Calibration":
        self.steps.extend(steps)
        return self

    def apply(self, run: dict) -> dict:
        units = run.setdefault("units", {})
        for key, value in run.items():
            if isinstance(value, np.ndarray) and key != "time":
                units.setdefault(key, "V")
        for step in self.steps:
            step.apply(run, units)
        run["calibration"] = self.describe()
        return run

    def describe(self) -> dict:
        out = {"steps": [s.describe() for s in self.steps]}
        if self.note:
            out["note"] = self.note
        return out

    def __repr__(self):
        return f"Calibration({', '.join(type(s).__name__ for s in self.steps)})"


# ---------------------------------------------------------------------------
# lab presets
# ---------------------------------------------------------------------------
# Pressure-transducer voltages to stress for the biaxial apparatus
# (Tseng 2026, NTU thesis, Appendix B.3.3 / Table B.1): V = a * sigma + b on the
# normal-stress line and V = c * tau on the shear jack, by fault section.
PRESSURE_PRESETS = {
    "1D-5cm":  {"normal_v_per_mpa": 0.222, "normal_offset_v": 0.091, "shear_v_per_mpa": 0.109},
    "1D-10cm": {"normal_v_per_mpa": 0.444, "normal_offset_v": 0.091, "shear_v_per_mpa": 0.217},
    "2D-50cm": {"normal_v_per_mpa": 0.739, "normal_offset_v": 0.091, "shear_v_per_mpa": 1.087},
}


def pressure_transducers(section: str = "1D-5cm", normal_source: str = "pressure_1",
                         shear_source: str = "pressure_2") -> list:
    """``normal_stress`` and ``shear_stress`` (MPa) from the two pressure voltages."""
    p = PRESSURE_PRESETS[section]
    return [
        Linear("normal_stress", normal_source, 1.0 / p["normal_v_per_mpa"],
               -p["normal_offset_v"] / p["normal_v_per_mpa"], "MPa"),
        Linear("shear_stress", shear_source, 1.0 / p["shear_v_per_mpa"], 0.0, "MPa"),
    ]


__all__ = ["Step", "Linear", "EddySlip", "Friction", "Calibration", "PRESSURE_PRESETS", "pressure_transducers"]
