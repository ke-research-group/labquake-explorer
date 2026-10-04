"""Raw-data sources referenced from a run.

A run dict can point at the acquisition files its time history was built from,
so that event extraction can read the full-rate records around picked event
times without copying them into the experiment file.  Each pointer is a small
JSON-like dict with a ``format`` key::

    run['strain'] = {'format': 'tpc5',   'filename': 'run1.tpc5', 'time_offset': 2.39, 'fields': [...], ...}
    run['ni']     = {'format': 'ni_npz', 'filename': 'run1.npz',  'time_offset': 0.0,  'fields': [...], ...}

``time_offset`` puts the file's clock on the run's clock
(``t_run = t_file + time_offset``); ``fields`` names the run field each file
channel feeds.  :func:`open_source` turns a pointer back into the matching
:class:`Source`, which reads windows by run time.  A new recorder is added by
subclassing :class:`Source` and decorating it with :func:`register_source`.

The PSU-era layout (``strain`` with ``filename``, ``time_offset``, ``time``,
``raw`` and no ``format``) is handled by :class:`LegacyTpc5Source`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import h5py
import numpy as np

from labquake_explorer.utils import tpc5
from labquake_explorer.utils.ni_npz import NIRecord, open_ni_npz

SOURCE_FORMATS: dict = {}


def register_source(cls):
    """Class decorator: make ``cls`` the reader for references with ``format == cls.format``."""
    if not getattr(cls, "format", ""):
        raise ValueError(f"{cls.__name__} needs a non-empty 'format'")
    SOURCE_FORMATS[cls.format] = cls
    return cls


def block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive blocks of ``factor`` samples along the last axis."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(data)
    n = (data.shape[-1] // factor) * factor
    return np.asarray(data[..., :n], dtype=np.float64).reshape(*data.shape[:-1], n // factor, factor).mean(axis=-1)


@dataclass
class Window:
    """Samples of one source between two run times."""
    time: np.ndarray            # run clock (s)
    data: np.ndarray            # (n_fields, n)
    fields: list
    sample_rate: float
    block: Optional[int] = None  # tpc5 block number, when applicable

    def as_dict(self, dtype=np.float32) -> dict:
        return {"time": np.asarray(self.time, dtype=np.float64), "raw": np.asarray(self.data, dtype=dtype),
                "fields": list(self.fields), "sample_rate": float(self.sample_rate), "block": self.block}


class Source:
    """A recording file plus the mapping of its clock and channels onto a run."""

    format: str = ""

    def __init__(self, path, fields: Sequence[str], time_offset: float = 0.0,
                 event_fields: Optional[Sequence[str]] = None,
                 event_window_s: Optional[Sequence[float]] = None, meta: Optional[Mapping] = None):
        self.path = Path(path)
        self.fields = [str(f) for f in fields]
        self.time_offset = float(time_offset)
        self.event_fields = [str(f) for f in event_fields] if event_fields else None
        self.event_window_s = (float(event_window_s[0]), float(event_window_s[1])) if event_window_s else None
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ clocks
    def to_file_time(self, t_run: float) -> float:
        return float(t_run) - self.time_offset

    def to_run_time(self, t_file) -> np.ndarray:
        return np.asarray(t_file, dtype=np.float64) + self.time_offset

    # ------------------------------------------------------------ channels
    def field_indices(self, fields: Optional[Sequence[str]]) -> list:
        if fields is None:
            return list(range(len(self.fields)))
        out = []
        for f in fields:
            if f not in self.fields:
                raise KeyError(f"{self.path.name}: unknown field {f!r}; known: {self.fields}")
            out.append(self.fields.index(f))
        return out

    # ------------------------------------------------------- abstract reads
    def time_history(self, decimation: int = 1):
        """``(t_run, data)`` of the whole-run record, block-mean decimated."""
        raise NotImplementedError

    def trigger_times(self) -> list:
        """Trigger times on the run clock (empty when the recorder has none)."""
        return []

    def read_window(self, t_from: float, t_to: float, fields: Optional[Sequence[str]] = None,
                    dtype=np.float32) -> Optional[Window]:
        """Samples between two run times, or None when no record covers them."""
        raise NotImplementedError

    def event_window(self, event_time: float, pre: float, post: float,
                     fields: Optional[Sequence[str]] = None, dtype=np.float32) -> Optional[Window]:
        """The window around an event, honouring the source's own ``event_window_s``
        and ``event_fields`` when they are set."""
        if self.event_window_s is not None:
            pre, post = self.event_window_s
        if fields is None:
            fields = self.event_fields
        return self.read_window(float(event_time) - float(pre), float(event_time) + float(post), fields, dtype)

    # ----------------------------------------------------------- references
    def to_reference(self, base_dir) -> dict:
        """The JSON-like pointer stored in the run dict."""
        ref = {
            "format": self.format,
            "filename": os.path.relpath(self.path, Path(base_dir)),
            "time_offset": self.time_offset,
            "fields": list(self.fields),
        }
        if self.event_fields is not None:
            ref["event_fields"] = list(self.event_fields)
        if self.event_window_s is not None:
            ref["event_window_s"] = list(self.event_window_s)
        for key, value in self.meta.items():
            ref.setdefault(key, value)
        return ref

    @classmethod
    def from_reference(cls, ref: Mapping, base_dir, run: Optional[Mapping] = None) -> "Source":
        raise NotImplementedError

    @staticmethod
    def _common_kwargs(ref: Mapping) -> dict:
        return {
            "time_offset": float(ref.get("time_offset", 0.0)),
            "event_fields": list(ref["event_fields"]) if ref.get("event_fields") is not None else None,
            "event_window_s": list(ref["event_window_s"]) if ref.get("event_window_s") is not None else None,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path.name}, fields={len(self.fields)}, time_offset={self.time_offset:+.4f})"


# ---------------------------------------------------------------------------
# Elsys TranAX tpc5 (single block or ECR dual mode)
# ---------------------------------------------------------------------------
def short_label(name: str) -> str:
    """'A1 (0.A1)' -> 'A1'."""
    return str(name).split()[0] if str(name).strip() else str(name)


@register_source
class Tpc5Source(Source):
    """A tpc5 file: block 1 is the whole-run (lowest-rate) record, any further
    blocks are high-rate records around triggers, all on one clock."""

    format = "tpc5"

    def __init__(self, path, fields, channel_numbers: Sequence[int], blocks: Sequence[tpc5.BlockInfo],
                 channel_names: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.channel_numbers = [int(c) for c in channel_numbers]
        self.channel_names = [str(n) for n in channel_names] if channel_names else [str(c) for c in self.channel_numbers]
        if len(self.channel_numbers) != len(self.fields):
            raise ValueError("fields and channel_numbers must have the same length")
        self.blocks = list(blocks)
        self.continuous = tpc5.continuous_block(self.blocks)
        self.trigger_blocks = tpc5.trigger_blocks(self.blocks)

    @classmethod
    def open(cls, path, channel_map: Optional[Mapping[str, str] | Callable[[str], str]] = None,
             channels: Optional[Sequence[int]] = None, **kwargs) -> "Tpc5Source":
        """Open a tpc5 file; ``channel_map`` turns a channel label ('A1') into a field name."""
        with h5py.File(path, "r") as f:
            infos = tpc5.list_channels(f)
            if channels is not None:
                wanted = {int(c) for c in channels}
                infos = [c for c in infos if c["number"] in wanted]
            blocks = tpc5.list_blocks(f, infos[0]["number"])
        names = [c["name"] for c in infos]
        fields = [_map_field(short_label(n), channel_map) for n in names]
        return cls(path, fields, [c["number"] for c in infos], blocks, channel_names=names, **kwargs)

    @property
    def start_time(self) -> str:
        return self.continuous.start_time

    def time_history(self, decimation: int = 1, dtype=np.float32):
        with h5py.File(self.path, "r") as f:
            t, data = tpc5.read_block(f, self.continuous.block, self.channel_numbers, dtype=np.float64)
        if decimation > 1:
            data = block_mean(data, decimation)
            t = block_mean(t, decimation)
        return self.to_run_time(t), np.asarray(data, dtype=dtype)

    def trigger_times(self) -> list:
        return [b.trigger_time + self.time_offset for b in self.trigger_blocks]

    def record_for(self, t_run: float) -> Optional[tpc5.BlockInfo]:
        """The high-rate block containing a run time; the whole-run block when the
        file has no trigger blocks; None otherwise."""
        t = self.to_file_time(t_run)
        if self.trigger_blocks:
            return tpc5.find_block(self.trigger_blocks, t)
        return self.continuous if self.continuous.contains(t) else None

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        block = self.record_for(0.5 * (float(t_from) + float(t_to)))
        if block is None:
            return None
        idx = self.field_indices(fields)
        lo = max(self.to_file_time(t_from), block.start)
        hi = min(self.to_file_time(t_to), block.end)
        with h5py.File(self.path, "r") as f:
            t, data = tpc5.read_window(f, block, lo, hi, [self.channel_numbers[i] for i in idx], dtype=np.float64)
        return Window(self.to_run_time(t), np.asarray(data, dtype=dtype), [self.fields[i] for i in idx],
                      block.sample_rate, block.block)

    def to_reference(self, base_dir) -> dict:
        ref = super().to_reference(base_dir)
        ref.update({
            "channel_numbers": list(self.channel_numbers),
            "channel_names": list(self.channel_names),
            "start_time": self.start_time,
            "continuous": self.continuous.as_dict(),
            "blocks": tpc5.block_table(self.trigger_blocks),
        })
        ref["blocks"]["trigger_time_run"] = self.trigger_times()
        return ref

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "Tpc5Source":
        cont = ref["continuous"]
        blocks = [tpc5.BlockInfo(int(cont["block"]), float(cont["sample_rate"]), int(cont["n_samples"]),
                                 int(cont["trigger_sample"]), float(cont["trigger_time"]), str(cont.get("start_time", "")))]
        blocks += tpc5.blocks_from_table(ref["blocks"], str(ref.get("start_time", "")))
        meta = {k: v for k, v in ref.items() if k not in _TPC5_KEYS}
        return cls(Path(base_dir) / str(ref["filename"]), list(ref["fields"]), ref["channel_numbers"], blocks,
                   channel_names=ref.get("channel_names"), meta=meta, **cls._common_kwargs(ref))


_TPC5_KEYS = {"format", "filename", "time_offset", "fields", "event_fields", "event_window_s",
              "channel_numbers", "channel_names", "start_time", "continuous", "blocks"}


def _map_field(label: str, channel_map) -> str:
    if channel_map is None:
        return label
    if callable(channel_map):
        return str(channel_map(label))
    return str(channel_map.get(label, label))


# ---------------------------------------------------------------------------
# National Instruments npz
# ---------------------------------------------------------------------------
@register_source
class NINpzSource(Source):
    """An NI ``.npz`` recording (see :mod:`labquake_explorer.utils.ni_npz`)."""

    format = "ni_npz"

    def __init__(self, path, fields, decimation: int = 1, record: Optional[NIRecord] = None, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.decimation = int(decimation)
        self._record = record

    @classmethod
    def open(cls, path, channel_map: Optional[Mapping[str, str] | Callable[[str], str]] = None, **kwargs) -> "NINpzSource":
        record = open_ni_npz(path)
        fields = [_map_field(name, channel_map) for name in record.channels]
        return cls(path, fields, record=record, **kwargs)

    @property
    def record(self) -> NIRecord:
        if self._record is None:
            self._record = open_ni_npz(self.path)
        return self._record

    def time_history(self, decimation: Optional[int] = None, dtype=np.float32):
        factor = self.decimation if decimation is None else int(decimation)
        t, data = self.record.decimate(factor, dtype=dtype)
        return self.to_run_time(t), data

    def trigger_times(self) -> list:
        t = self.record.trigger_time
        return [] if t is None else [t + self.time_offset]

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        rec = self.record
        centre = self.to_file_time(0.5 * (float(t_from) + float(t_to)))
        if not rec.contains(centre):
            return None
        idx = self.field_indices(fields)
        lo = max(self.to_file_time(t_from), 0.0)
        hi = min(self.to_file_time(t_to), (rec.n_samples - 1) / rec.sample_rate)
        t, data = rec.read_window(lo, hi, idx, dtype=np.float64)
        return Window(self.to_run_time(t), np.asarray(data, dtype=dtype), [self.fields[i] for i in idx], rec.sample_rate)

    def to_reference(self, base_dir) -> dict:
        ref = super().to_reference(base_dir)
        info = self.record.as_dict()
        info.pop("format", None)
        for key, value in info.items():
            ref.setdefault(key, value)
        ref["decimation"] = self.decimation
        return ref

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "NINpzSource":
        meta = {k: v for k, v in ref.items() if k not in _NI_KEYS}
        return cls(Path(base_dir) / str(ref["filename"]), list(ref["fields"]),
                   decimation=int(ref.get("decimation", 1)), meta=meta, **cls._common_kwargs(ref))


_NI_KEYS = {"format", "filename", "time_offset", "fields", "event_fields", "event_window_s", "decimation",
            "sample_rate", "n_samples", "duration_s", "dtype", "channels", "labels", "trigger_sample_index",
            "trigger_time", "member_offsets"}


# ---------------------------------------------------------------------------
# PSU-era single-block tpc5 referenced from the legacy ``strain`` layout
# ---------------------------------------------------------------------------
@register_source
class LegacyTpc5Source(Source):
    """A single-block tpc5 whose first sample sits at ``run_t0 + time_offset``
    on the run clock (the convention of the PSU-era ``strain`` dict)."""

    format = "tpc5_legacy"

    def __init__(self, path, fields, run_t0: float, channel_numbers: Sequence[int], block: tpc5.BlockInfo, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.run_t0 = float(run_t0)
        self.channel_numbers = [int(c) for c in channel_numbers]
        self.block = block

    @classmethod
    def open(cls, path, run_t0: float, **kwargs) -> "LegacyTpc5Source":
        with h5py.File(path, "r") as f:
            numbers = tpc5.channel_numbers(f)
            block = tpc5.list_blocks(f, numbers[0])[0]
        return cls(path, [f"ch{c}" for c in numbers], run_t0, numbers, block, **kwargs)

    @property
    def sample_rate(self) -> float:
        return self.block.sample_rate

    def sample_at(self, t_run: float) -> int:
        i = int(round((float(t_run) - self.run_t0 - self.time_offset) * self.block.sample_rate))
        return min(max(i, 0), self.block.n_samples - 1)

    def time_history(self, decimation: int = 1, dtype=np.float32):
        with h5py.File(self.path, "r") as f:
            _, data = tpc5.read_block(f, self.block.block, self.channel_numbers, dtype=np.float64)
        t = self.run_t0 + self.time_offset + np.arange(self.block.n_samples) / self.block.sample_rate
        if decimation > 1:
            data, t = block_mean(data, decimation), block_mean(t, decimation)
        return t, np.asarray(data, dtype=dtype)

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        i0, i1 = self.sample_at(t_from), self.sample_at(t_to) + 1
        if i1 - i0 < 2:
            return None
        idx = self.field_indices(fields)
        with h5py.File(self.path, "r") as f:
            _, data = tpc5.read_block(f, self.block.block, [self.channel_numbers[i] for i in idx], i0, i1, dtype=np.float64)
        t = self.run_t0 + self.time_offset + np.arange(i0, i1) / self.block.sample_rate
        return Window(t, np.asarray(data, dtype=dtype), [self.fields[i] for i in idx], self.block.sample_rate, self.block.block)

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "LegacyTpc5Source":
        if run is None or "time" not in run:
            raise ValueError("a legacy strain reference needs the run (its time axis sets the clock)")
        return cls.open(Path(base_dir) / str(ref["filename"]), float(np.asarray(run["time"])[0]),
                        time_offset=float(ref.get("time_offset", 0.0)))


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def is_legacy_strain(value) -> bool:
    return (isinstance(value, Mapping) and "format" not in value and "filename" in value
            and "time_offset" in value and "raw" in value)


def reference_format(value) -> Optional[str]:
    """The source format a run entry points to, or None if it is not a reference."""
    if not isinstance(value, Mapping):
        return None
    fmt = value.get("format")
    if fmt in SOURCE_FORMATS:
        return str(fmt)
    if is_legacy_strain(value):
        return LegacyTpc5Source.format
    return None


def run_sources(run: Mapping) -> dict:
    """``{key: reference}`` for every top-level entry of a run that points at a raw file."""
    return {key: value for key, value in run.items() if reference_format(value) is not None}


def open_source(ref: Mapping, base_dir, run: Optional[Mapping] = None) -> Source:
    fmt = reference_format(ref)
    if fmt is None:
        raise ValueError(f"not a raw-data reference: {list(ref)[:8]}")
    return SOURCE_FORMATS[fmt].from_reference(ref, base_dir, run)


# ---------------------------------------------------------------------------
# waveform blocks inside an extracted event
# ---------------------------------------------------------------------------
def is_waveform_block(value) -> bool:
    """True for an event entry holding ``original: {time, raw}`` (a full-rate record)."""
    if not isinstance(value, Mapping):
        return False
    original = value.get("original")
    return isinstance(original, Mapping) and "time" in original and "raw" in original


def waveform_blocks(event: Mapping) -> dict:
    """``{key: block}`` for every full-rate record of an event, in event order
    (``'strain'`` for the PSU layout, the reference keys such as ``'elsys'`` or
    ``'ni'`` for files extracted through the sources)."""
    return {key: value for key, value in event.items() if is_waveform_block(value)}


def pick_waveform_block(event: Mapping, prefer_fields: Sequence[str] = (),
                        prefer_keys: Sequence[str] = ("strain",)) -> Optional[str]:
    """The key of the record a view should open first: the first block that has
    a field starting with one of ``prefer_fields``, else the first key in
    ``prefer_keys`` that exists, else the first block; None without any."""
    blocks = waveform_blocks(event)
    if not blocks:
        return None
    for key, block in blocks.items():
        fields = [str(f) for f in (block.get("fields") or [])]
        if any(f.startswith(p) for f in fields for p in prefer_fields):
            return key
    for key in prefer_keys:
        if key in blocks:
            return key
    return next(iter(blocks))


def block_channel_labels(block: Mapping) -> list:
    """Channel labels of a waveform block: its ``fields`` when stored, else indices."""
    raw = np.asarray(block["original"]["raw"])
    n = int(raw.shape[0]) if raw.ndim == 2 else 0
    fields = block.get("fields")
    if fields is not None and len(fields) == n:
        return [str(f) for f in fields]
    return [str(i) for i in range(n)]


__all__ = ["SOURCE_FORMATS", "register_source", "Source", "Window", "Tpc5Source", "NINpzSource",
           "LegacyTpc5Source", "short_label", "block_mean", "is_legacy_strain", "reference_format",
           "run_sources", "open_source", "is_waveform_block", "waveform_blocks", "pick_waveform_block",
           "block_channel_labels"]
