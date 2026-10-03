"""Reader for the lab's National Instruments recordings saved as ``.npz``.

Layout written by the acquisition script: one ``aiN.npy`` member per analog
input (equal length, one dtype), plus ``sample_rate.npy`` (Hz),
``channels.npy`` (labels or indices, one per channel; defaults to the ``aiN``
numbers) and ``trigger_sample_index.npy`` (the sample at which the external
trigger, i.e. the first Elsys trigger, was received; a negative value means no
trigger).  Members are normally stored uncompressed, so each channel is
exposed as a read-only ``numpy.memmap`` located through the zip's local file
header: nothing is loaded until it is sliced.  Compressed members fall back to
loading the whole array once.

Times are seconds on the recorder's own clock, ``t = i / sample_rate``.
"""
from __future__ import annotations

import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIG = 0x04034B50
_CHANNEL_RE = re.compile(r"^(ai\d+)\.npy$")


def _npy_header(fh):
    """Parse an .npy header from the current position; returns (shape, fortran, dtype)."""
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fh)
    return np.lib.format.read_array_header_2_0(fh)


def _member_layout(path: Path, info: zipfile.ZipInfo):
    """(absolute data offset or None if compressed, shape, dtype) of an .npy member."""
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        fields = _LOCAL_HEADER.unpack(fh.read(_LOCAL_HEADER.size))
        if fields[0] != _LOCAL_SIG:
            raise ValueError(f"{path.name}: bad local header for {info.filename}")
        name_len, extra_len = fields[9], fields[10]
        data_start = info.header_offset + _LOCAL_HEADER.size + name_len + extra_len
        if info.compress_type != zipfile.ZIP_STORED:
            return None, None, None
        fh.seek(data_start)
        shape, fortran, dtype = _npy_header(fh)
        if fortran:
            raise ValueError(f"{info.filename}: Fortran-ordered arrays are not supported")
        return fh.tell(), tuple(shape), np.dtype(dtype)


def _load_small(z: zipfile.ZipFile, name: str, default=None):
    if name not in z.namelist():
        return default
    import io
    return np.load(io.BytesIO(z.read(name)), allow_pickle=False)


@dataclass
class NIRecord:
    """An NI ``.npz`` recording opened for slice-wise access."""
    path: Path
    sample_rate: float
    channels: list                      # member names in channel order, e.g. ['ai0', ...]
    labels: list                        # the stored ``channels`` array (labels or indices)
    n_samples: int
    dtype: np.dtype
    trigger_sample_index: Optional[int] = None
    _offsets: dict = field(default_factory=dict, repr=False)
    _shapes: dict = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ basics
    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def trigger_time(self) -> Optional[float]:
        return None if self.trigger_sample_index is None else self.trigger_sample_index / self.sample_rate

    def index(self, channel: Union[int, str]) -> int:
        if isinstance(channel, (int, np.integer)):
            if not 0 <= int(channel) < len(self.channels):
                raise IndexError(f"channel {channel} out of range (0..{len(self.channels) - 1})")
            return int(channel)
        if channel in self.channels:
            return self.channels.index(channel)
        raise KeyError(f"unknown channel {channel!r}; known: {self.channels}")

    def channel(self, channel: Union[int, str]) -> np.ndarray:
        """The whole channel as a read-only memmap (or a loaded array for compressed members).

        Memmaps are created once per channel and reused.  A compressed member
        has to be decompressed whole, so at most one such array is kept: asking
        for another compressed channel releases the previous one.
        """
        name = self.channels[self.index(channel)]
        if name in self._cache:
            return self._cache[name]
        offset = self._offsets[name]
        if offset is None:
            for other in [k for k, v in self._cache.items() if self._offsets[k] is None]:
                del self._cache[other]
            with np.load(self.path, allow_pickle=False) as z:
                self._cache[name] = np.asarray(z[name])
        else:
            self._cache[name] = np.memmap(self.path, dtype=self.dtype, mode="r",
                                          offset=offset, shape=self._shapes[name])
        return self._cache[name]

    def time(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        stop = self.n_samples if stop is None else min(int(stop), self.n_samples)
        return np.arange(int(start), stop) / self.sample_rate

    def sample_at(self, t: float) -> int:
        """Nearest sample index to time ``t`` (clipped to the record)."""
        i = int(round(float(t) * self.sample_rate))
        return min(max(i, 0), self.n_samples - 1)

    def contains(self, t: float) -> bool:
        return 0.0 <= t <= (self.n_samples - 1) / self.sample_rate

    # ------------------------------------------------------------ reading
    def read(self, channels: Optional[Sequence[Union[int, str]]] = None, start: int = 0,
             stop: Optional[int] = None, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
        """Time axis and samples ``start:stop`` as ``(n_channels, n)``."""
        names = self.channels if channels is None else [self.channels[self.index(c)] for c in channels]
        stop = self.n_samples if stop is None else min(int(stop), self.n_samples)
        start = max(int(start), 0)
        t = self.time(start, stop)
        data = np.empty((len(names), t.size), dtype=dtype)
        for i, name in enumerate(names):
            data[i] = self.channel(name)[start:stop]
        return t, data

    def read_window(self, t_from: float, t_to: float,
                    channels: Optional[Sequence[Union[int, str]]] = None,
                    dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
        """Samples between two times on the recorder's clock (inclusive)."""
        return self.read(channels, self.sample_at(t_from), self.sample_at(t_to) + 1, dtype)

    def decimate(self, factor: int, channels: Optional[Sequence[Union[int, str]]] = None,
                 chunk_samples: int = 5_000_000, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
        """Block-mean decimation by an integer ``factor`` (an anti-aliased
        average of each block of ``factor`` samples), reading in chunks so the
        whole channel is never in memory.  Times are the block centres."""
        factor = int(factor)
        if factor < 1:
            raise ValueError("factor must be >= 1")
        if factor > self.n_samples:
            raise ValueError(f"factor {factor} exceeds the record length {self.n_samples}")
        names = self.channels if channels is None else [self.channels[self.index(c)] for c in channels]
        n = (self.n_samples // factor) * factor
        n_out = n // factor
        chunk = max(factor, (int(chunk_samples) // factor) * factor)
        out = np.empty((len(names), n_out), dtype=dtype)
        for i, name in enumerate(names):
            src = self.channel(name)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                block = np.asarray(src[s:e], dtype=np.float64).reshape(-1, factor).mean(axis=1)
                out[i, s // factor:e // factor] = block
        t = (np.arange(n_out) * factor + 0.5 * (factor - 1)) / self.sample_rate
        return t, out

    # ------------------------------------------------------------ record
    def as_dict(self) -> dict:
        """JSON-like description for storing in a run dict (no data)."""
        return {
            "format": "ni_npz",
            "sample_rate": float(self.sample_rate),
            "n_samples": int(self.n_samples),
            "duration_s": float(self.duration),
            "dtype": str(self.dtype),
            "channels": list(self.channels),
            "labels": [str(x) for x in self.labels],
            "trigger_sample_index": self.trigger_sample_index,
            "trigger_time": self.trigger_time,
            "member_offsets": {k: (None if v is None else int(v)) for k, v in self._offsets.items()},
        }


def open_ni_npz(path: Union[str, Path]) -> NIRecord:
    """Open an NI ``.npz`` recording without loading its channels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        infos = {i.filename: i for i in z.infolist()}
        members = sorted((m.group(1) for m in map(_CHANNEL_RE.match, infos) if m),
                         key=lambda s: int(s[2:]))
        if not members:
            raise ValueError(f"{path.name}: no aiN.npy channel members found")
        rate = _load_small(z, "sample_rate.npy")
        if rate is None:
            raise ValueError(f"{path.name}: sample_rate.npy is missing")
        sample_rate = float(np.ravel(rate)[0])
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError(f"{path.name}: sample_rate must be positive, got {sample_rate}")
        labels = _load_small(z, "channels.npy")
        labels = ([int(m[2:]) for m in members] if labels is None else np.ravel(labels).tolist())
        if len(labels) != len(members):
            raise ValueError(f"{path.name}: channels.npy has {len(labels)} labels for {len(members)} channels")
        trig = _load_small(z, "trigger_sample_index.npy")
        trigger = None if trig is None else int(np.ravel(trig)[0])
        if trigger is not None and trigger < 0:      # the acquisition script's "no trigger" sentinel
            trigger = None
    offsets, shapes, dtypes = {}, {}, {}
    with zipfile.ZipFile(path) as z:
        for name in members:
            member = f"{name}.npy"
            offset, shape, dtype = _member_layout(path, infos[member])
            if offset is None:  # compressed: read the header through the zip stream
                with z.open(member) as fh:
                    shape, _, dtype = _npy_header(fh)
                shape, dtype = tuple(shape), np.dtype(dtype)
            offsets[member], shapes[member], dtypes[member] = offset, shape, dtype
    lengths = {s[0] if s else 0 for s in shapes.values()}
    kinds = set(dtypes.values())
    if len(lengths) != 1 or any(len(s) != 1 for s in shapes.values()):
        raise ValueError(f"{path.name}: channels must be 1-D and equal length, got {shapes}")
    if len(kinds) != 1:
        raise ValueError(f"{path.name}: channels have mixed dtypes {kinds}")
    n_samples = int(lengths.pop())
    if trigger is not None and trigger >= n_samples:
        raise ValueError(f"{path.name}: trigger_sample_index {trigger} is beyond the {n_samples} samples")
    rec = NIRecord(path=path, sample_rate=sample_rate, channels=[f"{m}.npy" for m in members],
                   labels=labels, n_samples=n_samples, dtype=kinds.pop(),
                   trigger_sample_index=trigger, _offsets=offsets, _shapes=shapes)
    rec.channels = [c[:-4] for c in rec.channels]
    rec._offsets = {k[:-4]: v for k, v in offsets.items()}
    rec._shapes = {k[:-4]: v for k, v in shapes.items()}
    return rec


__all__ = ["NIRecord", "open_ni_npz"]
