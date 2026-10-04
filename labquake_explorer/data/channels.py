"""Channel arrays: several channels sampled on the run's time axis, kept as one block.

A channel array is a dict::

    {'data': float32 (n_channels, n),       # one row per channel, aligned with 'time'
     'channels': ['pzt_1', ..., 'eddy_8'],  # row names
     'unit': 'V',
     'positions': {'x': [...], 'y': [...], 'z': [...], 'unit': 'mm', 'frame': '...'},  # optional, NaN = unknown
     ...}                                    # any further per-array metadata (recorder, source, ...)

A run stores the voltages of each recorder as ``run[<recorder>]['raw_data']``
(``run['elsys']['raw_data']``, ``run['ni']['raw_data']``, next to that
recorder's file reference) and sensor arrays derived from them (slip along the
fault, strain gauges, ...) as top-level channel arrays such as ``run['slip']``;
scalar physical channels (``normal_stress``, ``displacement``, ...) stay 1-D
arrays at the top level.  Views address a row by path
(``'slip/slip_3'``, ``'elsys/raw_data/pzt_1'``); the helpers here list and
resolve those paths, so a view never cares where a field lives.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

RESERVED = {"data", "channels", "unit", "positions"}


def positions_table(channels: Sequence[str], positions: Optional[Mapping] = None,
                    unit: str = "mm", frame: str = "") -> dict:
    """``{'x', 'y', 'z'}`` per channel (NaN where unknown) plus ``unit``/``frame``.

    ``positions`` maps channel name -> (x, y, z); missing channels get NaN.
    """
    positions = positions or {}
    xyz = np.full((len(channels), 3), np.nan)
    for i, name in enumerate(channels):
        p = positions.get(name)
        if p is not None:
            xyz[i] = np.asarray(p, dtype=float)[:3]
    return {"x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
            "unit": unit, "frame": frame}


def channel_array(data, channels: Sequence[str], unit: str = "", positions: Optional[dict] = None,
                  dtype=np.float32, **meta) -> dict:
    """Build a channel array; ``positions`` is a table from :func:`positions_table`."""
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 1:
        data = data[None, :]
    channels = [str(c) for c in channels]
    if data.ndim != 2 or data.shape[0] != len(channels):
        raise ValueError(f"data must be (n_channels, n) with {len(channels)} channels, got {data.shape}")
    out = {"data": data, "channels": channels, "unit": str(unit)}
    if positions is not None:
        out["positions"] = positions
    for key, value in meta.items():
        if key in RESERVED:
            raise ValueError(f"{key!r} is reserved in a channel array")
        out[key] = value
    return out


def is_channel_array(value) -> bool:
    if not isinstance(value, Mapping) or "data" not in value or "channels" not in value:
        return False
    data = value["data"]
    try:
        channels = list(value["channels"])
    except TypeError:
        return False
    return isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == len(channels)


def channel_names(value) -> list:
    return [str(c) for c in value["channels"]] if is_channel_array(value) else []


def row(value, channel) -> Optional[np.ndarray]:
    """One channel of a channel array by name or index; None if absent."""
    if not is_channel_array(value):
        return None
    names = channel_names(value)
    if isinstance(channel, (int, np.integer)):
        i = int(channel)
    elif str(channel) in names:
        i = names.index(str(channel))
    else:
        return None
    if not 0 <= i < len(names):
        return None
    return value["data"][i]


def channel_arrays(container: Mapping) -> list:
    """``[(path, array)]`` for the channel arrays at the top level and one level
    down inside other dicts (a recorder's ``raw_data``), in container order."""
    out = []
    for key, value in container.items():
        if is_channel_array(value):
            out.append((str(key), value))
        elif isinstance(value, Mapping):
            for sub, item in value.items():
                if is_channel_array(item):
                    out.append((f"{key}/{sub}", item))
    return out


def find_channel(container: Mapping, name: str):
    """``(path, index)`` of the channel array row named ``name``, or None."""
    for path, value in channel_arrays(container):
        names = channel_names(value)
        if name in names:
            return path, names.index(name)
    return None


def get_channel(container: Mapping, name: str) -> Optional[np.ndarray]:
    """A 1-D top-level array called ``name``, else the row ``name`` of any channel array."""
    value = container.get(name)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value
    hit = find_channel(container, name)
    if hit is None:
        return None
    path, i = hit
    return _array_at(container, path)["data"][i]


def _array_at(container: Mapping, path: str) -> Mapping:
    current = container
    for part in path.split("/"):
        current = current[part]
    return current


def aligned_fields(container: Mapping, n: int, prefix: str = "", recurse: bool = True) -> list:
    """Paths of every 1-D series of length ``n``: top-level arrays, rows of
    channel arrays (``'slip/slip_3'``, including those inside a recorder's
    reference, ``'elsys/raw_data/pzt_1'``) and, with ``recurse``, arrays inside
    plain nested dicts (``'strain/time'``)."""
    out = []
    for key, value in container.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if is_channel_array(value):
            if value["data"].shape[1] == n:
                out.extend(f"{path}/{c}" for c in channel_names(value))
        elif isinstance(value, Mapping) and "format" in value:
            for sub, item in value.items():           # a recorder's raw_data
                if is_channel_array(item) and item["data"].shape[1] == n:
                    out.extend(f"{path}/{sub}/{c}" for c in channel_names(item))
        elif isinstance(value, np.ndarray):
            if value.ndim == 1 and value.shape[0] == n and value.dtype.kind in "iufb":
                out.append(path)
        elif isinstance(value, list):
            if len(value) == n and value and not isinstance(value[0], (dict, list, str)):
                out.append(path)
        elif recurse and isinstance(value, Mapping) and "format" not in value:
            out.extend(aligned_fields(value, n, path, recurse))
    return out


def get_field(container: Mapping, path: str) -> Optional[np.ndarray]:
    """Resolve a path from :func:`aligned_fields` (or a plain key) to its array."""
    parts = [p for p in str(path).split("/") if p]
    current = container
    for i, part in enumerate(parts):
        if is_channel_array(current):
            if i != len(parts) - 1:
                return None
            return row(current, part)
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    if isinstance(current, (np.ndarray, list)):
        return np.asarray(current)
    return None


def slice_channel_array(value: Mapping, sl: slice) -> dict:
    """A copy with ``data[:, sl]``; every other entry is kept as is."""
    out = dict(value)
    out["data"] = value["data"][:, sl]
    return out


__all__ = ["RESERVED", "positions_table", "channel_array", "is_channel_array", "channel_names", "row",
           "channel_arrays", "find_channel", "get_channel", "aligned_fields", "get_field", "slice_channel_array"]
