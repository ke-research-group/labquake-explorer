"""Readers for Elsys TranAX ``.tpc5`` files (HDF5).

Layout: ``/measurements/00000001/channels/<channel>/blocks/<block>/raw`` holds
unsigned 16-bit samples; channel attributes give the bin-to-volt scaling and
the analog/marker bit masks, block attributes give the sample rate, the
trigger sample and the trigger time in seconds on the file's clock.

TranAX numbers channels and blocks from 1.  In ECR dual mode the first block
is the continuous low-rate record of the whole run and the following blocks
are high-rate records around each trigger; every block's time axis is
``trigger_time + (i - trigger_sample) / sample_rate`` on the same clock, so
a time read off the continuous block locates the trigger block that holds the
high-rate data for it (:func:`find_block`).

The lower-case functions are the Elsys reference helpers kept for
compatibility; the ones below them are the API used by the package.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

MEASUREMENT = '/measurements/00000001'


''' Get Raw Dataset Block'''
def getDataSetName(channel, block = 1):
    blockString   = "%08d" % block
    channelString = "%08d" % channel
    name = '/measurements/00000001/channels/' + channelString + '/blocks/' + blockString + '/raw'
    return name

def getChannelGroupName(channel):
    channelString = "%08d" % channel   
    return '/measurements/00000001/channels/' + channelString +'/'

def getBlockName(channel, block):
    blockString   = "%08d" % block
    channelString = "%08d" % channel
    name = '/measurements/00000001/channels/' + channelString + '/blocks/' + blockString + '/'
    return name

def getVoltageData(fileRef, channel, block = 1):
    channel_group           = fileRef[getChannelGroupName(channel)]
    dataset_name            = getDataSetName(channel,block)

    ''' Get Scaling Parameters '''
    binToVoltageFactor      = channel_group.attrs['binToVoltFactor']
    binToVoltageConstant    = channel_group.attrs['binToVoltConstant']

    ''' Get Analog and Digital Mask for Data separation '''
    analogMask              = channel_group.attrs['analogMask']
    markerMask              = channel_group.attrs['markerMask']

    analogData              = fileRef[dataset_name] & analogMask
    
    ''' Scale To voltage '''
    return analogData * binToVoltageFactor + binToVoltageConstant

def getPhysicalData(fileRef, channel, block = 1):
    channel_group           = fileRef[getChannelGroupName(channel)]
    dataset_name            = getDataSetName(channel,block)

    ''' Get Scaling Parameters '''
    binToVoltageFactor      = channel_group.attrs['binToVoltFactor']
    binToVoltageConstant    = channel_group.attrs['binToVoltConstant']
    VoltToPhysicalFactor    = channel_group.attrs['voltToPhysicalFactor']
    VoltToPhysicalConstant  = channel_group.attrs['voltToPhysicalConstant']

    ''' Get Analog and Digital Mask for Data separation '''
    analogMask              = channel_group.attrs['analogMask']
    markerMask              = channel_group.attrs['markerMask']
    
    analogData              = fileRef[dataset_name] & analogMask
    
    ''' Scale To voltage '''
    voltageData = analogData * binToVoltageFactor + binToVoltageConstant
    return voltageData * VoltToPhysicalFactor + VoltToPhysicalConstant

def getChannelName(fileRef, channel):
    channel_group           = fileRef[getChannelGroupName(channel)]
    return channel_group.attrs['name']

def getPhysicalUnit(fileRef, channel):
    channel_group           = fileRef[getChannelGroupName(channel)]
    return  channel_group.attrs['physicalUnit']

def getSampleRate(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['sampleRateHertz']

def getTriggerSample(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['triggerSample']

def getTriggerTime(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['triggerTimeSeconds']

def getStartTime(fileRef, channel, block = 1):
    block_group             = fileRef[getBlockName(channel,block)]
    return block_group.attrs['startTime']

def getNChannels(fileRef, block = 1):
    return len(fileRef['/measurements/00000001/channels'])

def getNSamples(fileRef, channel, block = 1):
    return len(fileRef[getBlockName(channel,block)+'raw'])


# ---------------------------------------------------------------------------
# package API
# ---------------------------------------------------------------------------
def _attr(obj, key, default=None):
    value = obj.attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    return value


def channel_numbers(f: h5py.File) -> list[int]:
    """Channel numbers present in the file, ascending (TranAX counts from 1)."""
    return sorted(int(name) for name in f[f'{MEASUREMENT}/channels'].keys())


def channel_info(f: h5py.File, channel: int) -> dict:
    """Name, unit, range and bin-to-volt scaling of one channel."""
    g = f[getChannelGroupName(channel)]
    return {
        'number': int(channel),
        'name': str(_attr(g, 'name', f'ch{channel}')),
        'unit': str(_attr(g, 'physicalUnit', 'V')),
        'range_min': float(_attr(g, 'rangeMin', np.nan)),
        'range_max': float(_attr(g, 'rangeMax', np.nan)),
        'bin_to_volt_factor': float(_attr(g, 'binToVoltFactor', 1.0)),
        'bin_to_volt_constant': float(_attr(g, 'binToVoltConstant', 0.0)),
        'n_blocks': len(g['blocks']) if 'blocks' in g else 0,
    }


def list_channels(f: h5py.File) -> list[dict]:
    return [channel_info(f, c) for c in channel_numbers(f)]


@dataclass(frozen=True)
class BlockInfo:
    """One recording block; times are seconds on the file's clock."""
    block: int
    sample_rate: float
    n_samples: int
    trigger_sample: int
    trigger_time: float
    start_time: str = ''        # ISO timestamp of the recording start

    @property
    def start(self) -> float:
        return self.trigger_time - self.trigger_sample / self.sample_rate

    @property
    def end(self) -> float:
        return self.start + (self.n_samples - 1) / self.sample_rate

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    def time(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """Time of samples ``start:stop`` of this block."""
        stop = self.n_samples if stop is None else min(int(stop), self.n_samples)
        return self.trigger_time + (np.arange(int(start), stop) - self.trigger_sample) / self.sample_rate

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def sample_at(self, t: float) -> int:
        """Nearest sample index to time ``t`` (clipped to the block)."""
        i = int(round((t - self.trigger_time) * self.sample_rate)) + self.trigger_sample
        return min(max(i, 0), self.n_samples - 1)

    def as_dict(self) -> dict:
        d = asdict(self)
        d['start'] = self.start
        d['end'] = self.end
        return d


def list_blocks(f: h5py.File, channel: Optional[int] = None) -> list[BlockInfo]:
    """All blocks of a channel (default: the first channel), ascending."""
    channel = channel_numbers(f)[0] if channel is None else int(channel)
    group = f[getChannelGroupName(channel)]['blocks']
    blocks = []
    for name in sorted(group.keys(), key=int):
        bg = group[name]
        blocks.append(BlockInfo(
            block=int(name),
            sample_rate=float(_attr(bg, 'sampleRateHertz')),
            n_samples=int(bg['raw'].shape[0]),
            trigger_sample=int(_attr(bg, 'triggerSample', 0)),
            trigger_time=float(_attr(bg, 'triggerTimeSeconds', 0.0)),
            start_time=str(_attr(bg, 'startTime', '')),
        ))
    return blocks


def continuous_block(blocks: Sequence[BlockInfo]) -> BlockInfo:
    """The block with the lowest sample rate (the whole-run record in dual mode).

    A single-block file returns that block.
    """
    if not blocks:
        raise ValueError('file has no blocks')
    return min(blocks, key=lambda b: (b.sample_rate, b.block))


def trigger_blocks(blocks: Sequence[BlockInfo]) -> list[BlockInfo]:
    """All blocks except the continuous one, in block order."""
    cont = continuous_block(blocks)
    return [b for b in blocks if b.block != cont.block]


def find_block(blocks: Iterable[BlockInfo], t: float) -> Optional[BlockInfo]:
    """The block whose time span contains ``t``; the nearest trigger if several."""
    hits = [b for b in blocks if b.contains(t)]
    if not hits:
        return None
    return min(hits, key=lambda b: abs(b.trigger_time - t))


def read_channel(f: h5py.File, channel: int, block: int,
                 start: int = 0, stop: Optional[int] = None) -> np.ndarray:
    """Samples ``start:stop`` of one channel/block in volts."""
    g = f[getChannelGroupName(channel)]
    raw = f[getDataSetName(channel, block)][start:stop]
    analog = raw & np.uint16(_attr(g, 'analogMask', 0xFFFF))
    return analog * float(_attr(g, 'binToVoltFactor', 1.0)) + float(_attr(g, 'binToVoltConstant', 0.0))


def read_block(f: h5py.File, block: int, channels: Optional[Sequence[int]] = None,
               start: int = 0, stop: Optional[int] = None,
               dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Time axis and voltages ``(n_channels, n)`` of one block.

    ``channels`` defaults to all channels; ``start``/``stop`` are sample
    indices into the block.
    """
    channels = channel_numbers(f) if channels is None else [int(c) for c in channels]
    info = [b for b in list_blocks(f, channels[0]) if b.block == int(block)]
    if not info:
        raise KeyError(f'block {block} not found')
    info = info[0]
    stop = info.n_samples if stop is None else min(int(stop), info.n_samples)
    t = info.time(start, stop)
    data = np.empty((len(channels), t.size), dtype=dtype)
    for i, c in enumerate(channels):
        data[i] = read_channel(f, c, info.block, start, stop)
    return t, data


def read_window(f: h5py.File, block: BlockInfo, t_from: float, t_to: float,
                channels: Optional[Sequence[int]] = None,
                dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """Time axis and voltages of ``block`` between two times on the file clock."""
    i0 = block.sample_at(t_from)
    i1 = block.sample_at(t_to) + 1
    return read_block(f, block.block, channels, i0, i1, dtype)


def block_table(blocks: Iterable[BlockInfo]) -> dict:
    """Columns of block metadata, convenient for storing in a run dict."""
    blocks = list(blocks)
    return {
        'block': [b.block for b in blocks],
        'sample_rate': [b.sample_rate for b in blocks],
        'n_samples': [b.n_samples for b in blocks],
        'trigger_sample': [b.trigger_sample for b in blocks],
        'trigger_time': [b.trigger_time for b in blocks],
        'start': [b.start for b in blocks],
        'end': [b.end for b in blocks],
    }


def blocks_from_table(table: dict, start_time: str = '') -> list[BlockInfo]:
    """Inverse of :func:`block_table` (accepts lists or arrays)."""
    n = len(table['block'])
    return [BlockInfo(int(table['block'][i]), float(table['sample_rate'][i]), int(table['n_samples'][i]),
                      int(table['trigger_sample'][i]), float(table['trigger_time'][i]), start_time)
            for i in range(n)]


def read_continuous(path, channels: Optional[Sequence[int]] = None, dtype=np.float32) -> dict:
    """Read the continuous (lowest-rate) block of a tpc5 file.

    Returns ``time`` (s, file clock), ``data`` (n_channels, n) in volts,
    ``channels`` (info dicts), ``block`` (BlockInfo of the record read) and
    ``trigger_blocks`` (BlockInfo list of the high-rate records).
    """
    with h5py.File(path, 'r') as f:
        infos = list_channels(f)
        if channels is not None:
            wanted = {int(c) for c in channels}
            infos = [c for c in infos if c['number'] in wanted]
        blocks = list_blocks(f, infos[0]['number'])
        cont = continuous_block(blocks)
        t, data = read_block(f, cont.block, [c['number'] for c in infos], dtype=dtype)
    return {
        'time': t,
        'data': data,
        'channels': infos,
        'block': cont,
        'trigger_blocks': trigger_blocks(blocks),
        'start_time': cont.start_time,
    }


__all__ = [
    'MEASUREMENT', 'BlockInfo', 'channel_numbers', 'channel_info', 'list_channels',
    'list_blocks', 'continuous_block', 'trigger_blocks', 'find_block',
    'read_channel', 'read_block', 'read_window', 'block_table', 'blocks_from_table',
    'read_continuous',
    'getDataSetName', 'getChannelGroupName', 'getBlockName', 'getVoltageData',
    'getPhysicalData', 'getChannelName', 'getPhysicalUnit', 'getSampleRate',
    'getTriggerSample', 'getTriggerTime', 'getStartTime', 'getNChannels', 'getNSamples',
]
