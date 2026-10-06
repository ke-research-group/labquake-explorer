# Extending Labquake Explorer

## Adding a view

1. Create `labquake_explorer/ui/views/<name>_view.py` with a class deriving from
   `EventView` (one event), `RunView` (one run) or `BaseView` (anything else)
   from `labquake_explorer/ui/views/base.py`.
2. Decorate it with `@register_view(label, kinds=[...])` from
   `labquake_explorer/ui/actions.py`. `kinds` are tree-node kinds from
   `labquake_explorer/ui/context.py` (`EVENT`, `RUN`, `RUN_ARRAY`,
   `EVENT_ARRAY`, `EVENT_EXTRACTION`, `ARRAY`, `STRING`). The main window builds
   the right-click menu for a node from the registry; nothing else to edit.
3. Import the module in `labquake_explorer/ui/views/__init__.py` so it is
   registered at start-up.
4. Put numerics in `labquake_explorer/analysis/` as pure functions (arrays in,
   dataclass or JSON-like dict out; never raise on degenerate data, return a
   `valid` flag and a reason) and test them with synthetic signals in `tests/`
   (`tests/synthetic.py` builds stick-slip runs with known answers).

```python
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

@register_view("My Analysis", kinds=[EVENT], order=50)
class MyView(EventView):
    window_title = "My Analysis"
    result_key = "my_analysis"          # saved under runs/[r]/events/[e]/my_analysis

    def build_ui(self):                 # widgets; call self.build_event_selector(frame, row=0, column=0)
        ...
        self.make_figure(figsize=(8, 5), row=2, column=0, sticky="nsew")

    def on_event_loaded(self):          # refresh from self.event; self.load_results() gives the saved dict
        ...

    def save(self):
        self.save_results({"version": 1, ...})   # writes the dict and refreshes the tree
```

`EventView` provides `run_idx`, `event_idx`, `event`, `event_path`, `set_event()`,
the event combobox, `load_results()` / `save_results()`, `make_figure()`,
`toolbar_active()` (ignore drags while pan/zoom is on) and `on_close()`.
`RunView` provides `run_idx`, `run`, `run_path`, `load_results()` /
`save_results()` under `runs/[r]/<result_key>`, `make_figure()`,
`toolbar_active()`, `on_close()` and the hook `on_run_loaded()` (called once
the run is loaded; there is no event selector and no `on_event_loaded()`).

A menu command that is not a window registers a function instead:

```python
@register_action("Do Something", kinds=[RUN_ARRAY])
def do_something(app, ctx): ...       # ctx.path, ctx.run_idx, ctx.event_idx, ctx.key
```

## Data rule

Raw extracted data lives at the top level of an event. Each analysis view
(`event_analysis`, `interevent`, `pzt_spectrum`, `source_scaling`) writes only
under its own key, and every saved dict carries a `version` integer, the
inputs needed to reproduce the numbers, and outputs with units in the key
names. Cross-view reads go through the other view's key.

Two older views predate the rule and are kept as they are so existing files
stay readable: Pick Arrivals writes into the event's raw layout (see its
schema below; the PZT view's "picked arrival" trigger reads
`strain/original/rupture_arrival_time` and the CZM fitter reads
`rupture_speed` directly), and `czm_parms` carries no `version`.

## Run layout

A run built by the preprocessing package (`labquake_explorer/preprocessing/`)
looks like this:

| key | content |
|---|---|
| `time` | 1-D, seconds on the run clock |
| `elsys`, `ni`, ... | one block per recorder: its raw-file reference (next section) plus `raw_data`, the channel array of the voltages it recorded: `data` (channels x samples, float32), `channels` (`pzt_1`, `pressure_1`, `eddy_3`, ...), `unit` (`V`), `positions` |
| `normal_stress`, `shear_stress`, `friction`, `LP_displacement` | 1-D physical channels (MPa, um, ...). PSU-era files also have `displacement` (fault slip); without it the views take the first `slip` channel |
| `slip` | channel array of the eddy-current sensors in um: `data`, `channels` (`slip_1`...), `source` (the voltage channel of each row), `slope_mm_per_v`, `positions` |
| `units` | unit of every top-level array and channel array |
| `calibration` | the steps that produced the physical channels (see the preprocessing package) |
| `sources` | `{recorder key: format}` |
| `event_extraction`, `events` | the Pick Events form's record, with the picks (`event_indices`) inside (schema below), and the extracted events |

A **channel array** (`labquake_explorer/data/channels.py`) is a dict with
`data` shaped `(n_channels, n)`, `channels` naming the rows, `unit`, and an
optional `positions` table (`x`, `y`, `z` per channel, NaN when unknown, plus
`unit` and `frame`). Any further key is metadata. Views address a row by its
path (`slip/slip_3`, `elsys/raw_data/pzt_1`); `aligned_fields(container, n)`
lists every 1-D series of length `n` this way and `get_field(container,
path)` resolves one, so a field is the same whether it is a top-level array
or a row. Event extraction slices channel arrays along the sample axis and
keeps their metadata: `slip` is sliced, each recorder's `raw_data` slice stays
under the recorder's key (`elsys/raw_data`, the same path as in the run), and
the recorder's full-rate record becomes the channel array
`waveform/<recorder>` (with `time`, `sample_rate`, `filename` and, for a
recorder that writes separate records, `block`). Older files with
top-level 1-D channels keep working: a channel array is only an additional
place a field can live.

## Saved result schemas

### `event_extraction` (EventPickerView, run level, version 1)

What the Pick Events form last saved, and what it starts from when reopened:
`event_indices` (the picks, sample indices on the run's time axis), `start_s` and
`end_s` (the window around each pick, seconds, start negative), `x_field` and
`y_field` (the series shown), `n_events` (how many events the last Extract
wrote). Save picks updates `event_indices` only; Extract writes the whole
record. Older files that kept `event_indices` at the run's top level (and the
window under `event_window`) are read; saving moves both into the record.
`labquake_explorer/data/picks.py` gives readers `picked_indices(run)` for
either layout.

### `event_analysis` (EventAnalyzerView, version 3)

Six picked samples on the event slice and what they give on the chosen
fields, nothing more. The numbers are slopes and differences of whatever
`x_field` and `y_field` were: with fault slip on X and shear stress on Y the
slopes are stiffnesses, `delta_x` the coseismic slip and `-delta_y` the
stress drop.

| key | meaning |
|---|---|
| `version` | 3 |
| `x_field`, `y_field` | the fields plotted |
| `loading_indices`, `unloading_indices` | inclusive index ranges of the two slopes |
| `delta_indices` | the two samples differenced (start, end) |
| `loading_slope`, `unloading_slope` | dY/dX by least squares over each range (NaN when degenerate) |
| `delta_x`, `delta_y` | X and Y at `delta_indices[1]` minus at `delta_indices[0]` |

Version 1 and 2 records (`rupture_start_index`/`rupture_end_index`,
`stress_drop`, `displacement`, trend fits) are still read for their picks;
the inter-event and scaling views take `displacement`/`stress_drop` from
them and `delta_x`/`-delta_y` from version 3. "Apply to All Events" turns
the current picks into times relative to the event, places them on every
event of the run and saves each result.

### Pick Arrivals (DynamicStrainArrivalPickerView, no result key)

Written into the event's existing layout, without a `version`:

| key | meaning |
|---|---|
| `rupture_speed` | fitted rupture speed `Cf` in m/s; present only when the picks define a line (a degenerate fit removes the key) |
| `strain/enabled_channels`, `strain/fitting_channels` | one bool per strain channel: shown, and used in the `Cf` fit |
| `strain/original/picked_idx` | picked arrival sample index per channel on `strain/original/time` |
| `strain/original/rupture_arrival_time` | the picked arrival times (s), read by the PZT view's "picked arrival" trigger |

### `czm_parms` (CZMFitterView, no version)

Dict with `Cf`, `y`, `Xc`, `Gc`, `x_min`, `x_tip`, `x_max`, `x_lim_min`,
`x_lim_max`, `strain_gauge`; `Cf` defaults to `|rupture_speed|` when Pick
Arrivals has saved one. A legacy list of eight values is still read.

### `interevent` (InterEventView, run level, version 1)

`delay_s`, `width_s`, `lp_field`, `slip_field`, `event_times`, `recurrence`,
`lp_after`, `slip_after`, `lp_per_cycle`, `slip_per_cycle`, `coseismic_slip`
(from each event's `event_analysis.delta_x`, `displacement` in older records, used only when that
analysis was computed on the same X field, recorded as `coseismic_field`;
events skipped for that reason are listed in `coseismic_skipped`), `creep`. Each event is
sampled by a mean over `width_s` starting `delay_s` after its `event_time`;
per-cycle values are differences between consecutive events; the first event
and any window outside the run are NaN.

### `pzt_spectrum` (PZTSpectrumView, version 1)

`{"version": 1, "channels": {"ch<k>": record}}`, one record per analysed
channel. A record stores every control (trigger source and time, pre/post
window in ms, taper, baseline mode, step removal, calibration path/unit/dB,
t*, fit band, bins per decade, SNR threshold, n fixed, loss), the binned
spectrum (`spectrum`: binned frequencies, amplitudes, noise, SNR, counts and
the window/taper/calibration metadata), the fit (`fit`: `omega0` in the
spectrum's units, `fc`, `n`, weighted rms of ln residuals, bins used, band
used, `converged`, `at_bounds`, `band_limited`, `valid`, `reason`) and, once
"Compute source" has run, `source` (phase, medium constants, distance,
radiation coefficient, free-surface factor, `seismic_moment_nm`, `mw`,
`source_radius_m`, `rupture_area_m2`, `stress_drop_pa`, `kR`, warnings).
Calibration CSVs are recognised by header: the frequency column is any
`frequency`/`freq`/`f` column, with or without a unit (`Frequency (Hz)`,
`freq_kHz`; kHz is converted), the gain column is the one mentioning gain,
amplitude, response, ratio, dB, psi or calib (or the only other column), and a
`(dB)` header is honoured when the dB flag is left to auto. A `Q^-1(f)` table
(`freq_kHz,Qp_inv_median`) with a travel time gives the frequency-dependent
attenuation correction `exp(pi f T Q^-1(f))` inside the table's band.
Conventions: amplitude = |FFT(taper * x)| dt (one-sided, no coherent-gain
division); noise divided by sqrt(mean(taper^2)); calibration converts to a
displacement spectral density in m s and is masked outside the table; fit
`ln A = ln Omega0 - ln(1 + (f/fc)^n)` in `(ln Omega0, ln fc, n)` with n fixed
at 2 by default over log-binned, SNR-gated bins.

### `source_scaling` (SourceScalingView, run level, version 2)

Also `exclude_near_field` (whether records flagged near-field, kR < 3 at the
plateau frequency, were excluded from the fit; by default they are only
warned about).

The channel and Y variable used, quality-gating choice, OLS exponent with
standard error and bootstrap 16-84 range, reduced-major-axis exponent,
coefficient, r2, the fitted `(x, y)` pairs and their event indices, excluded
events with their flags, and the distinct medium/model constant sets of the
records used.

## File formats

"Save As" writes `.npz` (a pickled experiment dict) or `.h5`/`.hdf5`; any
other suffix is refused. Files are written to a temporary sibling and moved
into place once complete, so a failed save leaves the previous file intact.
In HDF5, dicts and lists become groups tagged with a `container` attribute
so digit-keyed dicts and lists round-trip as what they were (older files
without the tag treat a group keyed `0..n-1` as a list); `None` values are
dropped; a scalar string comes back as `str` and any string list as a
`list` (a one-element list stays a list); lists numpy cannot stack (ragged
arrays, mixed content) become a group with one entry per index.

## Raw-data references and event extraction

A run may point at the acquisition files it was built from so that **Extract
Events** reads the full-rate records around picked event times instead of
copying them into the experiment file. A reference is a top-level run entry, a
dict with a `format` key, handled by `labquake_explorer/data/sources.py`:

| format | class | file |
|---|---|---|
| `tpc5` | `Tpc5Source` | Elsys TranAX tpc5, single block or ECR dual mode (block 1 continuous, blocks 2.. around triggers) |
| `ni_npz` | `NINpzSource` | National Instruments recording saved as npz (`aiN.npy` members, `sample_rate`, `channels`, `trigger_sample_index`) |
| `tpc5_legacy` | `LegacyTpc5Source` | the PSU-era `strain` dict (`filename`, `time_offset`, `time`, `raw`; no `format` key) |

Reference keys name the recorder (`elsys`, `ni`); `strain` is the PSU-era
key. Every reference carries `filename` (relative to the experiment file),
`time_offset` (`t_run = t_file + time_offset`), `fields` (the run field each
file channel feeds) and optionally `event_fields` (subset copied into events)
and `event_window_s` (`[pre, post]` seconds, replacing the start/end typed in
the Pick Events form for that source). A `Tpc5Source` also stores the block table so the
explorer knows the trigger times without opening the file; an `NINpzSource`
stores the record's metadata. `open_source(ref, base_dir, run)` rebuilds the
reader; `source.read_window(t_from, t_to, fields)` returns samples on the run
clock, or None when no record covers the time (a tpc5 with trigger blocks only
answers inside a trigger block).

**Adding a recorder**: subclass `Source`, implement `time_history`,
`read_window`, `to_reference`/`from_reference` (and `trigger_times` if it has
any), decorate the class with `@register_source`, and write a synthetic-file
test next to `tests/test_sources.py`.

`EventProcessor.extract_events(run, indices, window, pre, post)` slices every
1-D run field aligned with `time` around each pick, keeps each recorder's
`raw_data` slice under the recorder's key (`elsys/raw_data`, as in the run; the
file reference itself is not copied), and adds `waveform/<recorder>`, the
full-rate record as a channel array (`data (n_channels, n)`, `channels`,
`unit`, `positions`, `time` on the run clock, `sample_rate`, `filename`,
`block`). Which samples make the record
is the source's decision, `Source.waveform(event_time, pre, post)`: a
recorder that writes separate records around triggers (tpc5 ECR mode) copies
the whole record that contains the event time and overlaps the most of the
chosen window; a continuous recorder (NI npz, single-block tpc5) copies the
window itself, its own `event_window_s` taking precedence. The references are
not copied into the event. Views find the records with
`sources.waveform_blocks(event)` (keys `waveform/<recorder>`, or `strain`)
and read them through `waveform_time`, `waveform_data`, `waveform_channels`
and `waveform_store` (where per-record picks such as `picked_idx` and
`rupture_arrival_time` are kept), so the PSU layout and the new one look the
same to them; the PZT Spectrum view offers every record and opens the one
with `pzt` channels, the strain-gauge views (arrival picker, CZM) open
`strain` when present, else the first record. Recorders with no record
covering the event are listed in `notes`. The legacy layout keeps its
historical output (`time`/`raw` downsampled copies plus `original` with the
first 1 % removed as baseline).

## Preprocessing package

`labquake_explorer/preprocessing/` turns raw files into the experiment dict:

- `build.run_from_tpc5(path, channel_map, base_dir, calibration, ...)` and
  `build.run_from_tpc5_ni(tpc5_path, ni_path, elsys_map, ni_map, base_dir,
  calibration, ...)` produce run dicts (time history, volt fields, `units`,
  references under `elsys` / `ni`, `calibration` record); `run_from_sources` is the general form
  (any sources, one of them the time base, the others interpolated onto its
  axis for fields it lacks). `experiment(name, runs, **metadata)` wraps them.
- `calibration.Calibration(*steps)` applies ordered steps and records them:
  `Linear` (`field = factor * source + offset`), `EddySlip` (eddy-current volts
  to `slip_k` in um, zeroed at the run start, plus `displacement`), `Friction`.
  `pressure_transducers(section)` gives the lab's stress conversions. A new
  conversion is a class with `apply(run, units)` and `describe()`.
- `alignment.offset_from_trigger(ni, elsys)` puts the Elsys clock on the NI
  clock from the NI trigger sample; `slip_step_table` checks that slip follows
  each PZT trigger.

The notebooks in `examples/preprocessing/` are thin scripts over this package:
paths, channel maps, the calibration, the builders, plots, save.
