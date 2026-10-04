# Extending Labquake Explorer

## Adding a view

1. Create `labquake_explorer/ui/views/<name>_view.py` with a class deriving from
   `EventView` (one event), `RunView` (one run) or `BaseView` (anything else)
   from `labquake_explorer/ui/views/base.py`.
2. Decorate it with `@register_view(label, kinds=[...])` from
   `labquake_explorer/ui/actions.py`. `kinds` are tree-node kinds from
   `labquake_explorer/ui/context.py` (`EVENT`, `RUN`, `RUN_ARRAY`,
   `EVENT_ARRAY`, `EVENT_INDICES`, `ARRAY`, `STRING`). The main window builds
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
| `raw_data` | channel array of every recorded voltage: `data` (channels x samples, float32), `channels` (`pzt_1`, `pressure_1`, `eddy_3`, ...), `unit` (`V`), `recorder` (`elsys` / `ni` per channel), `positions` |
| `normal_stress`, `shear_stress`, `friction`, `LP_displacement`, `displacement` | 1-D physical channels (MPa, um, ...) |
| `slip` | channel array of the eddy-current sensors in um: `data`, `channels` (`slip_1`...), `source` (the voltage channel of each row), `slope_mm_per_v`, `positions` |
| `units` | unit of every top-level array and channel array |
| `calibration` | the steps that produced the physical channels (see the preprocessing package) |
| `sources`, `elsys`, `ni`, ... | raw-file references (next section) |
| `event_indices`, `events` | picks and extracted events |

A **channel array** (`labquake_explorer/data/channels.py`) is a dict with
`data` shaped `(n_channels, n)`, `channels` naming the rows, `unit`, and an
optional `positions` table (`x`, `y`, `z` per channel, NaN when unknown, plus
`unit` and `frame`). Any further key is metadata. Views address a row by the
path `'<array>/<channel>'` (`slip/slip_3`); `aligned_fields(container, n)`
lists every 1-D series of length `n` this way and `get_field(container,
path)` resolves one, so a field is the same whether it is a top-level array
or a row. Event extraction slices channel arrays along the sample axis and
keeps their metadata, so events carry `raw_data` and `slip` too. Older files
with top-level 1-D channels keep working: a channel array is only an
additional place a field can live.

## Saved result schemas

### `event_analysis` (EventAnalyzerView, version 2)

Sign convention: stress drop positive, slip positive. Windows are times in
seconds relative to `event_time`.

| key | meaning |
|---|---|
| `version` | 2 |
| `x_field`, `y_field`, `fit_method` | fields plotted and `ols` or `theilsen` |
| `loading_indices`, `unloading_indices`, `rupture_start_index`, `rupture_end_index`, `post_indices` | picked sample indices on the event slice |
| `loading_window`, `unloading_window`, `rupture_window`, `post_window` | the same as relative times |
| `loading_stiffness`, `loading_fit` | dY/dX over the loading range and the full fit record (slope, intercept, r2, stderr, n, x_range, valid, reason) |
| `unloading_stiffness`, `unloading_fit` | same over the unloading range |
| `stress_drop` | Y(rupture start) - Y(rupture end) |
| `displacement` | X(rupture end) - X(rupture start) |
| `stress_drop_trend` | pre-trend(t_event) - post-trend(t_event); trends are Y(t) lines over the loading and post ranges |
| `displacement_trend` | post-trend_X(t_event) - pre-trend_X(t_event); the same trend lines fitted to X(t), subtracted in the reversed order so slip is positive (creep-corrected coseismic slip) |
| `pre_trend`, `post_trend`, `pre_trend_x`, `post_trend_x` | fit records of the four trend lines |

Version 1 dicts (six indices, absolute `stress_drop`) are still read; the post
range falls back to defaults. "Apply to All Events" converts the current
windows (relative times) to picks on every event of the run and saves each
result, so one carefully placed set of windows can be propagated.

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
(from each event's `event_analysis.displacement`, used only when that
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
and `event_window_s` (`[pre, post]` seconds, replacing the window typed in the
dialog for that source). A `Tpc5Source` also stores the block table so the
explorer knows the trigger times without opening the file; an `NINpzSource`
stores the record's metadata. `open_source(ref, base_dir, run)` rebuilds the
reader; `source.read_window(t_from, t_to, fields)` returns samples on the run
clock, or None when no record covers the time (a tpc5 with trigger blocks only
answers inside a trigger block).

**Adding a recorder**: subclass `Source`, implement `time_history`,
`read_window`, `to_reference`/`from_reference` (and `trigger_times` if it has
any), decorate the class with `@register_source`, and write a synthetic-file
test next to `tests/test_sources.py`.

`EventProcessor.extract_events(run, indices, window)` slices every 1-D run
field aligned with `time` around each pick and adds, per reference key,
`{format, filename, fields, sample_rate, block, original: {time, raw}}` with
`raw` shaped `(n_fields, n)`. Views find these blocks with
`sources.waveform_blocks(event)`; the PZT Spectrum view offers every block
and opens the one with `pzt` fields, the strain-gauge views (arrival picker,
CZM) open `strain` when present, else the first block.
Events no record covers get a `notes` entry instead. The legacy layout keeps
its historical output (`time`/`raw` downsampled copies plus `original` with the
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
