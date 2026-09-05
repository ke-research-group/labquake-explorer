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
`RunView` provides the same for `runs/[r]`.

A menu command that is not a window registers a function instead:

```python
@register_action("Do Something", kinds=[RUN_ARRAY])
def do_something(app, ctx): ...       # ctx.path, ctx.run_idx, ctx.event_idx, ctx.key
```

## Data rule

Raw extracted data lives at the top level of an event. Each view writes only
under its own key, and every saved dict carries a `version` integer, the
inputs needed to reproduce the numbers, and outputs with units in the key
names. Cross-view reads go through the other view's key.

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
| `displacement_trend` | the same construction on X (creep-corrected coseismic slip) |
| `pre_trend`, `post_trend`, `pre_trend_x`, `post_trend_x` | fit records of the four trend lines |

Version 1 dicts (six indices, absolute `stress_drop`) are still read; the post
range falls back to defaults.

### `czm_parms` (CZMFitterView)

Dict with `Cf`, `y`, `Xc`, `Gc`, `x_min`, `x_tip`, `x_max`, `x_lim_min`,
`x_lim_max`, `strain_gauge`. A legacy list of eight values is still read.

### `interevent` (InterEventView, run level, version 1)

`delay_s`, `width_s`, `lp_field`, `slip_field`, `event_times`, `recurrence`,
`lp_after`, `slip_after`, `lp_per_cycle`, `slip_per_cycle`, `coseismic_slip`
(from each event's `event_analysis.displacement`), `creep`. Each event is
sampled by a mean over `width_s` starting `delay_s` after its `event_time`;
per-cycle values are differences between consecutive events; the first event
and any window outside the run are NaN.
