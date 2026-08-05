# Labquake Explorer Potter and Tim Integration Summary

## 1. Integration objective

This integration keeps the Official Labquake Explorer architecture and canonical
run/event schema. Potter and Tim production calculations are the numerical
reference; adaptations are limited to explicit inputs, structured outputs,
canonical context resolution, GUI presentation, and I/O boundaries. No
canonical schema or writer was redesigned. Preview results are not automatically
written back to a run, event, or source file.

## 2. Current branch and baseline

- Branch: `integration/potter`
- HEAD when this document was prepared: `a633eee2bd3ddbe4b52d5bf7dee4cc2eb5333802`
- Automated suite: 419 tests passing
- Working tree before documentation was added: clean

## 3. Potter integration

| Feature | Analysis module | GUI entry | Input binding | Numerical parity | Persistence |
|---|---|---|---|---|---|
| Event signal drop | `analysis/event_drop.py` | **Analyze Event Drop** | One explicitly selected generic event-local signal and one fitting-control set | Potter moving average, inclusive window, baseline subtraction, trend fits, and signed `val_pre_0 - val_post_0`; GUI also displays magnitude | Preview only |
| Inter-event D_Push | `analysis/event_drop.py` | **Analyze Event Drop** | Current/previous finite event times and explicit push speed | Signed `(current - previous) * push_speed` | Preview only |
| D_max | `analysis/event_drop.py` | **Analyze Event Drop** | Independently selected full-run signal, run time, delay, smoothing window | Full-run smoothing followed by delayed nearest-sample absolute difference | Preview only |
| D_reference | `analysis/event_drop.py` | **Analyze Event Drop** | Independently selected full-run reference signal, run time, and delay | Raw delayed nearest-sample absolute difference | Preview only |
| Loading stiffness | `analysis/k_stiffness.py` | **Analyze Loading Stiffness** | Explicit Tau-like response and Slip-like predictor signals | Potter processing order and signed slope; sklearn-first RANSAC | Preview only |
| Colored slip lines | None (raw plotting utility) | **Analyze Colored Slip Lines** | Explicit exact run-level signal keys and per-row display settings | No numerical analysis | No persistence |

Event Drop deliberately has one generic event-local **Signal** selector, not
Tau/Mu/LVDT/Slip roles. D_max and D_reference are separate full-run bindings and
may point to the same signal. Delayed targets outside the run range follow
Potter's `numpy.argmin` nearest-sample behavior: the nearest endpoint is used,
possibly producing zero or a first-to-last difference. D_reference does not
infer E3, and D_max does not infer `LP_displacement`.

Loading stiffness labels are analysis roles: the user decides which exact
signals act as the Tau-like response and Slip-like predictor. Both signals use
Potter's moving-average, optional high-pass, optional low-pass, and baseline
order. The slope `k` remains signed. RANSAC uses the default sklearn constructor,
including its unfixed `random_state`; only `ImportError` selects Potter's
deterministic fallback. A cutoff `<= 0` or `>= Nyquist` means that filter is off.

Colored slip lines plots raw time versus raw signal. It performs no offset,
smoothing, normalization, filtering, analysis, export, or save operation.

## 4. Tim integration

| Feature | Analysis module | GUI entry | Explicit inputs | Numerical parity | Persistence |
|---|---|---|---|---|---|
| PZT time-domain window | `analysis/pzt_analysis_seismology.py` | **Preview PZT Time Domain** | Canonical `strain.original.time/raw`, channel index, canonical `event_time` through `BlockTrace` | Explicit trigger replaces only Tim peak selection; later window processing is retained | Preview only |
| PZT spectrum | `analysis/pzt_analysis_seismology.py` | **Preview PZT Spectrum** | `BlockTrace`, calibration CSV, Q CSV, FFT/window parameters | Tim baseline, filtering, taper, FFT, calibration, Q correction, and resampling | Preview only |
| BAC omega-n fitting | `analysis/pzt_analysis_seismology.py` | **Preview PZT Spectrum** | Existing `SpectrumResult` and explicit fit bounds | Tim omega-n fit, model curve, and log-space R² | Preview only |
| EGF spectrum | `analysis/pzt_analysis_egf.py` | **Preview PZT EGF** | Explicit strain channel, canonical trigger, calibration CSV, window/FFT parameters | Tim trim, baseline, filtering, taper, pre-zero padding, calibration, and log interpolation | Preview only |
| EGF omega-n fitting | `analysis/pzt_analysis_egf.py` | **Preview PZT EGF** | Existing `EGFSpectrumResult` and fit bounds | Tim `EGF_SCALE_FIX`, omega-n fit, M0, and Mw conversion | Preview only |
| BAC source parameters | `analysis/pzt_source_parameters.py` | No standalone GUI entry | Explicit omega0, corner frequency, distance, radiation coefficient, and constants | Tim 1-D/2-D `parameter_row` formulas; signed omega0 | API/test verified only |
| BAC source geometry | `analysis/pzt_source_geometry.py` | No standalone GUI entry | Explicit sensor/source coordinates, fault dimensions, and grid/radiation inputs | Tim rectangular/circular grid and interpolation behavior | API/test verified only |
| Power-law scaling | `analysis/pzt_source_scaling.py` | No standalone GUI entry | Explicit paired numeric arrays | Tim positive-pair guards and unweighted log10 regression | API/test verified only |

The PZT views read canonical `strain.original.time` and `strain.original.raw`,
require an explicit channel index, and pass canonical `event_time` through the
in-memory `BlockTrace` boundary. Legacy Tim auto-peak APIs remain public for
parity, but these GUIs use the explicit-trigger APIs and do not fall back to a
different pulse. Calibration and Q paths are supplied by the caller rather than
discovered from project defaults.

The EGF view does not implement event pairing, spectral ratios, source-time
functions, or deconvolution. Source-parameter results are M0 in N m, source
radius in m, rupture area in m², and stress drop in Pa. Geometry coordinates and
fault dimensions are input in cm; distance is returned in m. The default
rectangular/circular search is 200 by 200. Min/Max use `distance/Sa`; Middle is
the fault-center result, not a median. No case or sensor identity is inferred.
Power-law scaling filters to positive x/y pairs, fits unweighted in log10 space,
returns coefficient/exponent and log-space R², and does not aggregate paper
records.

## 5. Intentional architecture adaptations

- Dataset/file loaders became explicit in-memory arrays or `BlockTrace` inputs.
- Fixed or discovered signal names became explicit user/caller bindings.
- Potter E3 inference became an explicit D_reference signal.
- Potter `LP_displacement` inference became an explicit D_max signal.
- Tim case/sensor discovery became caller-provided geometry and identity.
- Tim project-default calibration/Q paths became explicit `Path` inputs.
- Legacy NaN/nested result shaping became Official structured valid/invalid results where applicable.
- Analysis was separated from persistence; preview views do not save results.
- Canonical `event_time` is the trigger used by Official PZT views.
- Event Drop was intentionally simplified to one generic event-local signal workflow.

These are architecture-boundary changes, not rewrites of the retained numerical
formulas.

## 6. Preserved legacy numerical behavior

- Potter signed event-drop delta and signed loading stiffness `k`.
- Potter delayed nearest-endpoint selection for inter-event sampling.
- Potter sklearn default RANSAC randomness contract and ImportError fallback.
- Potter filter cutoff semantics.
- Tim legacy auto-peak APIs.
- Tim explicit-trigger paths changing only the peak source.
- Tim BAC post-zero padding and EGF pre-zero padding.
- Tim coherent-gain handling, including BAC signal/noise asymmetry.
- Tim calibration and Q-correction flow.
- Tim `EGF_SCALE_FIX` and M0/Mw constants.
- Tim source-parameter formulas, geometry grids/interpolation, and power-law guards.

Preservation establishes implementation parity with the referenced student
production workflows. It does not mean this integration independently validated
their scientific correctness or suitability for every experiment.

## 7. Deliberately excluded functionality

- Canonical schema changes or writer redesign
- Automatic result persistence
- Batch analysis, summary, or export
- NPZ/CSV/JSON result writers
- Tim paper-record aggregation and paper-only plotting workflows
- Case registry, filename inference, or sensor-identity inference
- EGF pairing, spectral ratio, STF, or deconvolution
- Mechanical rupture-comparison helpers
- Automatic D_max/reference selection

## 8. Result ownership and persistence

All newly integrated GUI workflows are preview-only. They do not call
`DataManager.set_data()`, mutate canonical event/run objects, or export
NPZ/CSV/JSON results. The displayed values remain in view memory. A formal
result schema and writer are a separate teacher decision.

## 9. Test and parity evidence

- The full automated suite contains 419 passing tests at this baseline.
- Potter calculations have direct-import/reference parity coverage, including
  signed results, RANSAC behavior, preprocessing order, and strict endpoint parity.
- Tim unchanged numerical bodies have AST parity checks that ignore documentation.
- Tim numerical workflows have end-to-end and randomized parity coverage.
- GUI tests run headlessly and cover explicit bindings, invalidation boundaries,
  event switching, expected/unexpected errors, and absence of persistence.
- Draggable callback lifecycle tests cover repeated motion, release redraw, and cleanup.

## 10. Known limitations and human decisions

- GUI layout and terminology may still be adjusted by the teacher.
- Real experimental calibration/Q files and rendered plots need manual validation.
- Sklearn RANSAC uses its default, unfixed random state.
- Constant or near-constant slip can produce a numerically valid but physically
  unhelpful `k`; this retains Potter behavior.
- Endpoint sampling can produce zero or endpoint-to-endpoint differences.
- Tim source geometry, parameter constants, and other legacy assumptions were
  preserved but not scientifically revalidated here.
- There is no approved persistence design for these preview results.
- The EGF GUI displays Omega0, fc, n, R², M0, and Mw; it does not expose a
  separate `t_star` result.

## 11. Recommended review order

1. Read the analysis module docstrings and public API contracts.
2. Review the GUI views and their explicit context/binding boundaries.
3. Review direct parity, randomized, boundary, and headless GUI tests.
4. Confirm the unchanged canonical DataManager/schema contract.
5. Decide formal persistence and final GUI layout only after manual data review.

## 12. Commit inventory

| Group | Important commits |
|---|---|
| Event Drop core/editor | `07419da` Add schema-neutral event drop calculation core; `41e6d92` Add single-signal event drop analysis helper; `c2debe7` Add single-signal event drop preview; `08b924f` Add draggable event drop fitting windows |
| Event Drop orchestration/simplification | `f1550c5` Add in-memory event drop metric orchestration; `9debb1b` Route event drop preview through metric orchestration; `b1a36d9` Simplify event drop signal workflow |
| D metrics | `7ec66d8` Add inter-event D Push analysis; `5f96f3a` Add inter-event D max analysis; `8211902` Add configurable reference displacement metric; `dabe397` Add inter-event displacement preview |
| D parity corrections | `8c79e08` Characterize Potter inter-event boundary behavior; `7d8b3d5` Align inter-event sampling with Potter endpoints |
| Loading stiffness | `9f100e5` Add schema-neutral event loading stiffness analysis; `c54ddd9` Add explicit event loading stiffness preview; `ff00675` Add Potter parity characterization tests for loading stiffness; `6ff6b4f` Align loading stiffness with Potter RANSAC behavior |
| Colored slip lines | `997ad14` Add explicit colored slip line preview |
| Tim BAC/PZT | `44212ae` Import Tim BAC PZT numerical analysis core; `f924dbf` Use canonical event time for PZT waveform preparation; `55723c2` Add canonical-trigger PZT spectrum analysis; `b07fa46` Add explicit PZT spectrum preview; `808f5aa` Add omega-n fitting to PZT spectrum preview |
| Tim EGF | `c478327` Import Tim EGF numerical analysis core; `daedb1e` Add canonical-trigger EGF spectrum analysis; `787ff16` Add explicit PZT EGF spectrum and fitting preview |
| Source parameters | `9cc3ca6` Import Tim BAC source parameter calculations |
| Geometry | `01d99f9` Import Tim BAC source geometry calculations |
| Scaling | `ff77ddf` Import Tim power-law scaling calculations |
| Documentation | `a633eee` Document Potter and Tim integration boundaries |
