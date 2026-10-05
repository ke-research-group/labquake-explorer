# Preprocessing examples

Notebooks that turn raw acquisition files into the HDF5 (or NPZ) experiment
file Labquake Explorer loads. They are thin scripts over
`labquake_explorer.preprocessing` (builders, calibration, clock alignment) and
`labquake_explorer.data.sources` (readers for the raw files): each notebook
sets paths, channel maps and the calibration, builds the runs, plots a few
checks and saves. See `docs/extending.md`, "Raw-data references" and
"Preprocessing package".

| notebook | acquisition |
|---|---|
| `t0211_tpc5.ipynb` | Elsys TranAX tpc5 only, ECR dual mode (2 kHz continuous block + 2 MHz trigger blocks) |
| `t0207_tpc5_ni.ipynb` | two recorders: mechanical channels on a National Instruments logger (npz, 500 kHz), PZT on the Elsys; clocks aligned by the trigger pulse |

Each run of the produced file holds `time`, one block per recorder (`elsys`,
`ni`) with its raw-file reference and its voltages as the `raw_data` channel
array (sensor positions per channel), the calibrated scalars `normal_stress`, `shear_stress`, `friction`,
`displacement`, the `slip` channel array (um, one row per eddy-current sensor,
positions attached), `units`, a `calibration` record, and references to the
raw files (`elsys`, `ni`) that **Pick Events → Extract Events** reads for each
event's `waveform` records (`<recorder>/raw_data` holds the 2 kHz voltages of the window, as in the run). See `docs/extending.md`, "Run layout".

The notebooks expect the repository root on `sys.path` (they add it
themselves) or `pip install -e .`, plus `pandas`, `ipympl` and `jupyter`.
Outputs go next to the raw data because raw-file paths are stored relative to
the output file.
