# Preprocessing examples

Notebooks that turn raw acquisition files into the HDF5 (or NPZ) experiment
file Labquake Explorer loads. Each produces one file with

```
name, date, ... (experiment metadata)
runs/[i]/name, time, <channels...>, units, normal_stress_level, strain{...}
```

`strain` records where the high-rate data lives so that **Extract Events** can
read it later for picked event times (see `docs/extending.md`, "Raw-data
references").

| notebook | acquisition | status |
|---|---|---|
| `t0211_tpc5.ipynb` | Elsys TranAX tpc5 only, ECR dual mode (2 kHz continuous block + 2 MHz trigger blocks) | done |
| `t0207_tpc5_ni.ipynb` | two recorders: mechanical channels on a National Instruments logger (npz, 500 kHz continuous), PZT on the Elsys (tpc5, dual mode); runs without an NI file use the tpc5-only path | done |

The notebooks expect the repository root on `sys.path` (they add it
themselves) or `pip install -e .`, plus `pandas`, `ipympl` and `jupyter`.
Raw data paths are set in the "Paths" cell; outputs go next to the raw data
because tpc5 paths are stored relative to the output file.
