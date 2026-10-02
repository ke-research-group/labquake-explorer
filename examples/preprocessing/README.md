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
| tpc5 + NI npz | mechanical data from a National Instruments logger (npz), dynamic data from tpc5 | to do |

The notebooks expect the repository root on `sys.path` (they add it
themselves) or `pip install -e .`, plus `pandas`, `ipympl` and `jupyter`.
Raw data paths are set in the "Paths" cell; outputs go next to the raw data
because tpc5 paths are stored relative to the output file.
