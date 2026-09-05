<div align="center">
  <img src="assets/icons/labquake_explorer.svg" width="150" height="150">
</div>

# Labquake Explorer

A GUI application for analyzing and visualizing labquake event data.

## Installation

```bash
pip install -e .
```

## Usage

Run using the installed entry point:
```bash
labquake-explorer
```

Or run directly:
```bash
python -m labquake_explorer.main
```

Right-click a node in the data tree to see the analyses that apply to it
(runs, events, arrays). Results are saved into the loaded experiment under
the node they belong to and written to disk with "Save As".

## Features

- Load and analyze labquake data stored in NPZ and HDF5 formats
- Pick events on a run and extract event windows
- Event Analyzer: loading/unloading stiffness, stress drop and slip from
  draggable ranges, plus trend-extrapolated stress drop and slip; apply one
  set of windows to all events of a run
- Inter-event Metrics: recurrence, load-point advance, fault slip and creep
  per stick-slip cycle
- Pick dynamic strain arrivals and estimate rupture speed
- Fit a Cohesive Zone Model to dynamic strain records
- PZT Spectrum: calibrated amplitude spectrum, omega^-n source fit and
  source parameters (moment, magnitude, radius, stress drop) per channel
- Source Scaling: power-law fits of source parameters across events
- Plot Run Signals: overlay run-level signals with event markers

## Development

Tests use synthetic stick-slip runs with known answers and run headlessly:

```bash
python -m pytest tests -q
```

See `docs/extending.md` for how to add a view or an analysis and for the
saved result schemas.
