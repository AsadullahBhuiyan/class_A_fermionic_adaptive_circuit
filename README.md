# Class A Fermionic Adaptive Circuit

## Project summary

This repository contains simulation and analysis code for 2D free-fermion dynamics in a Class A model, with a focus on:

- domain-wall (DW) physics,
- adaptive/Markov measurement-feedback circuits,
- entanglement contour diagnostics,
- local Chern marker and current-based characterization,
- system-size and parameter sweeps.

The core workflow is:

1. generate covariance trajectories (`G_hist`) or final states (`G_final`),
2. cache results under `cache/G_history_samples/`,
3. run analysis scripts to produce publication-style figures in `figs/`.

---

## Repository layout

```text
.
├── src/
│   └── fgtn/
│       ├── __init__.py
│       └── classA_U1FGTN.py         # main simulation/analysis class
├── scripts/                         # runnable Python scripts
├── notebooks/                       # exploratory and analysis notebooks
├── cache/
│   └── G_history_samples/
│       ├── N12x12/
│       ├── N12x31/
│       ├── N16x16/
│       └── ...                      # grouped by system size N{Nx}x{Ny}
├── figs/                            # generated plots/artifacts (kept as-is)
└── README.md
```

---

## Notes on organization

- `src/` follows a standard source layout so the core model code is separated from scripts and notebooks.
- `scripts/` and `notebooks/` include bootstrap path setup so they can import `fgtn.classA_U1FGTN`.
- Cache files are organized by system size key (`N{Nx}x{Ny}`) for faster navigation and cleaner long runs.
- Figure outputs are intentionally centralized in `figs/`.

---

## Typical usage

Run scripts from repository root:

```bash
python scripts/run_markov_p_sweep.py
python scripts/run_markov_size_sweep_integrated_contour_fit.py
```

For interactive work, open notebooks from `notebooks/`.
