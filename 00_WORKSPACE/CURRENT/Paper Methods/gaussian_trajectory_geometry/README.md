# Gaussian trajectory geometry

This directory contains the standalone PRX Quantum-style methods note
`gaussian_trajectory_geometry.tex`.  It consolidates the formalism and numerical
implementation behind native occupied-frame evolution, the normalized covariance
Möbius tangent cocycle, redesigned G4/G5 spectral extraction, and the Gaussian
reference-ancilla spacetime-anisotropy estimator.

The coarse anisotropy table is a reproducible worked example read from the completed
coarse audit files.  It is not a calibrated anisotropy result and intentionally excludes
the refinement points and paired-bootstrap acceptance gates.
Its immutable source root is
00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/outputs/l24_trunc_aniso_v3_20260827_202740/,
stage audit, with 32 samples, 72 follow cycles, and the third 24-cycle window.

## Canonical sources

- `src/fgtn/occupied_frame.py` and `src/fgtn/occupied_frame_gpu.py`: native frame
  gain, loss, Householder deletion, and QR stabilization.
- `src/fgtn/classA_U1FGTN.py` and `src/fgtn/classA_U1FGTN_gpu.py`: canonical
  normalized circuit and tangent updates.
- `00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_operator_cft/`:
  redesigned G4 contract and streaming estimator.
- `00_WORKSPACE/CURRENT/final_production_new_designs/05_pure_tangent_stability/`:
  redesigned G5 contract and occupied--empty core SVD.
- `00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/`: reference insertion,
  matching estimator, and immutable L=24 audit artifacts.
- `00_WORKSPACE/CURRENT/Paper Methods/open_system_monitored_system_response_theory/`:
  normalized covariance tangent derivation and fixed-record conventions.

## Build

Compile from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build gaussian_trajectory_geometry.tex
```

The canonical PDF is `build/gaussian_trajectory_geometry.pdf`.  The main text is limited
to five pages; references begin after an explicit page break and are excluded from that
limit.
