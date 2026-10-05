# Endpoint Chern-radius stability at alpha_1=1

This directory contains an offline analysis of saved complex128 occupied frames.
It does not rerun the adaptive dynamics. The analysis reconstructs the spectral
projector as `Gamma=(V V^dagger)^T=V* V^T` and evaluates the repository's
periodic three-sector real-space Chern estimator at multiple radii centered in
the topological slab.

The two source cohorts remain separate:

- `new_width_sweep`: the Ny=24 GPU width sweep with hard and soft walls,
  `n_shell=1,infinity`, and its original S=25 bridge or S=100 primary ensemble;
- `old_fixed_nx`: the Nx=20, Ny=24,28,30 CPU endpoint series with hard and soft
  walls, `n_shell=1`, and S=100 for each ensemble.

For every trajectory and radius, the estimator is first averaged over all
transverse origins `y0`. Means and standard errors are then computed across
independent trajectories. Transverse origins are not treated as independent
samples, and no trajectories are pooled across cohorts or campaign revisions.

Reproduce the analysis from the pilot directory with:

```bash
python analyze_endpoint_chern_radius_stability.py --workers 8
```

The `analysis/` directory contains the per-origin NPZ, trajectory and ensemble
CSVs, stability summary, provenance/validation JSON, and PDF/PNG figures.
