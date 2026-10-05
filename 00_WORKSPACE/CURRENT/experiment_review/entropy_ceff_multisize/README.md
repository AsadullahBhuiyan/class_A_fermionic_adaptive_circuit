# Cycle-resolved effective central charge across circumference

This analysis recreates panel (d) of the legacy evidence atlas with the three
completed hard-wall S100 datasets at `Nx=20`, `Ny=30,40,50`, and `nshell=1`.
All three sources use pure random initialization, raster-y ordering, perfect
correction, complex128, `alpha_1=1`, `alpha_2=30`, and run through `2Ny` cycles.

At each cycle the script first averages the full-x strip entropy over the 100
independent trajectories, then fits the mean curve against
`log[sin(pi Ay/Ny)]` on `Ay=8,...,Ny/2`. The plotted quantity is three times
that slope. Error bars reproduce the atlas convention: they are OLS errors of
the fitted mean-curve shape, not confidence intervals across trajectories.

The figure shows every fifth cycle from cycle 10 onward. Cycle 5 is retained in
the CSV but omitted from the panel because its large initialization transient
(`c_eff` about 3.6--5.7) would compress the converged regime near one.

Run from the repository root with:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/entropy_ceff_multisize/plot_multisize_ceff_vs_cycle.py
```

The script validates the complete shared scientific contract and source-array
shapes before producing PDF, 300-dpi PNG, a complete per-cycle CSV, and a
checksum-bearing figure manifest.

## Every-cycle line version

`plot_ceff_every_cycle.py` reads the same complete CSV and produces
`figures/ceff_every_cycle_Nx20_Ny30_40_50_S100.pdf` and `.png`, with its own
`ceff_every_cycle_manifest.json`. It preserves the earlier marker figure.
Both panels connect every saved physical cycle (1 through 2Ny), without
markers, smoothing, or subsampling. The first panel includes the full initial
transient; the second zooms to cycles 10 onward to make convergence legible.
Shaded bands retain the original ±1 regression-fit SE, not trajectory SEM.
The ensemble, initialization, estimator order, and fit window are unchanged.

```bash
python 00_WORKSPACE/CURRENT/experiment_review/entropy_ceff_multisize/plot_ceff_every_cycle.py
```
