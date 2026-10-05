# Growth with cycle at each fixed system size

No dynamics or manuscript edits. Campaign 13: Nx20, Ny20,24,30,36,44,56,60,
100 independent Born trajectories each, hard walls, alpha1=1, alpha2=30,
nshell1, raster-y, complex128, perfect correction, slab-only measurements
with Born-conditioned exterior. All 140 pairs validated. The separate
full-measurement Ny30 Campaign 21 (20 pairs) is included in CSVs only and
never pooled with slab-only data.

Take minimum absolute modular energy inside each sample at each cycle,
excluding capped occupations, then average the 100 resulting gap histories.
Fits use the ensemble-mean curve, not an averaged occupation spectrum.
All saved times in the chosen interval have equal OLS weight.

Two descriptive models: mean g(t)=A*t**p (OLS on log of the mean), and
mean g(t)=a*t+b (OLS on the raw mean, with free intercept). Ordinary
sampling SEMs propagate the full trajectory covariance across time:
exactly for affine fits, first-order delta method for log-mean fits.
No bootstrap, independent-time assumption, or residual-based parameter error.
R² is descriptive; high R² alone does not establish either asymptotic model.

Primary window: last half of each slab-only history, 2Ny..4Ny. Sensitivity
windows: all positive saved times, common physical cycles20..60, Ny..4Ny,
and 3Ny..4Ny. Separate full-measurement Ny30 primary window is30..60.
Paired changes in fit coefficients retain the same trajectories across windows.

| Ny | Primary cycles | Power p | Affine a | Affine b | Affine raw R² |
|---:|---:|---:|---:|---:|---:|
| 20 | 40–80 | 1.447 ± 0.081 | 0.14306 ± 0.00811 | -2.449 ± 0.439 | 0.99622 |
| 24 | 48–96 | 1.353 ± 0.074 | 0.11878 ± 0.00728 | -2.028 ± 0.434 | 0.99888 |
| 30 | 60–120 | 1.297 ± 0.083 | 0.08452 ± 0.00549 | -1.619 ± 0.450 | 0.99782 |
| 36 | 72–144 | 1.369 ± 0.090 | 0.07103 ± 0.00455 | -1.928 ± 0.448 | 0.99802 |
| 44 | 88–176 | 1.505 ± 0.074 | 0.06206 ± 0.00326 | -2.538 ± 0.370 | 0.99204 |
| 56 | 112–224 | 1.298 ± 0.082 | 0.04265 ± 0.00265 | -1.577 ± 0.424 | 0.99842 |
| 60 | 120–240 | 1.372 ± 0.088 | 0.04200 ± 0.00269 | -1.974 ± 0.451 | 0.99715 |

The corresponding finite-time rate is Delta=g/(2t). A power fit implies
Delta ~ t**(p-1) only within that fitted interval. An affine raw-gap fit
implies Delta=a/2+b/(2t): a negative b allows a rising rate even if the
raw growth slope is constant. a/2 is a LOCAL slope-derived rate, not a
validated infinite-time extrapolation. Fits are displayed only within the
observed fit windows, not extrapolated. Positive fitted p>1 does not establish
permanent superlinear growth: a negative-intercept affine law can mimic it.

raw_gap_growth_by_size.pdf/png: each size separately, mean±SEM, both fits.
rate_with_affine_growth_fits.pdf/png: measured rates with the affine fits
divided by 2t, shown against both physical and scaled cycle. All times and
uncertainties retain the original sampling units. Previous figures unchanged.
Reproduce: python fit_cycle_growth.py.
