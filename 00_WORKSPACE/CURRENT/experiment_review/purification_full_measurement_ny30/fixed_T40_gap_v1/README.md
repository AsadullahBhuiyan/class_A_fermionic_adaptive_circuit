# Purification figure with panel C at fixed T=40

Panel C is recomputed from Campaign 13, not Campaign 26. Nx=20,
Ny=20,24,30,36,44,56,60; 100 independent Born trajectories per size.
Hard walls, alpha1=1, alpha2=30, nshell=1, maxmix with Born-conditioned
exterior, perfect correction, raster-y, complex128, meas_slab_only=True.
All 140 result/receipt pairs and all sample IDs are verified.

At exactly the saved cycle T=40, take each trajectory's minimum absolute
modular energy log[(1-nu)/nu], excluding saved pure-mode caps, and divide
by 2T=80. Average these 100 gaps; SEM=sample SD/sqrt(100). The minimum is
not taken after spectral or covariance averaging. Cross-check every gap
against the stored soft-mode flip costs. No bootstrap or new simulation.

| Ny | Mean finite-time half gap ± SEM |
|---:|---:|
| 20 | 0.041357 ± 0.002482 |
| 24 | 0.037433 ± 0.001832 |
| 30 | 0.021958 ± 0.001459 |
| 36 | 0.017405 ± 0.001314 |
| 44 | 0.014194 ± 0.000875 |
| 56 | 0.011773 ± 0.000767 |
| 60 | 0.009286 ± 0.000583 |

The fit preserves the original panel C estimator: weighted least squares
of log(mean gap) against log(Ny), weights (mean/SEM)^2. Propagated
one-standard-error fit uncertainties use the absolute sampling SEMs,
without residual rescaling. z=1.371678 ± 0.058650,
chi-square=13.9004 for 5 degrees
of freedom. This is a descriptive fixed-time fit, not an established
infinite-time Lyapunov exponent. The divisor is constant across all sizes;
fitting raw modular gaps produces the same exponent.

The all-size fit has reduced chi-square 2.78,
so a single power law does not describe all means within their sampling errors
particularly well. Restricting the fit to Ny>=30 changes the estimate to
z=1.135 ± 0.113
(chi-square=3.193 for
3 degrees of freedom). The strong size
decrease is robust, but the full-range exponent should not be treated as a
precise universal exponent; small-size corrections and finite observation
time remain possible. Statistical fit errors do not include this model/window
dependence. The original T=2Ny fit and this T=40 fit use correlated observations
from the same trajectories, not independent experiments.

The previous panel C used T=2Ny. Its images and data are preserved. All
other panels use the same data and styling as before: A/B/D remain the
Ny=30 full-measurement Campaigns 21–22 with endpoint T=60. Their means,
SEMs and spatial heatmap were checked to be exactly unchanged. Panel C
is a different slab-only protocol and must not be presented as a size
sweep of A/B/D. No 30x30 simulation has been performed.

Files: gap_vs_Ny_T40.pdf/png is the standalone plot;
purification_ny30_gap_T40_4x1.pdf/png is the revised four-panel plot;
caption.tex explicitly states both protocol and time differences.
CSV tables preserve all 700 sample gaps, means/SEMs and fit-window
sensitivity. analysis_manifest.json binds inputs and output hashes.
The manuscript and original figure assets have not been overwritten.
