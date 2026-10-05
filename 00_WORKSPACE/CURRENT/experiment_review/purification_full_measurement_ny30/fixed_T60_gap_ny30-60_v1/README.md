# Panel C at fixed T=60

Campaign 13, Nx=20, Ny=30,36,44,56,60, 100 trajectories per size.
The smaller Ny=20,24 cases also contain cycle 60 but are omitted by the
user's explicit size selection. This is not a data-availability limitation.
All 140 campaign result/completion pairs were checksum- and identity-verified;
the selected 100 shards supply exactly 500 trajectories, IDs 0–99 per size.

Hard walls, alpha1=1, alpha2=30, nshell=1, maxmix, Born-conditioned exterior,
perfect correction, raster-y, complex128, meas_slab_only=True.
For each sample take min_j abs(log[(1-nu_j)/nu_j]) at saved cycle 60,
exclude the stored pure caps, divide by 2T=120, then average the 100 gaps.
Errors are ordinary sample SD/sqrt(100), not bootstrap or residual errors.
All gaps are cross-checked against saved soft-mode flip costs.

| Ny | Mean half gap ± SEM |
|---:|---:|
| 30 | 0.028034 ± 0.001465 |
| 36 | 0.019143 ± 0.001288 |
| 44 | 0.016207 ± 0.001046 |
| 56 | 0.011828 ± 0.000761 |
| 60 | 0.008625 ± 0.000634 |

Preserve panel C's original weighted log-space power-law estimator:
log(mean gap)=log(A)-z*log(Ny), with weights (mean/SEM)^2. The result is
z=1.503733 ± 0.104996;
chi-square=7.1005 for 3 degrees
of freedom. The fit uncertainty is propagated from sampling SEMs without
residual rescaling. It does not include finite-time or model uncertainties.

| Fit window | z ± propagated statistical error |
|---:|---:|
| 30–60 | 1.5037 ± 0.1050 |
| 36–60 | 1.4108 ± 0.1690 |
| 44–60 | 1.7905 ± 0.2912 |

All sizes share a divisor of 120. Raw modular gaps produce exactly the
same size exponent. This fixed-time result does not establish the
infinite-time limit. No interpolation, new simulation or pooling with
Campaign 26 is performed.

The four-panel version changes only C. A/B/D retain Campaigns 21–22
full-system measurement data at Ny=30, with endpoint T=60. Their means,
SEMs and heatmap are exactly unchanged. Although C and D now share an
observation time, their slab-only versus full-measurement protocols remain
different; C is not a size sweep of the protocol in A/B/D.

Saved as standalone gap_vs_Ny_T60.pdf/png and compound
purification_ny30_gap_T60_4x1.pdf/png, with caption.tex and raw/sample
summary CSVs. Original and T40 figure assets are preserved; the manuscript
has not been changed. Reproduce with python remake_gap_t60.py.
