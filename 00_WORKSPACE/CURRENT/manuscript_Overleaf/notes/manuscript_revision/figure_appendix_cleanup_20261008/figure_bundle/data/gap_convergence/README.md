# Gap convergence at fixed system size

Existing downloaded data only; no new dynamics or Campaign 28 output is used.
All 160 NPZ/completion pairs were independently rehashed and their metadata and
sample IDs validated: 20 full-measurement Campaign 21 shards and 140 slab-only
Campaign 13 shards. Raw data and manuscript figures remain unchanged.

Both campaigns use Nx=20, hard/support-truncated walls, alpha1=1, alpha2=30,
nshell=1, complex128, maximally mixed initialization, perfect correction and
raster_y. They are NOT pooled:

- Campaign 21: global maxmix, meas_slab_only=False, no exterior preparation;
  Ny=30, 100 samples, every cycle 0..60. This is the relevant existing protocol
  comparison for the new Campaign 28.
- Campaign 13: slab-only with Born-conditioned exterior preparation;
  Ny=20,24,30,36,44,56,60, 100 samples each, selected spectra through 4Ny.
  These longer histories provide supporting fixed-size convergence diagnostics
  but cannot establish full-measurement convergence at the other sizes.

## Estimator

At each positive saved cycle, compute for each trajectory separately:

    g_mod(t) = min_j |log[(1-nu_j(t))/nu_j(t)]|
    Delta(t) = g_mod(t)/(2t).

Only occupations strictly between 1e-9 and 1-1e-9 have finite modular energy;
pure caps are excluded. There are no fully capped/censored sample-times in
these data. Cycle zero is omitted because the time-normalized rate is undefined.
The minimum is taken BEFORE averaging over trajectories, not from an averaged
spectrum or covariance. Bands show ordinary sample SD/sqrt(100), no bootstrap.
For time differences, form the difference inside each trajectory and then its
mean and SEM. Percent changes are ratios of ensemble means, with a delta-method
SEM retaining their covariance. Time points are not independent replicates.

## Results

Full-measurement Nx20, Ny30:

| Cycle | Mean Delta | SEM | Mean raw modular gap |
|---:|---:|---:|---:|
| 20 | 0.020861 | 0.001461 | 0.834425 |
| 40 | 0.021278 | 0.001343 | 1.702279 |
| 60 | 0.025034 | 0.001571 | 3.004116 |

The 20-to-40 paired change is 0.000418 +/- 0.001830; the apparent plateau over
that interval is compatible with sampling fluctuations. The 40-to-60 paired
change is 0.003756 +/- 0.001491, or 17.65 +/- 7.51 percent (one sampling SEM).
This is evidence of remaining finite-time drift, not a precision determination
of an infinite-time gap. Forty cycles are useful as a common-time comparison,
but these data do not demonstrate convergence of the asymptotic rate by then.

In the distinct slab-only protocol, Delta increases by 22–35 percent between
2Ny and 4Ny across the seven sizes. Even the last interval, 3Ny to 4Ny, shows
mean increases of roughly 7–12 percent. The trend slows but is not an exactly
flat plateau; neither the final value nor a fitted asymptote is assumed exact.

The raw modular gap grows rather than saturating. The observed time dependence
is therefore not simply a constant numerator divided by t. A nonzero converged
rate would correspond to asymptotically linear g_mod(t), not saturation of g_mod.
No time fit, universal exponent, or extrapolated infinite-time value is imposed.

## Outputs

- full_measurement_ny30_gap_convergence.pdf/png: two panels, mean Delta(t)
  and mean g_mod(t), Nx20 Ny30 full measurement, 100 samples, cycles 1..60.
- slab_only_fixed_size_gap_convergence.pdf/png: same diagnostics for the seven
  separately evolved slab-only sizes. Each curve follows one fixed size.
- In both figures, shading is one trajectory SEM and the vertical dashed line
  marks t=40. Dimensions are 3.375 x 4.5 inches; PNGs are 300 dpi.
- cycle_gap_summary.csv and sample_cycle_gaps.csv/npz retain all sample-resolved
  gap histories and ensemble summaries; paired_time_changes.csv retains paired
  differences, uncertainty and relative changes.
- analysis_manifest.json binds input receipts/results, executed sources and
  output hashes. Reproduce with `python analyze.py`; no GPU is needed.
