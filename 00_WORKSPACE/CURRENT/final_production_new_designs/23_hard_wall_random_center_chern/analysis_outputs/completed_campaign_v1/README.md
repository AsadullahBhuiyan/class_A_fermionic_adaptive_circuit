# Completed hard-wall random-center Chern campaign

Offline analysis of all 13 verified batches: 300 trajectories and 183,000
individual Chern measurements. No circuit dynamics were rerun. Reproduce with
`OPENBLAS_NUM_THREADS=1 python ../../analyze_results.py` from this directory.
Raw data, deployed code, and the manuscript are unchanged.

## Setup and statistics

Nx=20, Ny=20/30/40; alpha1=1, alpha2=30; nshell=1; hard support truncation,
perfect correction, no postselection, raster-y order, periodic boundaries,
complex128. Each geometry has 100 pure trajectories, initially half-filled,
with hard-wall exterior preparation before cycle zero, evolved for 2Ny cycles.
The Chern disk is centered at x=10 with R=4, strictly inside the interfaces
at x=5,15. Ten distinct periodic y centers are redrawn for each trajectory
and cycle using RNG independent of the circuit.

First average the ten Chern values within each trajectory, then take the mean
and sample SEM across 100 trajectories (ddof=1). For time-window summaries,
average cycles within each trajectory before computing SEM. Neither centers
nor successive cycles are counted as independent trajectories. Center variance
is the sample variance across the ten centers (ddof=1), then averaged across
trajectories; it is a spatial-fluctuation diagnostic, not the error on the mean.

## Main results

| Geometry | Endpoint mean ± SEM | Final-20-cycle mean ± SEM | Window |
| --- | --- | --- | --- |
| 20x20 | 0.998240 ± 0.000352 | 0.997772 ± 0.000115 | 21–40 |
| 20x30 | 0.997930 ± 0.000389 | 0.997747 ± 0.000107 | 41–60 |
| 20x40 | 0.997646 ± 0.000381 | 0.997669 ± 0.000104 | 61–80 |

- The mean rises from approximately zero to 0.79–0.80 after one cycle and
  0.967–0.968 after two, reaching approximately 0.997 by five cycles.
- The late-time deviation is approximately 0.22–0.23%, with no resolved
  dependence on Ny in these ensembles. Using the common window 21–40 gives
  0.997772 ± 0.000115, 0.997700 ± 0.000108, and 0.997823 ± 0.000106.
- Paired differences between the final ten and preceding ten cycles are
  -0.000140 ± 0.000272, +0.000161 ± 0.000235, and
  +0.000253 ± 0.000193. These provide no clear evidence for continuing
  late-time drift; they are an exploratory check, not proof of stationarity.
  Long evolution beyond the initial transient does not visibly reduce this
  local Chern deficit. Other observables may relax on different timescales.

## Spatial variation and equilibrium reference

At the endpoints, the mean within-trajectory center variances are
7.07e-5, 9.62e-5, and 1.20e-4, respectively. Their square roots (RMS spatial
standard deviations) are 0.00841, 0.00981, and 0.01094. Final-20-cycle averaged
variances are 1.31e-4, 1.19e-4, and 1.38e-4; thus the endpoint ordering alone
should not be interpreted as a reliable size trend.

Endpoint center medians are 0.999952, 0.999955, and 0.999940, whereas their
1st percentiles are 0.9732, 0.9548, and 0.9601. The distribution is asymmetric:
the lower tail pulls the mean below the value typical of most centers.
These quantiles pool centers descriptively, without treating them as
independent observations for inference. Redrawn centers do not track a fixed
spatial location over time.

The previously saved half-filled ground states of the hard-wall truncated-OW
parent give C_G≈0.99995785 with center variances of order 1e-31. They use the
same partition and estimator, whose source hash is checked by this analysis.
The circuit's late-time deficit is approximately 53–55 times the ground-state
deficit. This is not a failed quantization gate: a finite-disk estimator need
not be exactly integer. The circuit samples a Born-conditioned ensemble with
exterior preparation and fluctuating total charge, whereas the reference is
a deterministic half-filled ground state. Matching topology does not require
these states or their local fluctuations to agree exactly. The present data
do not isolate the contributions of truncation, width, radius, and trajectory
fluctuations. Nx and R are fixed, so this is not a thermodynamic-width or
radius-stability study.

## Charge

Endpoint mean absolute relative deviations from half filling,
100 mean(|Q−NxNy|)/(NxNy), are 1.678 ± 0.129%, 1.170 ± 0.094%, and
1.108 ± 0.085%. These percentages are relative to the half-filled charge,
not the filling fraction. Endpoint charge standard deviations are 8.46,
9.00, and 10.91 particles. Each pure number-conserving Gaussian trajectory
has a definite integer charge; this spread is between records, not intrinsic
quantum charge variance within a trajectory. These are global charges, not
charges restricted to the Chern disk or the topological slab.

## Products and figure caption

- `campaign_summary.pdf` / `.png`: vector and 300-dpi four-panel overview.
- `cycle_statistics.csv`: all cycle means, SEMs, and spatial variances.
- `trajectory_statistics.csv`: endpoint and late-window trajectory statistics.
- `summary.json`: scientific contract, numerical summaries, source identities,
  current input checksums, prior full-array validation identity, and output hashes.

**Hard-wall bulk topology and fluctuations.** (a) Early-time center- and
trajectory-averaged real-space Chern number. (b) Its absolute deviation from
unity over the full evolution; the gray line is the deterministic ground-state
reference. (c) Mean variance across ten centers within each trajectory.
(d) Mean absolute global-charge deviation relative to half filling, in percent.
Colors and markers distinguish Ny at fixed Nx=20, nshell=1. Initially random
pure half-filled states undergo hard-wall exterior preparation and 2Ny cycles
of perfect-correction dynamics. Centers are averaged before trajectories;
bands are one SEM over S=100 independent trajectories (transformed through
the absolute value in panel b). The disk has x0=10 and R=4 with periodic y
coordinates. Panel (a) displays cycles 0–10 only; no fit is performed.
