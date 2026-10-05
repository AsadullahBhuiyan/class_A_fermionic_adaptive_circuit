# Square hard-wall Chern relaxation

Reproduce with `python ../../analyze_relaxation.py` from this directory.
This offline analysis uses the previously verified downloaded snapshot: L=20
and 30 each have 100 trajectories; L=40 has only 20 and is preliminary.
No dynamics, deployment files, raw data, or technical-report figures are changed.
The script rechecks all nine NPZ checksums and completion identities, cycle and
sample coverage, and the mean over ten centers before analysis. Full endpoint
frame validation was performed during the original snapshot import.

`all_sizes_all_cycles.pdf` (and 300-dpi PNG) shows the trajectory-averaged Chern
number and its absolute deviation from unity for all sizes over cycles 0–40.
Shading is one sample SEM, with ten centers averaged within each trajectory.
The absolute-deviation interval is the image of the mean ± SEM interval;
zero lower endpoints are clipped to the displayed logarithmic floor of 1e-6.

`early_relaxation.pdf` shows the excess deficit relative to each size's measured
late-time baseline, C_inf = mean over trajectories of the within-trajectory
cycles-21–40 mean. Error bars are the SEM of the paired, trajectory-level
baseline-minus-current-cycle differences. Lines are descriptive log-linear
fits C_inf − mean(C_G(t)) = A exp(−k t) over cycles 2–5, not connecting lines.
Cycle 1 is displayed but not included in these fits.

| L | Samples | k (cycle^-1) | 1/k (cycles) |
|---|---:|---:|---:|
| 20 | 100 | 1.171 | 0.854 |
| 30 | 100 | 1.005 | 0.995 |
| 40 | 20 (partial) | 0.966 | 1.035 |

The practical scale is approximately one cycle per e-fold, with the curves
near their late-time plateau after roughly 5–10 cycles. A single exponential
does not describe the entire transient: fitting cycles 0–3 yields rates near
1.5, while windows beginning at cycle 1 or 2 yield approximately 0.86–1.30.
These window variations are not confidence intervals. No claim of a unique
asymptotic relaxation gap or a resolved size dependence of the rate is made.

`cycles.csv` contains means, SEMs, paired excess deficits and their SEMs;
`rates.csv` contains all five fit-window choices. `summary.json` records
source identities, checksums, formulas, fit results, and generated-file hashes.
All SEMs use ddof=1 over independent trajectories; centers and cycles are not
treated as independent samples. No bootstrap resampling is used.
