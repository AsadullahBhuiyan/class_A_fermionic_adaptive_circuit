# Why hard-wall global charge spread need not spoil boundary scaling

Analysis date: 2026-09-14. Reproduce with `python
00_WORKSPACE/CURRENT/experiment_review/hard_soft_global_charge/analyze_charge_origin.py`
from the repository root. Results are in `outputs/charge_origin/`.

## Finding

The large hard-wall **across-trajectory** total-charge variance is predominantly
in the exterior, rather than the topological slab. Independently, matched-size
endpoint entropy/charge datasets show hard-wall scaling coefficients closer to
the reference value one despite their larger total-charge variance. These are
compatible observations, not contradictory diagnostics of state quality.

For dense support, Nx=32, Ny=24, 100 trajectories per wall protocol:

| Variance contribution | Hard wall | Soft wall |
| --- | ---: | ---: |
| Slab conditional mean charge | 6.289 | 3.115 |
| Exterior conditional mean charge | 96.694 | 3.611 |
| Twice slab/exterior covariance | -1.402 | 1.259 |
| Total charge | 101.582 | 7.985 |

The exterior variance is 95.2% of the hard-wall total variance; this is a ratio,
not an additive independent fraction, because the covariance also contributes.
The total hard/soft variance ratio is 12.7, but the slab ratio is only 2.02.
Dense Nx=24,28 and the saved nshell=1 cases also have exterior-dominated
hard-wall variance. Thus the effect is not confined to one endpoint geometry.

## Exact decomposition and protocol interpretation

For each trajectory, integrate the saved density over the inclusive wall-to-wall
slab and its complement. The analysis verifies that their sum equals the saved
occupied-frame rank Q, and checks

```text
Var_trajectories(Q) = Var_trajectories(q_slab)
                   + Var_trajectories(q_exterior)
                   + 2 Cov_trajectories(q_slab, q_exterior).
```

These regional quantities are conditional expectation values, not intrinsic
regional quantum variances. Each saved pure Slater determinant has definite
total Q and zero intrinsic full-system charge variance. Its total Q can differ
from that of another trajectory.

The hard protocol combines support truncation with Born-conditioned exterior
occupation preparation and slab-only subsequent dynamics. The exterior is
therefore frozen after preparation. The soft protocol is untruncated and
updates the exterior as well. This explains why hard-wall total charge can
retain exterior preparation variability without implying a poorly prepared
slab. It is a comparison of these whole protocols, not an isolated intervention
on the support mask. No claim is made that exterior variability is the only
source of hard/soft differences.

## Same-trajectory boundary check

Use the separate hard/soft batched endpoint entropy-charge campaigns, Nx=20,
Ny=40,60, nshell=1, S=100 each, endpoint cycle 2Ny. Fit origin-averaged full-x
strip curves against log(sin(pi Ay/Ny)) on Ay=8,...,Ny/2 with a free intercept.
Extract c_eff=3 m_S and k_eff=pi^2 m_F, with F the intrinsic subsystem charge
variance. These linear fit slopes average to the authoritative mean-curve fit;
the reported errors below are trajectory SEMs, not regression residual errors.

| Wall | Ny | Total charge variance | c_eff | k_eff |
| --- | ---: | ---: | ---: | ---: |
| Hard | 40 | 126.018 | 1.0557 +/- 0.0050 | 1.0535 +/- 0.0051 |
| Soft | 40 | 14.149 | 1.0836 +/- 0.0077 | 1.0859 +/- 0.0084 |
| Hard | 60 | 200.082 | 1.0402 +/- 0.0039 | 1.0389 +/- 0.0040 |
| Soft | 60 | 15.149 | 1.0676 +/- 0.0064 | 1.0689 +/- 0.0069 |

Within each geometry/protocol, correlate absolute total-charge offset from half
filling with c_eff, k_eff, their absolute deviations from one, and the entropy
fit residual. Hard-wall correlations with c_eff are 0.017 (Ny=40) and 0.100
(Ny=60); pointwise trajectory-bootstrap 95% intervals are [-0.137,0.143] and
[-0.152,0.376]. None of the 20 exploratory Pearson tests passes a 5%
Benjamini-Hochberg correction. This is **not** evidence of exact statistical
independence: the Ny=60 intervals still permit moderate associations. The soft
Ny=60 pointwise percentile-bootstrap intervals for some coefficients exclude
zero even though Pearson tests do not; these are different inferential
procedures, and do not establish a multiplicity-controlled discovery.

Conclusion: these saved endpoints support separating global filling stability
from boundary scaling quality. They do not prove a common asymptotic
universality class, stationarity, or a causal explanation of the remaining
finite-size coefficient biases. Regional decompositions and boundary fits come
from different campaigns; only correlations within a boundary campaign pair
observables on the same trajectories.

## Data and validation

- Spatial split: primary `wall_pump_width_endpoints_s100_v1/endpoints` collection
  under `frozen_record_flux_charge_pilot/imported_endpoints`, 200 five-sample
  shards, 1,000 trajectories. The S25 bridge collection is excluded.
- Boundary check: bundles `05_hard_wall_entropy_charge_batched_v2` and
  `10_soft_wall_entropy_charge_batched_v2`, Ny=40,60 only, 80 five-sample shards,
  400 trajectories. No pooling with the spatial dataset or older campaigns.
- All 280 input archives pass byte-count and SHA-256 checks against their
  completion JSONs. Every analyzed cell has sample IDs exactly 0,...,99.
- `source_manifest.json` records input paths and hashes. CSV files retain
  trajectory-level data, the decomposition, fit summaries, correlations,
  pointwise bootstrap intervals (5,000 resamples; seed 2026091402), and adjusted
  p-values. Existing data and campaign code were not changed.

## Figure caption

`charge_origin_and_boundary.pdf` (also 300-dpi PNG): (a,b) Across-trajectory
variance decomposition at dense support, Nx=24,28,32, Ny=24, after 48 cycles;
S=100 independent trajectories per point. Slab boundaries are inclusive. Lines
join descriptive sample variance/covariance estimates (no uncertainty bars).
(c,d) Individual-trajectory endpoint c_eff versus absolute total-charge offset
for Nx=20, Ny=40,60, nshell=1, after 2Ny cycles; S=100 independent trajectories
per size and wall protocol. Curves are averaged over strip origins within each
trajectory before fitting, on Ay=8,...,Ny/2; the dashed line is c_eff=1.
All datasets use pure default half-filled initialization, raster_y ordering,
perfect correction, alpha_1=1, alpha_2=30, and complex128 canonical GPU
dynamics. Hard-wall initialization includes exterior occupation preparation;
soft-wall initialization does not. Panel axes have different scales. These
are finite-cycle endpoint comparisons, not demonstrated steady-state limits.
