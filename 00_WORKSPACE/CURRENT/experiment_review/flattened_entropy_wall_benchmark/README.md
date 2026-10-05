# Flattened-parent entropy benchmark

This analysis applies the endpoint entropy estimators from the technical report
to the deterministic half-filled ground state of the same hard-wall flattened
OW parent,

\[
H_{\mathrm{OW}}=\sum_R\left(P_{A+}+P_{B+}-P_{A-}-P_{B-}\right).
\]

The comparison is deliberately estimator matched. Both the monitored endpoint
data and the flattened reference use `Nx=20`, `nshell=1`, `alpha_1=1`,
`alpha_2=30`, hard/support-truncated walls at `x=5,15`, the fit window
`Ay=8..Ny//2`, half-strip anchoring, and equal total weight per circumference.
The wall observables integrate the von Neumann contour over `x={4,5,6}` and
`x={14,15,16}`.

The flattened reference is evaluated for `Ny=30,35,40,45,50,55,60`. Its
conserved-momentum blocks are diagonalized exactly and the lowest half of the
one-body levels are occupied. Translation invariance along `y` makes one
relative-origin contour the exact periodic-origin average. It is a
deterministic benchmark and therefore has no trajectory sampling SEM.

The monitored full-strip comparison uses the verified hard-wall v2 endpoint
campaign (140 five-trajectory shards, 100 trajectories per size). The
wall-resolved comparison uses the completed Lane B portion of the independent
all-`Ay` contour campaign (100 shards, 100 trajectories at
`Ny=30,35,40,45,55`).

## Main result

The joint anchored fits give:

| Observable | monitored dynamics | flattened parent | target |
|---|---:|---:|---:|
| full-strip `c1=3m` | `1.04295 +/- 0.00183` | `0.99935` | `1` |
| left-wall weight `3mL` | `0.51865 +/- 0.00133` | `0.49929` | `1/2` |
| right-wall weight `3mR` | `0.48722 +/- 0.00160` | `0.49929` | `1/2` |

The deterministic left and right results agree to about `4e-13`. Their sum is
`0.99858`, while the paired monitored result is `1.00587 +/- 0.00197`. The
slight undershoot of the right monitored wall is therefore not evidence of a
one-sided bound or a built-in estimator bias: the same finite-size estimator
can lie on either side of the asymptotic value. The opposite left/right
monitored deviations instead diagnose how the finite-time contour weight is
distributed between the predeclared wall windows and their tails.

## Reproduction

Run:

```bash
python compare_flattened_entropy_wall.py
```

Verified deterministic products are cached under `flattened_cache/`. Every
cache file has a completion receipt binding its scientific identity, byte
count, and SHA-256. `analysis_manifest.json` records the verified campaign
inventory, closure diagnostics, fit contract, and hashes of all tables and
figures.

The report-facing comparison is
`figures/dynamics_vs_flattened_entropy_coefficients_2x1.pdf`. The other two
figures expose the corresponding anchored collapses directly.
