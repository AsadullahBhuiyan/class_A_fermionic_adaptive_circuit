# S100 uniform bulk-validation analysis

## Campaign status

The bulk-validation campaign was declared scientifically complete on
2026-09-09. The accepted balanced evidence comprises `L=12,16,20,24,28,32`,
all three `n_shell=1,2,dense` constructions, `S=100` independent trajectories
per case, and every saved cycle from 0 through 40. The planned `L=36,40`
top-up is retired. Partial `L=36` data are preserved but excluded from the
complete-grid headline comparison. See [`CAMPAIGN_CLOSURE.md`](CAMPAIGN_CLOSURE.md)
for the exact scope and decision.

This analysis reads the verified, read-only Google Drive import at

`00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/uniform_perfect_correction_40cycle_s100_maxl24_batched_v1/`

and recreates the four-panel perfect-correction validation figure using the new
100-trajectory ensemble.  The campaign contains the 12 combinations of
`L=12,16,20,24` and `n_shell=1,2,dense`, with compact real-space Chern and
global-charge observations at every cycle from 0 through 40.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/uniform_bulk_validation_analysis/build_uniform_bulk_validation_figure.py
```

The script verifies every NPZ/completion pair before analysis.  Outputs are
written below `outputs/uniform_perfect_correction_40cycle_s100_maxl24_batched_v1/`.
The input campaign is never modified.

## Size extension through L=32

`build_uniform_bulk_chern_L12_L32.py` combines the complete S100 cases from the
small-size campaign with the complete `L=28,32` cases from the separately
versioned large-L campaign. It produces a horizontal three-panel comparison,
one panel for each `n_shell`, using every saved cycle and deterministic 95%
whole-trajectory bootstrap bands. The two sampling revisions remain explicit
in the accompanying summary; they are joined only because their size grids are
disjoint and their locked scientific protocols agree.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/uniform_bulk_validation_analysis/build_uniform_bulk_chern_L12_L32.py
```

## Two-panel convergence and late-time size figures

`build_uniform_bulk_chern_convergence_figures.py` uses the same verified
`L=12..32` S100 grid to produce two manuscript-scale compound figures. The
first is a 3.375-inch single-column `2 x 1` layout with curves for
`L=16,24,32`: sample-averaged `C_G` versus cycle above and
`|mean(C_G)-1|` on a logarithmic scale below. System size is encoded by
red, green, and blue, while shell construction is encoded by marker and line
style. The second is
a 3.375-inch single-column `2 x 1` finite-size figure retaining all six sizes
that averages the `100 x 20` sample-cycle ensemble over cycles 21--40. Figure 3
uses the sample-to-sample standard error at each cycle. Figure 4 uses
`std(C_G, ddof=1) / sqrt(2000)` across its 2,000 sample-cycle values. The
absolute-deviation intervals propagate the mean plus or minus one standard
error through the absolute-value transform.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/uniform_bulk_validation_analysis/build_uniform_bulk_chern_convergence_figures.py
```

## Global-charge fluctuation figure

`build_uniform_bulk_charge_fluctuation_figure.py` combines the same complete
`L=12..32` S100 grid and analyzes the integer total charge relative to half
filling. It produces a 3.375-inch single-column `3 x 1` figure containing the
late-time relative deviation versus size, the every-cycle filling-fraction
deviation at `L=32`, and the cycle-40 charge histogram for `L=32`,
`n_shell=1`. Cycles 21--40 are averaged within each trajectory before the
finite-size mean and standard error are calculated. All uncertainties are
direct sample-to-sample standard errors with `ddof=1`.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/uniform_bulk_validation_analysis/build_uniform_bulk_charge_fluctuation_figure.py
```
