# Uniform bulk-validation campaign closure

Date: 2026-09-09

Decision: **scientifically complete; no further production runs required for
the current paper plan.**

## Accepted evidence

The headline dataset is the complete balanced grid

- `L = 12, 16, 20, 24, 28, 32`;
- `n_shell = 1, 2, dense`;
- `S = 100` independent trajectories per `(L, n_shell)` case;
- uniform geometry with `DW=False`;
- pure half-filled random initialization;
- random-serial perfect-correction dynamics in `complex128`;
- compact real-space Chern and charge observations at every cycle `t=0..40`.

This is 18 complete cases and 1,800 independent trajectories. Every result and
completion pair used by the combined analysis passed its recorded byte-count
and SHA-256 checks. At cycle 40, all 18 trajectory-averaged real-space Chern
estimates lie between `0.9955618488502392` and `0.9999996092068666`. The figure
uses deterministic 95% whole-trajectory bootstrap intervals from 20,000
resamples.

The canonical combined analysis artifact is
`outputs/uniform_perfect_correction_40cycle_s100_L12-L32/fig02_uniform_bulk_chern_L12_L32_s100_summary.json`.

## Retired extension

The original large-size add-on proposed a complete S100 grid through `L=40`.
At closure, the preserved large-size collection contains:

- complete `L=28`: 300 trajectories across all three shell choices;
- complete `L=32`: 300 trajectories across all three shell choices;
- partial `L=36`: 100 trajectories for `n_shell=1`, 100 for `n_shell=2`, and
  30 for `dense`;
- no completed `L=40` cells.

The 18 outstanding durable batches—70 remaining `L=36,dense` trajectories and
all 300 `L=40` trajectories—are canceled. They are not missing requirements.
Existing partial `L=36` products remain immutable supporting data and must not
be pooled or labeled as a balanced S100 size point.

## Interpretation boundary

“Complete” refers to the scientific bulk-validation claim supported by the
balanced `L=12..32` grid. It does not assert that the superseded original
`L=28,32,36,40` execution table reached 36/36 batches. Repository and Drive
artifacts are retained for provenance; no output is deleted or rewritten by
this decision.
