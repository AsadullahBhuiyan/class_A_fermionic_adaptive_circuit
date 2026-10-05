# Preserved B1 signed controller-frame CPU pilot

This package implements the exploratory B1 calculation in the working numerical
campaign. It compares the fixed-charge Ky Fan residual of the controller frame with
wrong-outcome activity from disjoint four-trajectory training and test sets.

Launch the complete CPU campaign in tmux:

```bash
bash experiment_review/b1_controller_frame/launch_b1_tmux.sh
```

Useful modes are:

```bash
bash experiment_review/b1_controller_frame/launch_b1_tmux.sh --dry-run
bash experiment_review/b1_controller_frame/launch_b1_tmux.sh --preflight-only
bash experiment_review/b1_controller_frame/launch_b1_tmux.sh --resume <campaign_id>
```

The launcher uses at most four physical cores and caps every numerical library at one
thread. By default it waits when four sufficiently idle cores cannot be found;
`CPU_LIST`, `MAX_WORKERS`, `BLAS_THREADS`, and `ALLOW_BUSY=1` are explicit overrides. Each
numerical trajectory calls the canonical CPU
`classA_U1FGTN.run_markov_circuit` entry point.

The four controller constructions are the explicit topological--trivial interface, its
all-trivial full-geometry control, the support-terminated interface, and an all-trivial
control retaining the same support boundary and active update count. The latter is a
generic-boundary control; removing its support boundary would not be geometry matched.

B1 is descriptive and nonblocking. Its report gives uncertainty-qualified profile
overlap and wall-to-bulk contrast relative to each matched control, but it does not
automatically launch a larger sweep or reinterpret the static minimizer as a tangent or
mean-channel Hamiltonian.

## Preserved partial campaign

Campaign `20260816_225717` was stopped intentionally on 2026-08-17 after completing eight
static cases and eight trajectories. Its directory is immutable partial provenance and
must not be resumed or pooled with the simplified GPU protocol. Configuration v2 had
corrected the v1 `/proc/stat` CPU-token parser and superseded queued campaign
`20260816_205624`; the earlier `20260816_204900` and `20260816_204938` attempts remain
aborted provenance artifacts.

The replacement production calculation lives in
`final_production_ready_figure_scripts/prior_designs/06_b1_controller_frame/`. It no longer performs
charge-sector weighting or activity-profile prediction. It tests finite-window approach
to the signed controller-frame ground-state manifold using two `20x48`, 100-trajectory
A100 cases and is the final nonblocking experiment.
