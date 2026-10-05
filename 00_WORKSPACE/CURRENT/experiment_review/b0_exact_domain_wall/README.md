# B0 exact-domain-wall campaign

This package performs the deterministic class-A domain-wall benchmark described in
Campaign B0. It imports the CPU `classA_U1FGTN` implementation and never modifies or
duplicates the adaptive-circuit dynamics engine.

Launch from anywhere inside the repository:

```bash
bash experiment_review/b0_exact_domain_wall/launch_b0_tmux.sh
```

Useful modes:

```bash
bash experiment_review/b0_exact_domain_wall/launch_b0_tmux.sh --dry-run
bash experiment_review/b0_exact_domain_wall/launch_b0_tmux.sh --preflight-only
bash experiment_review/b0_exact_domain_wall/launch_b0_tmux.sh --resume <campaign_id>
```

The launcher chooses one logical CPU from each idle physical core, balances the choice
across NUMA nodes, and starts a detached tmux session. `CPU_LIST`, `MAX_WORKERS`,
`BLAS_THREADS`, and `ALLOW_BUSY=1` are explicit overrides. The current numerical
configuration is locked in `campaign_config.v2.json`; v1 remains unchanged for provenance.
Numerical parameters are not accepted from the command line, and resume loads the exact
configuration recorded by the campaign manifest.

## Audited v2 result

The authoritative completed run is `results/20260816_191957/`. It selects `Nx=20` by
requiring every single-geometry calibration to pass at both the accepted width and the
next larger scanned width. All acceptance-ledger entries pass, including the coupled and
hard-exterior correlator, endpoint modular handedness, physical response, and twist flow.

Run `results/20260816_184251/` is preserved but superseded for interpretation. Its
center-injected absolute-COM modular estimator was a protocol mismatch with the legacy
entanglement-boundary experiment, and its mass-only gate selected `Nx=12` before the
coupled local observables had converged. The complete reconciliation is saved in the v2
run under `processed/legacy_reconciliation.json` and
`processed/tables/modular_protocol_reconciliation.csv`.

The v2 modular calculation evolves normalized sources separately at both entanglement
endpoints and both physical walls, for one- and three-column transverse support. It fits
the signed symmetry-cancelling endpoint statistic

```text
D_w(t) = [ybar_(w,0)(t) + ybar_(w,L-1)(t) - (L-1)] / 2
```

on modular time `0.1 <= t <= 2.0`, with three fit-window sensitivities. The separately
saved legacy aggregate reproductions retain initial charges 8 and 24. Modular-time slopes
are handedness diagnostics and are not compared in magnitude with physical velocities.

The full-width strip entropy is the genuine subsystem entropy. The two three-column wall
windows are unnormalized entropy-contour decompositions of that same strip. Each wall
window includes both entanglement cuts and is expected to carry half the full logarithmic
slope; it is not an isolated one-wall subsystem entropy.

Run the fast validation suite with:

```bash
python -m unittest discover -s experiment_review/b0_exact_domain_wall/tests -v
```
