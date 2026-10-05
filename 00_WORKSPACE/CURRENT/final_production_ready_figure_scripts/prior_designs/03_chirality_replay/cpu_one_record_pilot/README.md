# One-record CPU flux pilot (production-geometry correction v2)

This is a separate local diagnostic, not an A100 fallback and not part of the production
queue. It uses the canonical CPU entry point `classA_U1FGTN.run_markov_circuit` to generate
one trajectory at zero twist, saves its random site order and realized branch
outcomes compactly, and replays that exact record on a closed uniform-gauge flux grid.
Version 2 explicitly uses the frozen H3 GPU-production wall interval
`max(1, Nx // 4)` for both the dynamics and the wall-resolved observable.  This corrects
the superseded `N20x24_explicit_interface_grid9` run, whose CPU dynamics used the
canonical `Nx // 3` interval while its observable assumed the GPU-production interval.
The old result is preserved for provenance and must not be resumed with this runner.

From the repository root, run:

```bash
python 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/run_cpu_flux_pilot.py \
  --cpu-start 0 --cpu-stop 8 \
  --output-dir 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/results/N20x40_explicit_interface_grid33_production_geometry_v2
```

The CPU interval is Python-style `[start, stop)`, so the command above requests CPUs 0
through 7. The corrected default is a 33-point closed circle on `20x40`, `2*Ny`
cycles, one sample,
`nshell=1`, complex128, random site order, perfect correction, and the explicit topological
interface. `--nx` and `--ny` select the geometry.  Always use a new output directory when
changing the geometry or flux grid.  A matched-trivial control should be launched with
the same geometry, grid, seed, and a separate output directory:

```bash
python 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/run_cpu_flux_pilot.py \
  --cpu-start 8 --cpu-stop 16 --protocol matched_trivial \
  --output-dir 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/results/N20x40_matched_trivial_grid33_production_geometry_v2
```

The script prints canonical circuit progress bars plus a flux-point bar with elapsed time
and rolling ETA. It checkpoints after every flux point and resumes from the same output
directory. It refuses to start below the requested free-space threshold (1 GiB by default)
and never saves a covariance history. The active files are the compressed trajectory
record, initial/final parent states, one latest covariance for exact resume, a compact
observable checkpoint, and the final manifest. Expect tens of MiB, not multi-GiB output.

The manifest includes the zero-flux replay error, the successive final-covariance
Frobenius norm divided by the one-particle dimension, covariance and tracked-subspace
closure diagnostics, minimum replayed branch probability, wall crossings, and per-point
timings. Use `--protocol matched_trivial` and a different output directory for the control.

For a support-terminated slab-only replay, use `support_terminated` (or its matched-trivial
control). The pilot saves the post-exterior cycle-zero state, applies the uniform-gauge
transform to that state at each flux, freezes only the adaptive slab-site word, and reports
a separate cycle-zero covariance residual before checking the final zero-flux replay.
Reapplying the exterior canonical projections is idempotent on this saved cycle-zero state.

```bash
python 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/run_cpu_flux_pilot.py \
  --nx 16 --ny 20 --cycles 40 --grid-points 3 --tracked-modes 16 \
  --protocol support_terminated --cpu-start 0 --cpu-stop 8 \
  --output-dir 00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/03_chirality_replay/cpu_one_record_pilot/results/N16x20_support_terminated_grid3_gauge_init_v2
```
