# Frame-native v4 migration

The active pure-trajectory engine revision is
`production_10sample_v4_occupied_frame_cycle_resolved`.  It is intentionally
separate from the frozen P1 archive and the v3 jobs that were already running.
No v1/v2/v3 archive is overwritten or resumed with the new engine hash.

For a pure initial state, `run_markov_circuit` now resolves
`state_representation="auto"` to `physical_frame` on both CPU and GPU.  Mixed
initial states and mean-replacement channels remain covariance-based.  The GPU
state is a padded batch of orthonormal occupied frames with a rank per
trajectory; rank changes are therefore allowed within one shard.

Maintained v4 production observers use `native_cycle_observer`.  Chern,
density, purity, entropy, correlator, Bott, local-marker, and convergence
products are computed from frame rows and occupied-space overlaps.  Declared
observables retain the complete sample axis and cycle coordinate `0..T`.
Full covariance reconstruction occurs only for an explicitly declared state
checkpoint or a legacy callback/return, and each reconstruction is recorded in
the runner metadata.  Ordinary pure v4 shards set
`require_no_covariance_materialization=True`.

## Required validation before production

Run `00_validation/run_production_bundle.ipynb` on an A100 40 GB runtime.  The
production validation includes a frozen-record 16x16, 32-cycle, 10-trajectory
frame/covariance comparison at every cycle and reports runtime, peak allocated
memory, channel throughput, rank range, Gram residual, branch disagreements,
branch-probability error, covariance error, and materialization reasons.  Do
not launch v4 production unless every validation check passes.

The local CPU smoke suite is the same code path at reduced geometry.  It is not
a substitute for the A100 performance and memory report.

## Preserved revisions

- `production_25sample_v1` P1 remains frozen and scientifically passed with the
  already recorded contract qualification.
- `production_10sample_v3_fixed_nx20_ny20_30_40_50_60` remains provenance for
  completed or in-flight jobs.
- `01_p1_existing_completion/src/classA_U1FGTN_gpu.py` and engine copies inside
  frozen result archives remain pinned to their historical hashes; they are not
  maintained v4 source copies.
