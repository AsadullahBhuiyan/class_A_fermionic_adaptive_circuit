# CPU tangent replay of the completed slot-09 acquisition

This stage consumes the checksum-verified 1,200-trajectory acquisition under
`../gpu_data/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1`.
It does **not** rerun or alter the physical GPU campaign.

For every frozen trajectory it replays the prepared occupied frame and ordered
record through the canonical CPU `classA_U1FGTN.run_markov_circuit` entry point.
It propagates the exact occupied/empty one-leg tangent blocks without a Choi
covariance or covariance history. The replay endpoint must agree with the saved
GPU endpoint projector before the result is committed.

## Endpoint replay gates (v3)

The superseded v1 pilot used a relative projector-Frobenius threshold of
`2e-9`. Of the first three hard-wall `Ny=24`, `alpha_1=1` records, one matched
exactly while two completed replays differed from their saved GPU endpoints by
`1.557e-8` and `2.148e-8`. These are small CPU-versus-GPU floating-point
differences after 48 fixed-record cycles, but they correctly caused v1 to stop
rather than silently alter its preregistered gate.

The separately versioned v2 output used a `1e-6`
relative-projector gate. This is over 46 times the largest observed pilot error
while remaining a strict cross-device replay check. Every NPZ and completion
JSON preserves both the raw measured error and the threshold. The acquisition
records, measurement outcomes, CPU dynamics, tangent products, seeds, and
scientific observables were unchanged.

V2 then exposed a distinct validation bug at hard-wall `Ny=24`, `alpha_1=1`,
sample 18. Its full-window and late-window CPU replays each independently
matched the saved GPU endpoint, but the two CPU endpoint projectors differed by
`1.5454381073764906e-8`. The runner had compared those independent floating-point
replays with an uncalibrated hardcoded `1e-12` threshold, so it rejected the
valid result after about 51 minutes and stopped the queue fail-closed.

Production v3 retains the `1e-6` CPU-to-GPU gate and adds an explicit `2e-6`
full-versus-late projector gate. The latter is the triangle-inequality envelope
for two endpoints that each pass the `1e-6` reference gate, and is about 129
times the measured sample-18 discrepancy. Every v3 NPZ and completion JSON
stores the raw full, late, and cross-replay errors, both thresholds, and both
calibration revisions. V1 and v2 directories remain preserved and are never
pooled with v3.

## Permanent output per trajectory

- normalized dense `final_cocycle_hat` and scalar
  `final_cocycle_log_scale`, representing
  `K_T:0 = exp(log_scale) * final_cocycle_hat`;
- per-cycle QR log increments, cumulative logs, null masks, and branch
  probability diagnostics;
- full-window and late-window occupied/empty singular-value logs;
- the 16 slowest particle-hole tangent rates and their input/output vectors and
  x profiles;
- input/source/configuration identity and gauge-invariant endpoint replay error.

The covariance superoperator is represented by
`J_T:0[H] = K_T:0^dagger H K_T:0`; its enormous `d^2 x d^2` matrix is never
constructed. Dense one-cycle Jacobians, intermediate covariances, and
intermediate occupied frames are not permanent campaign products.

Selected dense one-cycle maps can be materialized outside the campaign output:

```bash
python run_cpu_tangent_replay.py \
  --materialize-task hard_Ny024_a1-1_sample-000_global-0000 \
  --materialize-cycles 1,24,48 \
  --materialize-root /tmp/pure_tangent_jacobians \
  --threads-per-worker 28 --skip-input-checksums
```

Each file stores a scale-separated `K_t(C_{t-1})`; the corresponding covariance
action is `J_t[H] = K_t^dagger H K_t`. Using `--materialize-cycles all` is
supported but intentionally opt-in because it creates tens of dense matrices.

Each sample has its own uncompressed NPZ and checksummed completion JSON. A
restart verifies and skips valid pairs, so interruption loses only active
sample computations.

## Parallel production

The production launcher checksum-audits all acquisition inputs once, then runs
a bounded, continuously refilled process pool. While the pre-existing
28-thread many-body campaign occupies CPUs 84--111, it runs six simultaneous
tangent trajectories with 14 BLAS threads each on CPUs 0--83. If those CPUs are
free at launcher startup, it instead runs eight simultaneous trajectories
across all 112 logical CPUs. Every CPU group is explicit and disjoint; a fast
worker takes the next pending sample immediately rather than idling behind a
slower sample.

```bash
bash launch_tmux.sh
tmux attach -t pure_tangent_cpu_replay_s100_v3
```

The outer `tqdm` bar reports completed samples and continuously displays each
active worker's `full` or `late` cycle. Output and the persistent log live under
`../cpu_data/pure_tangent_cpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v3/`.

Useful checks:

```bash
python run_cpu_tangent_replay.py --report-only
python run_cpu_tangent_replay.py --construction hard --ny 24 --alpha-1 1 --sample 0 \
  --workers 1 --threads-per-worker 28 --cpu-groups 0-27 --skip-input-checksums
```

Omitting `--skip-input-checksums` rereads and SHA-256 verifies all 24 GB of
acquisition inputs. Production does this once at startup. The skip flag is for
short local diagnostics after the imported campaign has already passed that
audit.
