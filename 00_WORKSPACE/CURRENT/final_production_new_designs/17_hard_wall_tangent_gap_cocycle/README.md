# Hard-wall tangent-gap and endpoint-cocycle campaign, execution v2

This self-contained two-lane A100 bundle measures trajectory-resolved,
finite-time pure-state tangent gaps.  It deliberately uses the quenched order
of operations: construct the endpoint cocycle for one Born trajectory, extract
its logarithmic singular spectrum and five slowest physical gaps, and only then
average the gaps over the 100 independent trajectories.  It never diagonalizes
an averaged cocycle.

## Locked scientific contract

- `Nx=20`, hard/support-truncated domain wall, `alpha_2=30`, and `nshell=1`
- `Ny=24,28,32` on the 21-point descending mutual-information `alpha_1` grid
- `Ny=36,40` at `alpha_1=3,1`
- 100 pure half-filled random trajectories per case
- raster-`y` ordering, perfect correction, no postselection, and complex128
- exactly `2*Ny` cycles accumulated from cycle 1; no burn-in or alignment window
- canonical `classA_U1FGTN_gpu.run_markov_circuit` execution

There are 67 cases, 6,700 sample rows, and 375 durable tasks. New batch sizes
are 25/20/15/10/10 at `Ny=24/28/32/36/40`, respectively. Lane A owns 190
tasks and lane B owns 185; their task tables are disjoint. The only larger
task is the already completed 25-row v1 archive, imported read-only.

The v1 `Ny=40, alpha_1=1, samples=0..24` task finished scientifically and
published its NPZ/completion pair, but took 6,120 seconds (1.70 hours).
It exceeded the one-hour interruption-cost gate, not the 38-GiB GPU-memory
limit (peak reserved was 13.3 GiB). The v1 output folder and receipt stay
unchanged. V2 pins the exact v1 file SHA-256, byte count, task/sample identity,
seed, scientific configuration, and source hashes. If that pair is missing or
changed, v2 stops rather than silently recomputing a 25-row task. V2 analysis
explicitly merges those first 25 independent v1 trajectories with 75 new v2
trajectories for that one case; the other 66 cases are entirely v2.

The six `Ny=24,28,32`, `alpha_1=1,3` cases reuse the 600 checksum-verified hard-
wall records from slot 09. Smaller v2 tasks select exact sample slices from
those unchanged slot-09 archives and retain the original source-batch seeds
per sample. Other records exist only in local `/content`
scratch long enough to replay the same physical trajectory with tangent
propagation.  They are not copied to Drive.

## Gap and cocycle convention

For each trajectory, the occupied and empty one-leg endpoint blocks have
singular logs `log_sigma_o` and `log_sigma_e`.  The saved physical gaps follow
the existing repository convention

```text
Delta_ij = -2 * (log_sigma_o_i + log_sigma_e_j) / (2*Ny).
```

The five finite pair rates closest to zero are retained, together with their
occupied/empty indices and numerical-null counts.  These are finite-time
endpoint tangent gaps, not an equilibrated or asymptotic Lyapunov spectrum.

Only `Ny=40`, `alpha_1=1,3` saves the whole scale-separated chronological
half-cocycle for every sample.  Its array shape is `(samples,880,1600)` in the
lossless hard-wall active-input representation.  The inactive exterior input
directions are omitted; the full spatial output is retained.  The 200 products
occupy about 4.2 GiB before small metadata overhead.

## Running in Colab

Upload this directory to
`MyDrive/final_production_new_designs/17_hard_wall_tangent_gap_cocycle`, then
open both notebooks in separate A100 runtimes:

- `run_hard_wall_tangent_lane_a.ipynb`
- `run_hard_wall_tangent_lane_b.ipynb`

Both real ten-sample `Ny=40` calibration tasks completed below one hour and the
38-GiB allocator ceiling, so the generated notebooks now default to
`MAX_NEW_TASKS=None` for production resume. Lane A verifies the saved v1 task;
both lanes validate and skip all durable NPZ/JSON pairs before continuing. Set
`MAX_NEW_TASKS` to an integer only when deliberately limiting a session. If any
new task exceeds a gate, its result is still published, but the queue stops
before another task so the batch design can be revised without losing valid
science.

The runner shows an outer lane `tqdm` bar and the canonical engine's cycle bar
for each acquisition and replay phase.  It computes under `/content`, writes a
temporary Drive file, reopens and checksums it, atomically renames it, and
publishes the completion JSON last.  There is no Drive API, dashboard, lease,
or pointer protocol.

The notebook now relays the runner's stdout and stderr through a live pipe,
including carriage-return `tqdm` updates. The update interval is five minutes
to keep a multi-day campaign below Colab's cell-output truncation limit while
still exposing live cycle progress. An already-executing Colab cell keeps its
loaded notebook code; do not interrupt a healthy batch solely to see the bars.
Reopen the updated notebook after that batch finishes or disconnects.

The padding hotfix also normalizes occupied-frame arrays when a smaller v2 task
crosses two immutable slot-09 archives whose final-frame padded rank widths
differ. The observed failure was `672` versus `688`; only zero columns beyond
each row's declared rank are added. Completed pre-hotfix v2 outputs remain
accepted when every scientific source hash matches and only the runner hash is
the pinned pre-hotfix value, so a restart skips the 90 lane-A tasks that were
already durable.

Current Python-3.13 Colab images can ship PyTorch built for CUDA 13.0 without
the matching `libnvrtc-builtins.so.13.0`, causing the first complex CUDA
exponential to fail before model construction. The runner now detects only
that exact mismatch, installs the pinned NVIDIA
`nvidia-cuda-nvrtc==13.0.88` runtime when necessary, adds its library directory
for the child process, and re-executes itself once. This changes no scientific
source, configuration, task boundary, seed, or output schema. Outputs from the
two earlier runner identities remain accepted only when every other source hash
matches exactly.

The pinned NVIDIA wheel stores these libraries under
`site-packages/nvidia/cu13/lib` (rather than the older
`site-packages/nvidia/cuda_nvrtc/lib` layout). Runtime discovery accepts both
layouts and still requires the exact `libnvrtc-builtins.so.13.0` soname.

After both lanes report complete, set `RUN_ANALYSIS=True` in either notebook.
The analysis writes the raw `(67,100,5)` gap table, means, SEMs, and fixed-seed
10,000-draw whole-trajectory bootstrap intervals under `OUTPUT_ROOT/analysis`.

## Output locations

```text
slot-09 input:
/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1

saved v1 import (never modified):
/content/drive/MyDrive/classA_final_production_outputs/hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v1

new v2 output:
/content/drive/MyDrive/classA_final_production_outputs/hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2
```
