# Pure tangent-replay acquisition (slot 09)

This is the fast acquisition half of the tangent-cocycle experiment. It runs
only the physical perfect-correction circuit and saves everything needed to
replay each trajectory later. It deliberately does **not** propagate tangent
frames, materialize covariance matrices, or track a Choi state.

## Locked campaign

- `Nx=20`, `Ny=24,28,32`, and `cycles=2*Ny`
- hard and soft domain walls, both with `alpha_1=1,3`, `alpha_2=30`
- `nshell=1`, pure half-filled random initialization, raster-y ordering
- perfect correction, no postselection, complex128
- 100 trajectories per configuration
- execution batches: 50 at Ny=24, 50 at Ny=28, and 25 at Ny=32

There are 12 configurations, 32 resumable batch tasks, and 1,200 independent
trajectory rows. The root seed is `2026090409`; task seeds and batch boundaries
are deterministic.

## Run in Colab

Upload this whole folder to
`MyDrive/final_production_new_designs/09_pure_tangent_replay_acquisition`, open
`run_pure_tangent_replay_acquisition.ipynb`, select an A100 40-GB runtime, and
run the notebook from top to bottom. The configuration cell supports:

- `REPORT_ONLY=True` to verify Drive results without launching CUDA work;
- `MAX_NEW_TASKS=<integer>` to limit the number of new batches in a session.

The notebook copies executable code to `/content` before starting. The outer
progress bar counts the 32 durable batches; the canonical engine prints its
inner cycle progress. Restarting the notebook verifies NPZ/completion pairs and
skips completed batches.

## Durable result format

Each batch produces one uncompressed NPZ and one `.complete.json`. The NPZ
contains, for every sample in the batch:

- the prepared cycle-zero occupied frame and rank;
- the final occupied frame and rank;
- the full ordered raster-y site record;
- bit-packed Born outcomes and controller targets;
- per-cycle and cumulative record log probabilities;
- case-local and campaign-global sample indices.

For the hard construction, the initial frame is captured after the
Born-conditioned exterior product preparation. Replay must therefore pass it
as `frame_init` with `frame_ranks` and `frame_init_prepared=True`. The soft
construction uses `frame_init_prepared=False`.

The physical endpoint correlation and centered covariance are recovered from
the active columns of the saved final frame:

```python
rank = int(final_ranks[sample])
F = final_frame[sample, :, :rank]
P = F @ F.conj().T
C = 2 * P - np.eye(P.shape[0])
```

Intermediate physical frames are intentionally absent. A later replay program
will unpack `record_outcomes_packed`, pass the saved schedule/outcomes through
the canonical engine's frozen-record interface, and compute each tangent
Jacobian and stabilized product then.

## Batched A100 tangent replay

That replay stage is now included in this bundle. Open
`run_pure_tangent_gpu_replay.ipynb` to process the saved records in 48 durable
tasks of 25 trajectories each. The A100 runner executes two frozen-record
windows per task, propagates the complete one-leg tangent basis in a leading GPU
batch, and saves the full scale-separated cocycle plus full/late singular and
slow-mode diagnostics. It does not construct a Choi covariance, covariance
history, dense `d^2 x d^2` superoperator, or per-cycle dense Jacobian.

The GPU replay has its own output revision and completion pairs, so it neither
overwrites the acquisition records nor silently pools with the CPU replay.
See `gpu_tangent_replay/README.md` for the precise contract and paths.

## Resume and publication

Computation and NPZ creation happen under `/content`. The runner copies the
closed NPZ to a temporary DriveFS path, checks byte count and SHA-256 by reading
that path, atomically renames it, and publishes the completion JSON last. A
batch is complete only if both files and all identity/checksum fields verify.

The runner records elapsed time and peak CUDA memory. It stops the queue after
safely publishing a batch if that batch exceeded one hour or 36 GiB reserved
CUDA memory, so batch geometry cannot silently change during the campaign.
