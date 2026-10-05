# Aggressively batched A100 tangent replay

This is the GPU analysis stage for the completed slot-09 acquisition records.
It does not resimulate new stochastic trajectories: it freezes each saved
schedule and Born record, replays the physical circuit, and propagates the
one-leg tangent map with the canonical GPU engine.

## Workload

- `Nx=20`; `Ny=24,28,32`
- hard and soft walls; `alpha_1=1,3`; `alpha_2=30`; `nshell=1`
- 100 saved trajectories per case, 12 cases, 1,200 trajectory rows
- 25 trajectories per GPU task, 48 independently resumable tasks
- two batched replays per task: cycles `1..2Ny` and `Ny+1..2Ny`
- A100 40-GB-class GPU and complex128

The 25-sample task is an explicit exception to the repository's usual five-
trajectory default. The tangent algebra is dominated by large dense matrix
operations, so a large leading batch is the point of this GPU version. The
first completed task records wall time and peak CUDA memory. The queue stops
before launching another task if it exceeded one hour or 36 GiB reserved.

## Tangent representation

For each saved prepared frame, the runner diagonalizes the restricted
correlation matrix and constructs its occupied/empty basis in a GPU batch. It
passes that batched basis to
`classA_U1FGTN_gpu.run_markov_circuit(..., lyapunov_basis_mode="canonical")`.
The engine evolves and QR-stabilizes the full one-leg map on-device. If `Q_T`,
`R_T` and `U_0` denote its final frame, scale-separated accumulated core, and
initial occupied/empty basis, the saved map is reconstructed as

```text
K_T:0 = Q_T R_T U_0^dagger,
J_T:0[H] = K_T:0^dagger H K_T:0.
```

The runner never constructs the `d^2 x d^2` superoperator. It saves the
normalized `K_T:0`, its logarithmic scale, full/late singular logs, cycle QR
diagnostics, and the 16 slowest occupied-empty tangent pairs and x profiles.
Because monitored trajectories can carry different charge at the same cycle,
occupied and empty block sizes are saved per sample; a common-rank assumption
is never used to obtain GPU batching.

## Resume and progress

Open `../run_pure_tangent_gpu_replay.ipynb` from the Drive bundle and run it top
to bottom. The notebook stages six executable files under `/content`. The
runner shows:

1. one outer 48-batch `tqdm` bar;
2. one two-window bar for the active task;
3. the canonical engine's cycle bar inside each window.

Each task is computed under `/content`, saved as one uncompressed NPZ, copied
to a temporary Drive path, reopened and checksummed, atomically renamed, and
followed by its completion JSON. Restarting verifies and skips completed pairs;
only the active 25-trajectory task is lost on interruption.

The default input and output collections are:

```text
/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1
/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_gpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1
```

CPU-v3 and GPU-v1 replay products derive from the same physical sample rows and
must not be pooled as independent trajectories. They are alternative numerical
realizations and may be used for cross-validation. The runner's conservative
size estimate is about 40 GiB for all 48 uncompressed result files.
