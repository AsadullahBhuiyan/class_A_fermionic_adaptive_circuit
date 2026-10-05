# Hard-wall entropy and charge scaling

This self-contained Colab bundle produces a new trajectory-resolved S100 data
set for the pure-state domain-wall conformal-field-theory tests. It uses the
canonical `classA_U1FGTN_gpu.run_markov_circuit` entry point and computes
entropy and intrinsic quantum charge variance from the same restricted-frame
singular-value decomposition. It never constructs or saves a covariance
matrix.

## Locked scientific contract

- `Nx=20`; `Ny=30,35,40,45,50,55,60`
- 100 independent Born trajectories per circumference and exactly `2*Ny`
  physical cycles
- hard/support-terminated wall with `DW_loc=[5,15]`, `nshell=1`,
  `alpha_1=1`, and `alpha_2=30`
- random pure half-filled initialization, `raster_y` measurement order,
  perfect correction, and no postselection
- native occupied-frame evolution in `complex128` on an A100 40-GB-class GPU
- sampling revision
  `hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v1` with root seed
  `2026090305`
- full-x periodic strips `[0,Nx) x [y0,y0+Ay)` for every
  `Ay=0,...,Ny//2` and every `y0=0,...,Ny-1`

The origin average is performed inside each trajectory. The `Ny` origins are
not treated as independent samples; uncertainty is obtained only from the 100
independent trajectories.

Every size records integer global charge at cycles `0,...,2*Ny` and the
trajectory-level, origin-averaged entropy, mean strip charge, and intrinsic
quantum charge variance at the endpoint. `Ny=40` additionally records all
three strip curves at cycles `10,...,80`, the half-strip entropy and charge-
variance contours at cycles `0,...,80`, and all-width contours at the final
cycle. Cycle zero is the state after the canonical Born-conditioned exterior
preparation.

## Two independent lanes

The two notebooks share one configuration hash and write disjoint directories:

- `run_lane_A_Ny40_Ny60.ipynb`: `Ny=40,60`, eight execution batches and 40
  durable five-trajectory result shards. This lane contains the detailed
  `Ny=40` work and can launch the final analysis after both lanes finish.
- `run_lane_B_endpoint_Ny30_35_45_50_55.ipynb`: the five remaining endpoint
  sizes, 17 execution batches and 100 durable result shards.

They may run concurrently in separate A100 runtimes because all result and
checkpoint paths include `lane_A` or `lane_B`. Do not run two copies of the
same lane simultaneously.

The A100 execution-batch sizes are:

```text
Ny       30  35  40  45  50  55  60
samples  80  60  40  30  25  20  20
```

The detailed `Ny=40` observer uses sample and origin chunks of `8 x 8`; the
endpoint observer uses `4 x 4`. These are execution settings only and do not
change the estimator.

## Run and resume

Upload this entire folder to
`MyDrive/final_production_new_designs/05_hard_wall_entropy_charge`, open the
desired notebook in an A100 runtime, and run its cells from top to bottom. The
notebook copies executable files to `/content`; Drive is used only for rolling
checkpoints and completed products.

Set `REPORT_ONLY=True` to verify the inventory without starting GPU work.
`MAX_NEW_EXECUTION_BATCHES` may be set to a nonnegative integer to bound a
session; `None` continues until the selected lane is complete or the runtime
disconnects. The outer `tqdm` bar counts durable five-trajectory result shards
(`40` in lane A and `100` in lane B). The inner bar reports physical-cycle
progress for the active execution batch and resumes at its verified durable
cycle.

Dynamics is split into canonical-engine calls of five cycles. After each call,
the runner writes one stable `checkpoint.npz` plus `checkpoint.json` containing
the occupied frame and ranks, NumPy RNG state, Torch CPU and CUDA RNG states,
the completed physical cycle, and every partial observer accumulator. A
continuation passes `frame_init_prepared=True`, so the already completed hard-
wall exterior preparation is not repeated. The next segment begins only after
the checkpoint has passed DriveFS byte-count and SHA-256 readback.

The checkpoint is retained through final result publication. It is removed
only after every component five-trajectory NPZ and completion JSON has been
verified. Thus an interruption during computation loses at most five cycles;
an interruption during publication resumes from the final-cycle checkpoint.

## Durable layout

```text
results/lane_A/Ny040/shard_000_samples_000-004.npz
results/lane_A/Ny040/shard_000_samples_000-004.complete.json
checkpoints/lane_A/Ny040/execution_000_samples_000-039/checkpoint.npz
checkpoints/lane_A/Ny040/execution_000_samples_000-039/checkpoint.json
```

Each result completion JSON is written last and binds the lane, execution
batch, five global sample IDs, batch seed, shared configuration hash, source
hashes, canonical entry point, result filename, byte count, and SHA-256. A
restart skips a result only when the entire pair reverifies. Missing, partial,
or mismatched pairs remain pending.

The largest rolling frame checkpoint is approximately 1 GiB before filesystem
overhead. Detailed `Ny=40` scientific products are well below 0.1 GiB raw for
all 100 samples; the other sizes save compact endpoint curves. Two concurrent
lanes therefore remain modest relative to the available 100-GB Drive quota.
The planning estimate is roughly 100--130 total A100-hours, or about 2.5--3
calendar days when both lanes run continuously; the first completed execution
batch in each lane provides the better empirical forecast printed by the
runner.

## Analysis

After the combined report shows `140/140` verified result shards, set
`RUN_ANALYSIS=True` in the lane-A notebook. `analyze_campaign.py` fits the
natural-log chord coordinate

```text
X(Ay) = log[(Ny/pi) sin(pi Ay/Ny)]
```

over `Ay=8,...,Ny//2`, using an intercept. It reports `c=3*m_S`,
`k=pi^2*m_F`, and the paired residual `c-k`, with deterministic 20,000-fold
whole-trajectory paired bootstrap intervals. The contour analysis reports the
left and right three-column wall contributions and leakage outside those
windows. Classical trajectory-to-trajectory variation of mean strip charge is
reported separately and is never substituted for the intrinsic quantum
variance used to extract `k`.

There is deliberately no Drive API, remote-status subprocess, lease,
generation/pointer checkpoint protocol, migration ledger, archive container,
or session dashboard.
