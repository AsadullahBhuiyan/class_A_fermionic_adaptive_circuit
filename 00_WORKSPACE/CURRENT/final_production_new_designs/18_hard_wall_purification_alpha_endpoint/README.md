# Hard-wall purification alpha endpoint sweep

This standalone two-lane Colab bundle measures endpoint occupation-derived
finite-time single-particle Lyapunov gaps. It does **not** measure the
tangent-cocycle or Choi Lyapunov spectrum.

## Locked campaign

- `Nx=20`; `Ny=24,28,32,36,40,44,50`.
- 21 `alpha_1` values:
  `1,1.2,1.4,1.6,1.7,1.8,1.85,1.9,1.95,1.975,2,2.025,2.05,2.1,2.15,2.2,2.3,2.4,2.6,2.8,3`.
- 100 independent trajectories per `(Ny, alpha_1)` and `T=2Ny`.
- Hard/support-truncated walls at `x=5,15`; `alpha_2=30`, `nshell=1`,
  maximally mixed initialization, raster-y order, perfect correction, and no
  postselection.
- Canonical covariance GPU dynamics in `complex128` on an A100 40-GB-class
  runtime under a strict 38-GiB PyTorch allocator ceiling.

The revision is
`hard_wall_maxmix_purification_alpha21_ny24-50_s100_2ny_endpoint_lyapunov_v1`.
Outputs go to `MyDrive/classA_final_production_outputs/<revision>`.

## Run order

Open the lane A and lane B notebooks in two different A100 runtimes. The lane
assignment alternates size/alpha parity, so each complete `(Ny, alpha_1)`
configuration belongs to exactly one notebook. Lane A owns 74 configurations,
7,400 trajectories, and 116 execution batches. Lane B owns 73 configurations,
7,300 trajectories, and 115 execution batches.

Both notebooks initially set `MAX_NEW_EXECUTION_BATCHES=1`. Run that default
once in each lane. The first task is an `Ny=50` value nearest `alpha_1=2`, so it
is the intended worst-case timing, storage, and memory qualification. Accept
the full launch only when each result verifies and the reported peak stays
below 38 GiB. Then set `MAX_NEW_EXECUTION_BATCHES=None` and rerun the launch
cell. Set `REPORT_ONLY=True` at any time to inspect progress without starting a
new batch.

The durable execution batches are 100 trajectories for `Ny=24,28,32`, `90+10`
for 36, `75+25` for 40, `60+40` for 44, and `50+50` for 50. They are processed
internally in CUDA microbatches of at most `100,100,64,50,40,30,25`,
respectively. This leaves working memory for the canonical rank-one resolvent
without changing task IDs, seeds, checkpoints, or result shards. A native outer
`tqdm` bar tracks durable five-trajectory shards and an inner bar tracks cycles
in the current execution batch. A measured remaining-time projection prints
after each completed batch.

## Resume and durability

The engine runs from `/content`. Every ten cycles, and at `T`, the runner
publishes one rolling covariance/RNG checkpoint through a Drive temporary file,
byte/SHA-256 readback, and atomic rename. A restart resumes exactly from the
last valid boundary. The final checkpoint remains present while the endpoint
eigensystem is computed and while every five-trajectory result shard is
published and verified. It is removed only after all shards from that execution
batch pass readback.

Results are immutable NPZ/completion-JSON pairs. A completion JSON is written
last and binds the configuration, executed source hashes, deterministic seed,
sample identities, byte count, and SHA-256. No Drive API, lease, archive,
dashboard, migration, intermediate spectrum, record probability, covariance
history, or final covariance archive is used.

## Endpoint convention

The active hard-wall slab contains `N_eff=22Ny` modes. At `T=2Ny`, the runner
Hermitizes and diagonalizes its centered covariance, obtaining `a_j` in
`[-1,1]`, then evaluates

`lambda_j = [log(1-a_j)-log(1+a_j)]/(2T) = -atanh(a_j)/T`.

Values within `1e-9` of `+1` or `-1` are exact caps with infinite exponent
magnitude. The `0.1` eigenvector retention margin never participates in cap or
gap classification. Every trajectory saves the legacy half-gap, a two-sided
gap when both signs exist, the centered half-gap, the 16 modes closest to zero,
cap counts, finite-mode count, and numerical residuals.

For `Ny=50`, every result also saves the complete sorted centered spectrum and
all eigenvectors with `abs(a)<=0.9`, together with full-spectrum indices,
sample offsets, rates, residuals, and active-basis indices. Each retained
vector is phase-fixed by making its largest-magnitude component real and
nonnegative. Degenerate-subspace rotations remain non-unique; localization
analysis must aggregate the corresponding subspace.

Expected runtime is roughly 300–500 aggregate A100 hours, split nearly evenly.
Permanent storage should usually be a few gigabytes, with a conservative
40–45 GB worst case if most `Ny=50` modes pass the eigenvector filter.
