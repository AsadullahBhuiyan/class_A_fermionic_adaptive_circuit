# Batched endpoint soft-wall entropy and charge v2

This self-contained bundle is the soft/untruncated counterpart of the active
hard-wall endpoint campaign. It deliberately uses a separate sampling revision,
root seed, Drive output tree, result identities, and checkpoints. Nothing in
this bundle reads, pools, or resumes hard-wall artifacts.

## Locked contract

- `Nx=20`; `Ny=30,35,40,45,50,55,60`; 100 trajectories per size
- soft/untruncated interface at `x=5,15`, `nshell=1`, `alpha_1=1`,
  `alpha_2=30`
- `DW=True`, `dw_truncation=False`, `meas_slab_only=False`; overcomplete
  Wannier (OW) modes may cross the interface and every top-layer cell is updated
- pure initialization, `raster_y`, perfect correction, complex128
- `2*Ny` physical cycles through the canonical GPU engine
- root seed `2026090305`, intentionally matching bundle 05 so corresponding
  hard/soft batches start from the same deterministic initialization stream;
  the subsequent Born records are not assumed to remain paired
- revision
  `soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint`

The only every-cycle scientific products are integer global particle number
and its offset from half filling. At the endpoint, the runner evaluates every
periodic origin and every `Ay=0,...,Ny//2`, retaining each trajectory's origin
average for von Neumann entropy, Rényi-2, Rényi-3, strip charge expectation,
and intrinsic quantum charge variance. It also saves four spatial contours
for the single fixed region `y0=0, Ay=Ny//2`. It never stores covariance or
frame histories and never computes time-dependent entropy curves or contours.

At fixed `Ay`, independent `(trajectory,y0)` restricted correlation matrices
are stacked and sent to `torch.linalg.eigvalsh`. The fixed half-strip uses one
batched `torch.linalg.eigh`; its eigenvectors are reused for all four contours.
All scalar quantities use the same occupations. Rényi-2 and Rényi-3 therefore
add only elementwise arithmetic, not more eigendecompositions.

## Mandatory A100 benchmark

Each lane first creates a nonproduction, production-shaped Ny=60 complex128
prepared frame. It checks Gram eigenvalues against the prior singular-value
estimator, checks all entropy/charge scalars and contour closure, benchmarks
candidate matrix batches `16,32,64,80,128`, and retains only candidates with at
least 8 GiB projected GPU headroom. It then times the complete five-trajectory
endpoint calculation. Production is locked unless the projected time for 20
trajectories is at most four hours. The accepted JSON binds the GPU, config,
and source hashes and can be reused by that lane.

## Lanes, progress, and resume

- Lane A: Ny=60 then 40; 8 execution batches and 40 durable result shards.
- Lane B: Ny=55,50,45,35,30; 17 batches and 100 durable result shards.

The execution-batch sizes remain `80,60,40,30,25,20,20` for Ny
`30,35,40,45,50,55,60`. Each batch publishes immutable five-trajectory NPZ
and completion-JSON pairs, so final acceptance is 140 verified shards.

Three visible progress levels are provided: durable five-sample shards,
physical cycles, and endpoint `Ay`. Dynamics checkpoints every five cycles.
The stable uncompressed checkpoint contains the occupied frame, ranks,
NumPy/Torch CPU/all-CUDA RNG states, charge history, and exact config/source
identity. The soft protocol has no exterior projection step; continuation uses
the saved frame directly and the engine is required to report no exterior
preparation on both initial and resumed segments.

At the final cycle, that checkpoint becomes the verified final-frame source.
After every completed `Ay`, `endpoint_progress.npz/json` publishes all partial
endpoint arrays and binds the final-frame checksum. A restart therefore:

1. resumes dynamics from its last five-cycle boundary;
2. or, when dynamics finished, resumes at the first missing `Ay`;
3. or skips any five-sample result whose completion pair re-verifies.

The final frame and endpoint progress are removed only after all component
results pass DriveFS byte-count and SHA-256 readback. The two lanes write
disjoint paths. Never launch two copies of the same lane simultaneously.

## Running

Upload this directory unchanged to
`MyDrive/final_production_new_designs/10_soft_wall_entropy_charge_batched_v2`.
Open either notebook on an A100 40-GB runtime and run top to bottom. It stages
the bundle under `/content` and streams unbuffered child output. Runtime knobs
are `REPORT_ONLY`, `BENCHMARK_ONLY`, and `MAX_NEW_EXECUTION_BATCHES`.

Outputs go to:

`MyDrive/classA_final_production_outputs/soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint`

After both report-only passes total 140/140, enable lane A's analysis cell. It
fits the locked `Ay=8,...,Ny//2` chord window, reports
`c1=3m1`, `c2=4m2`, `c3=9m3/2`, and `k=pi^2 mF`. Error bars are the
sample-wise standard error `SD/sqrt(100)` after fitting each trajectory.
Outputs include endpoint prefactors,
paired `cq-k`, representative curves, fixed-half-strip contours and wall
fractions, and global-charge wandering. No time-dependent central-charge or
contour-evolution products are produced.

The v2 configuration identity still contains the original unused
`bootstrap_replicates` and `bootstrap_seed` metadata fields. They are retained
only for configuration compatibility; the analysis does not bootstrap.

There is deliberately no Drive API, lease, dashboard, archive, migration,
generation, or pointer protocol.
