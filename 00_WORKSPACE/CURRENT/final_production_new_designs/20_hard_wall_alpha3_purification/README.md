# Alpha=3 hard-wall purification control

Run `run_hard_wall_alpha3_purification.ipynb` on an A100 40-GB-class runtime.
This is a fresh **100-sample Born ensemble**, not bundle 19's deterministic
postselection trajectory. No existing campaign data are changed or imported.

## Locked science

Nx=20, Ny=40, T=160 (4Ny), alpha1=3, alpha2=30, nshell=1, X trial orbitals,
hard support truncation, walls x=5,15, raster-y, perfect correction,
postselect=False, n_a=0.5, complex128 covariance dynamics through canonical
`classA_U1FGTN_gpu.run_markov_circuit`. The active slab is maximally mixed;
the canonical Born-conditioned exterior preparation occurs once, before t=0.
There are 880 active physical modes. A prepared checkpoint never repeats the
exterior preparation. The explicit-covariance continuation tolerance 0.50000001
only controls representation classification, not dynamics or observable tolerances.

Revision: `hard_wall_alpha3_maxmix_nx20_ny40_s100_4ny_v1`; root seed 2026092403.
Deterministic execution units have independent seeds and fixed sample ranges
0--39, 40--79, 80--99. Do not change batch sizes during an ensemble.

## Data

At every cycle 0,...,160 and for every sample, save the complete physical-layer
occupation spectrum, total entropy (nats), cell-resolved entropy contour,
charge, charge variance and its contour, Born log-probability increments and
cumulative log probability, event counts and numerical closure diagnostics.
The observer is byte-identical to bundle 07, including its entropy roundoff
floor, to match the alpha1=1 comparison. It computes each trajectory's
observables before any averaging. Tiny residual entropy near projector
precision is a numerical floor, not a surviving mixed mode.

Save the final physical-layer centered covariance G=2C-I, complex128,
for every sample. Thus endpoint natural orbitals/eigenvectors can be computed
offline without an eigenvalue-retention filter. No covariance history is saved.
Born weights start after exterior preparation, with omega(0)=0 and
log Z=880 log(2)+omega. No many-body CFT fit is performed by the notebook.

## Execution and resume

One notebook, one runtime at a time. Mount Drive once and stage executable
files in `/content/20_hard_wall_alpha3_purification`. The runner executes inside
the notebook kernel so native tqdm bars and tracebacks are visible; no hidden
subprocess output. It prints the config, source identities, paths, inventory,
task start, checkpoint status, memory and measured remaining-time projection.

Use fixed resident batches 40+40+20; eigendecomposition is chunked in fives.
The memory budget is **38 decimal GB = 38,000,000,000 bytes**, not 38 GiB.
PyTorch's allocator is capped at at most 35 decimal GB, reduced if pre-existing
external allocations need more headroom. Device-wide usage is checked after
each cycle; measured allocated/reserved peaks are reported. The allocator
limit cannot police transient external CUDA-library allocations between checks.
A100 execution is still needed to measure the actual peak; no untested claim
of a hard driver-level cap is made. Failure stops and preserves the last
verified checkpoint; it never changes sampling/batch identities automatically.

Every ten cycles replace one covariance/RNG/observer checkpoint NPZ plus its
checksum-bound JSON. Completed execution batches publish immutable five-sample
shards: 20 result NPZ/completion pairs. Each result/checkpoint is created locally,
copied to a temporary DriveFS filename, reopened and SHA-256 checked, renamed,
and followed by its JSON. Remove the final checkpoint only after every shard
verifies. Interrupted shard publication resumes from the final checkpoint.
Ordinary computation interruptions repeat up to ten cycles. A disruption
between replacing the two checkpoint files may leave an invalid pair and
require deterministic replay of that batch; this simple two-file protocol is
not an atomic multi-file transaction. Keep old completed shards regardless.
DriveFS verification is not independent server-side verification of cloud sync.

`REPORT_ONLY=True` verifies inventory without GPU work.
`MAX_NEW_EXECUTION_BATCHES=1` limits a session to one pending execution batch;
the default `None` runs the three-batch queue. The final cell disconnects.
Changing any locked scientific/configuration field requires a new campaign.

Output: `MyDrive/classA_final_production_outputs/hard_wall_alpha3_maxmix_nx20_ny40_s100_4ny_v1`.
Expect about 4.5--5 GB permanent output, plus rolling checkpoint/temp overhead;
start with at least 12 GiB free on Drive and 6 GiB in local scratch. DriveFS free
space reporting is a best-effort mounted-filesystem check, not an account quota API.

## Timing and validation

Matching archived bundle-07 alpha1=1 Ny40 batches took 19027.17, 19031.97,
9563.00 seconds: **13.23 A100-hours** total. Budget **13--16 A100-hours**, excluding
disconnect downtime. Alpha1=3 does not reduce fixed-depth covariance workload.
The first completed batch prints a measured ETA; it is not a speedup guarantee.

Local tests cover protocol expansion, observer formulas, canonical-source sync,
checkpoint/RNG replay on a small validation system, publication faults and
recovery after a final checkpoint, notebook visibility and memory-limit wiring.
There is no local A100: GPU throughput and the full-size CUDA memory peak remain
production-runtime checks. This bundle is derived from bundle 07 with no legacy
migration code and no dependencies on any other campaign directory.
