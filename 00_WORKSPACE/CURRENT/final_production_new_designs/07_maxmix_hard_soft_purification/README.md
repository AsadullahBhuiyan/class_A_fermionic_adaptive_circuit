# Max-mix hard/soft purification dynamics

This self-contained Colab bundle runs a new paired purification campaign through
the canonical `classA_U1FGTN_gpu.run_markov_circuit` entry point.  It saves the
complete occupation spectrum, absolute realized Born-record log probability,
and spatial entropy/charge-fluctuation contours at every cycle, plus the final
complex128 centered covariance for every trajectory.

## Locked contract

- `Nx=20`, `Ny=20,30,40`, 100 independent trajectories per construction
- hard/support-truncated and soft/untruncated domain walls at `x=[5,15]`
- maximally mixed initialization, `nshell=1`, `alpha_1=1`, `alpha_2=30`
- perfect correction, `raster_y`, `n_a=0.5`, complex128
- cycles `t=0,...,4*Ny`
- revision `maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3`
- root seed `2026090407`

The hard notebook uses `dw_truncation=True, meas_slab_only=True`; cycle zero is
after the canonical Born-conditioned exterior preparation.  The soft notebook
uses `dw_truncation=False, meas_slab_only=False`; cycle zero is the global
maximally mixed state.

For each physical cycle, the observer sums the float64 logarithms of all
realized conditional Born probabilities and stores both the cycle increment and

```text
omega(t) = log P(realized record from cycle 1 through t),  omega(0)=0.
```

For the hard construction the one-time exterior preparation probability is not
included: the transfer problem starts from the prepared active slab with
`N_eff=22*Ny`.  For the soft construction `N_eff=40*Ny`.  The absolute
many-body normalization is reconstructed without exponentiating tiny
probabilities,

```text
log Z(t) = N_eff*log(2) + omega(t).
```

For `C=(G+I)/2`, one complex128 Hermitian eigendecomposition supplies the raw
occupation eigenvalues and the cell-resolved contours

```text
s_r = sum_{orbital,a} |U_(r,orbital),a|^2 h(nu_a)
f_r = sum_{orbital,a} |U_(r,orbital),a|^2 nu_a (1-nu_a)
```

where `h(nu)=-nu log(nu)-(1-nu)log(1-nu)` is measured in nats.  The sums of
the two contours close to the total entropy and intrinsic quantum charge
variance.  `G_final` uses the canonical centered convention `G=2C-I`.

## Run and resume

Upload this folder to
`MyDrive/final_production_new_designs/07_maxmix_hard_soft_purification_v3` and run
either or both notebooks in separate A100 40-GB runtimes:

- `run_hard_wall_purification.ipynb`
- `run_soft_wall_purification.ipynb`

The two output trees are disjoint.  Do not run two copies of the same notebook
simultaneously.  Both notebooks copy executable files to `/content`, stream the
runner's nested `tqdm` bars, and disconnect only after the runner exits.

Execution batches are 100 samples at `Ny=20`, 100 at `Ny=30`, and `40+40+20`
at `Ny=40`.  Each batch is split into immutable five-sample result shards.  A
stable `checkpoint.npz` plus `checkpoint.json` is replaced after every ten
cycles and contains the covariance, NumPy/Torch CPU/all-CUDA RNG states, the
complete purification-observer prefix, and the completed record-weight prefix.
A hard-wall continuation uses
`G_init_prepared=True`, so exterior measurements are not repeated.  A restart
loses at most the active ten-cycle segment.

Set `REPORT_ONLY=True` to inspect the verified inventory without launching GPU
work.  Set `MAX_NEW_EXECUTION_BATCHES` to a nonnegative integer to bound a
notebook session; `None` runs until completion or disconnection.

Finished outputs live under
`MyDrive/classA_final_production_outputs/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3`.
Every NPZ has a completion JSON written last and bound to its byte count,
SHA-256, task identity, configuration hash, and source hashes.  Completion is
credited only after DriveFS readback succeeds.  The checkpoint is removed only
after all five-sample outputs from its execution batch reverify.

## Why v3 exists and how v2 resumes

V2 could checkpoint a large explicit covariance batch in which some
trajectories had already purified to projector precision while others remained
slightly mixed.  On the next segment, the canonical engine's automatic
pure/mixed representation detector rejected that heterogeneous batch before
cycle 71 even though the runner had explicitly requested covariance dynamics.
V3 uses the existing public engine contract without changing the canonical
engine.  Continuations still request `state_representation='covariance'`
explicitly and set the purity-classification tolerance to `0.50000001`.  For a
valid fermionic covariance every occupation lies in `[0,1]`, so the detector's
distance to the nearest projector eigenvalue is at most `0.5` (up to the
existing numerical slack).  The batch is therefore classified uniformly for
representation-selection purposes while it remains in the explicitly requested
covariance representation.  This tolerance is not used by the covariance
dynamics, Born sampling, or observer.  The dynamics, measurements, seeds,
observer, and complex128 contract are unchanged.

Both v3 notebooks expose `IMPORT_VERIFIED_V2_CHECKPOINTS=True`.  When a v3
checkpoint is absent, the runner looks read-only in the v2 output tree.  It
imports a v2 checkpoint only if its configuration hash, all four executed
source hashes, task/seed/sample identity, schema, byte count, SHA-256, covariance
shape/dtype, RNG payload, and observer prefix all match the frozen v2 contract.
It then writes and rereads a distinct v3 checkpoint before continuing.  The v2
checkpoint is never changed or deleted.  The known soft `Ny=20`, samples
`000--099` checkpoint therefore resumes from durable cycle 70 and starts with
cycle 71.

The superseded v1 and v2 output folders remain untouched.  The exact executed
v2 repository bundle is preserved under
`00_WORKSPACE/LEGACY/final_production_new_designs_quarantine_2026-09-07/`.

Expected finished storage is about 17--20 GB.  The initial estimate is 45--70
total A100-hours; the two construction notebooks can run concurrently.

## Completed analysis

The complete hard-v2 and corrected soft-v3 result trees are stored under this
bundle's `gpu_data/` directory.  Analyze and reverify all 600 trajectories with

```bash
python analyze_completed_campaign.py
```

The reader-facing two-column report, its machine-readable acceptance ledger,
CSV tables, and figure source assets are written to
`analysis_outputs/hard_v2_soft_v3_4ny_analysis_v1/`.  The analysis keeps the two
historical execution identities separate, reconstructs 64 leading
`log(sigma^2)` levels without enumerating Fock space, fits trajectories before
ensemble averaging, and bootstraps whole trajectories only.

## Hard-wall quenched log-polar notebook

`polar_effective_hamiltonian_analysis.ipynb` is a separate local CPU analysis
of the 300 completed hard-v2 endpoint covariances.  It uses only the decoupled
active slab `x=5,...,15`, whose cycle-zero state is maximally mixed after the
recorded exterior preparation.  Every source is also required to have an
exactly pure exterior with zero inter-`y` correlations and zero
active--exterior coupling; this makes active-slab strip entropy identical to
the full-system result.  For every trajectory it constructs the
endpoint-identifiable left polar generator

```text
A_xi = log(P_L,xi) = arctanh(G_xi),   h_xi = -2*A_xi,
```

before applying the `y` twirl or any sample average.  The notebook treats the
quenched mean and typical trajectory Hamiltonians on equal footing, constructs
their half-filled ground states, and compares their Renyi strip entropies with
the actual endpoint states and the accepted exact-B0 Cardy--Calabrese
calibration.  Annealed-covariance and flattened-parent results are labeled
controls.  The spectrum products retain both the endpoint generator `h` and
the per-cycle rate `h/(4*Ny)`, while `velocity_summary.csv` records localized
crossings, wall weights, velocities, and whole-trajectory intervals.

The `4*Ny` endpoints are already nearly pure, so the log-polar spectrum is
regularized at `A=+/-10` and repeated at caps `8` and `12`.  The claim ledger
releases an edge-spectrum statement only if the wall localization, opposite
velocity signs, and crossings are stable across all three caps.  Otherwise it
reports endpoint non-identifiability rather than interpreting numerical
saturation as missing edge physics.

Run the notebook from top to bottom.  Its first executable cell exposes CPU
affinity, worker count, sizes, sample limit, hash verification, caps, twists,
and output revision.  One checksum-bound cache is written per trajectory under
`analysis_outputs/hard_quenched_log_polar_v1/sample_cache`, so an interruption
repeats only the current trajectory.  Set `MAX_SAMPLES_PER_NY=1` for a smoke
run; the default `None` processes all 100 samples at each size.

## Soft-wall quenched log-polar notebook

`soft_polar_effective_hamiltonian_analysis.ipynb` applies the same estimator
order and acceptance ledger to the 300 completed soft-v3 trajectories.  It is
a separate, co-primary analysis under
`analysis_outputs/soft_quenched_log_polar_v1`; soft and hard trajectories are
never pooled.  The soft transfer problem begins from the global maximally
mixed state and retains the full `x=0,...,19` system, so its momentum blocks
are `40 x 40`, half filling contains `20*Ny` modes, and both three-column wall
windows `(4,5,6)` and `(14,15,16)` are represented without support
truncation.  This makes the soft result especially useful for distinguishing
physical wall modes from artifacts of the hard support boundary.

The notebook verifies the distinct v3 result/completion schemas and source
identity, reproduces the accepted exact coupled-domain-wall B0 calibration,
and writes independent checksum-bound per-trajectory caches.  It exposes the
same endpoint/per-cycle spectra, localized velocities, Renyi entropies,
quenched jackknife, trajectory bootstrap, cap family, twists, fit windows,
annealed control, and flattened control as the hard analysis.  Use
`MAX_SAMPLES_PER_NY=1` for a soft smoke run before committing the full CPU
campaign.
