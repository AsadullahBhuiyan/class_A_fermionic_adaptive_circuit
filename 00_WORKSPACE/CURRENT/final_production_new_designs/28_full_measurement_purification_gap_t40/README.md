# Full-measurement purification, fixed T=40

Independent campaign 28. Do not resume or pool slab-only campaigns 13, 25 or 26.
The full-measurement alpha1=1 campaign 21 is a protocol reference, not an input.

## Locked science

- Nx=20, Ny=30,36,42,48,54,60; 100 independent Born trajectories per size.
- Forty physical cycles, hard/support-truncated walls x=5,15, nshell=1.
- alpha1=1, alpha2=30, trial orbital X, raster_y, perfect correction.
- Global maximally mixed initial state; meas_slab_only=False. No initial
  Born-conditioned exterior preparation. The trivial slabs are measured too.
- complex128 covariance dynamics through the canonical GPU run_markov_circuit.
- Independent root seed 2026092828; seed/task identity includes locked batch bounds.
- Revision/output collection: full_measurement_gap_nx20_ny30-60_s100_t40_cycles6-40_v2_batched.
- No covariance clipping, postselection, contour computation or legacy imports.

## Products at every cycle 6–40

“After five” means strictly after cycle five: 35 observation times, including T=40.
Each five-sample/cycle NPZ stores:

- full-system occupation spectra (raw and tolerance-capped), cap masks;
- signed modular energies epsilon=log((1-nu)/nu) and rates epsilon/(2t);
- sample-wise raw modular half-gap min(abs(epsilon)) and Lyapunov half-gap
  min(abs(epsilon))/(2t), finite-gap flags and numerical bound diagnostics;
- all finite-mode eigenvectors in complex128, counts and spectrum indices;
- full spatial basis map, sample IDs, physical cycle, configuration and sources.

Finite modes mean 1e-9 < nu < 1-1e-9. This retains every mixed mode, including
zero-rate modes; no additional “slow” cutoff is imposed. Pure capped modes have
infinite modular energies and are not saved as eigenvectors. Raw occupations
remain available. This cap is an analysis convention, NOT a change to dynamics.
Occupation excess beyond 1e-9 stops visibly and preserves the state checkpoint.

The engine stores centered covariance G=2C-I. Its eigenvectors are also the
eigenvectors of C and the modular Hamiltonian. Full basis ordering is
mu + 2*x + 2*Nx*y; the NPZ coordinate map explicitly resolves x,y,orbital.
Each sample retains its own spectrum and eigenvectors, not those of an averaged
state. Vector phases and bases within degenerate eigenspaces are arbitrary:
compare projectors/densities, not phase-sensitive components across cycles.
The gap is a half-gap convention, not twice the nearest distance from zero.

## Batching and restart

| Ny | 30 | 36 | 42 | 48 | 54 | 60 |
|---|---:|---:|---:|---:|---:|---:|
| Resident trajectories | 100 | 50 | 50 | 25 | 25 | 20 |

18 execution batches, 600 trajectories, 4,200 immutable five-sample/cycle result
pairs. Sizes run descending. Spectral extraction benchmarks matrix batches 1,2,5 on the actual GPU at
each size and selects the fastest retaining 8 GiB headroom. It uses a genuine
stacked Hermitian eigh call, with an OOM fallback to a smaller microbatch.
This tunes observation only: dynamics batches and seeds stay unchanged.
Five samples publish together; mode arrays never accumulate across cycles.

A single rolling uncompressed covariance/RNG checkpoint is published every five
cycles (and at the endpoint). Spectral shards still publish every cycle 6–40.
On interruption the runner restores the last checkpoint, replays at most five
dynamics cycles and skips already verified spectral shards. Observations do not
consume simulation RNG. Final checkpoint cleanup follows verification of all
cycle products. If an older result is corrupt/missing, replay starts from zero.

Version 2 keeps all requested scientific products and the resident trajectory
batch table unchanged. It replaces single-matrix observation with measured
microbatch selection, uses lossless ZIP level-one compression, and avoids
redundant file hashing. Five-cycle rather than per-cycle state publication cuts
checkpoint write volume from about 1.31 TB to 262 GB for the full campaign.
The v1 deployment is preserved under PROJECT_ADMIN/deployment_snapshots/.
Use the new v2 output directory; no previous results are silently adopted.

Only local /content scratch and DriveFS temporary-copy/readback/SHA256 publication
are used. Completion JSON comes last. This is mounted-filesystem verification,
not a guarantee of server-side Drive synchronization. Keep sufficient free Drive
space and do not run concurrent writers. Existing unrelated outputs are untouched.

Checkpoint replacement follows the repository's simple stable NPZ/JSON design.
An interruption between their replacements can leave an invalid pair; it is
rejected on restart, with deterministic replay rather than unsafe continuation.

## Run

Upload this entire folder, including src, under MyDrive/final_production_new_designs.
Open run_full_measurement_purification_gap_t40.ipynb in an A100 40-GB-class runtime.
Mount once, inspect the complete configuration cell, then run the staged runner.
REPORT_ONLY=True verifies inventory without running dynamics.
MAX_NEW_EXECUTION_BATCHES=1 measures the first Ny=60 batch; None runs all pending
batches. Restart the same notebook to continue. Streamed cycle, sample and durable
shard progress is Jupyter-compatible. The final cell releases the runtime.

RUN_ANALYSIS=True requires all 4,200 pairs. It writes per-sample gaps and
cycle/size means with ordinary trajectory SEM, separate raw/normalized plots,
and a provenance manifest. It does not infer a scaling exponent automatically.

## Cost and validation

Worst-case uncompressed eigenvectors: 100*35*sum_Ny((40*Ny)^2)*16 =
1,145,088,000,000 bytes (~1.15 TB). This assumes every mode remains mixed.
Campaign 21's measured Ny30 mixed-mode counts suggest ~3.6 GB uncompressed
vectors over cycles 6–40 at that size, and ~50–100 GB across this sweep if
counts scale similarly. This is an extrapolation, not a storage guarantee. Full spectra add roughly 1–2 GB;
the largest rolling covariance is about 2.30 GB plus temporary copies.
Timing reference: campaign 21's S100 Ny30 full-measurement run took
13,598.31 seconds for 60 cycles, including observers but excluding checkpoints.
A cubic-in-Ny extrapolation to six sizes at 40 cycles gives ~58.9 A100-hours.
Allow roughly 60–90 hours including spectral computation and Drive writes;
this range is a planning estimate, not a benchmark or guaranteed completion.
The first actual batch prints stage timings and a measured remaining-batch ETA.
Every-cycle eigensolving and Drive writes cost more than endpoint-only acquisition. Your 5-TB allocation can accommodate the worst-case
mode budget if sufficient space is free.

Local tests cover full-measurement/no-exterior-preparation dynamics,
uninterrupted vs one-cycle checkpoint continuation with exact state/RNG parity,
full-basis eigenvectors and the 1/(2t) convention, partial spectral publication,
spectral failure recovery, missing prior-cycle replay, checksum/readback failure,
workload/sample coverage, notebook generation and canonical source byte identity.
CPU-backed GPU-class tests are not an A100 performance benchmark.
