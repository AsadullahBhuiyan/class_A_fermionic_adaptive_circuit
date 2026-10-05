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
- Revision/output collection: full_measurement_gap_nx20_ny30-60_s100_t40_cycles6-40_v1.
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
pairs. Sizes run descending. Spectral extraction handles one sample per GPU
eigh and publishes five samples at a time; mode arrays never accumulate across
cycles in memory.

A single rolling uncompressed covariance/RNG checkpoint is published after every
physical cycle, BEFORE diagonalization. The runner then verifies every spectral
shard for that cycle before evolving again. On interruption it restores the
current state/RNG, skips verified spectral shards, finishes missing ones, and
continues. Final checkpoint cleanup occurs only after all 35 times verify.
If an earlier result is later missing/corrupt, exact data at that earlier time
cannot be recovered from a later state: the affected execution batch restarts
deterministically from zero, retaining valid result pairs. No backward evolution.

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
Actual output depends strongly on purification and compression; no measured
production storage or A100 runtime is claimed. Full spectra add roughly 1–2 GB;
the largest rolling covariance is about 2.30 GB plus temporary copies.
Every-cycle eigensolving and Drive writes will cost substantially more than
endpoint-only acquisition. Your 5-TB allocation can accommodate the worst-case
mode budget if sufficient space is free.

Local tests cover full-measurement/no-exterior-preparation dynamics,
uninterrupted vs one-cycle checkpoint continuation with exact state/RNG parity,
full-basis eigenvectors and the 1/(2t) convention, partial spectral publication,
spectral failure recovery, missing prior-cycle replay, checksum/readback failure,
workload/sample coverage, notebook generation and canonical source byte identity.
CPU-backed GPU-class tests are not an A100 performance benchmark.

