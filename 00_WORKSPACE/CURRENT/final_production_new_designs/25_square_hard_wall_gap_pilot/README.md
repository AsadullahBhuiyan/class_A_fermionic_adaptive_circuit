# Square-system fixed-time gap pilot

One self-contained A100 40-GB-class Colab notebook, seven geometries, 70
independent Born trajectories, and 14 five-trajectory endpoint result shards.
This is a new ensemble; no existing campaign is overwritten or resumed.

## Scientific contract

- `Nx=Ny=L=20,24,28,32,36,40,44`; ten samples and **ten physical cycles** each.
- Hard/support-truncated walls at `L/4,3L/4`, `nshell=1`, `alpha_1=1`,
  `alpha_2=30`, trial orbital `X`, perfect correction, `raster_y`, complex128.
- Original gap protocol: `meas_slab_only=True`, `triv_region_local_mode=False`,
  maximally mixed initialization with canonical Born-conditioned exterior
  preparation. This is **not** campaign 21's full-measurement protocol.
- Canonical entry point: `classA_U1FGTN_gpu.run_markov_circuit`.
- Revision `square_hard_wall_gap_l20-44_s10_t10_v1`, root seed `2026092725`.
  One deterministic seed per resident ten-sample batch; the batch composition
  is locked and changing it changes the Born ensemble.

| L | Wall positions | Active single-particle modes |
|---:|:---:|---:|
| 20 | 5, 15 | 440 |
| 24 | 6, 18 | 624 |
| 28 | 7, 21 | 840 |
| 32 | 8, 24 | 1088 |
| 36 | 9, 27 | 1368 |
| 40 | 10, 30 | 1680 |
| 44 | 11, 33 | 2024 |

## What is saved and what the gap means

Only the cycle-ten **active-slab** occupation spectrum is observed. For each
trajectory, the occupations are eigenvalues of the restricted correlation
matrix `(I+G_centered)/2`; they are not many-body occupation probabilities.

\[
\epsilon_j=\log\frac{1-\nu_j}{\nu_j},\qquad
g_{\rm mod}=\min_j|\epsilon_j|,\qquad
\Delta=\frac{g_{\rm mod}}{2T}=\frac{g_{\rm mod}}{20}.
\]

Here `Delta` denotes the occupation-derived Lyapunov **half gap**, not the
correlation-matrix gap around one half. Pure endpoint caps within `1e-9` of
zero or one map to signed infinite modular energies, not artificially finite
floor values. Raw occupations, capped occupations, cap masks, modular energies,
signed finite-time rates, both gap definitions, diagnostics, sample IDs,
geometry, and configuration/source identity are retained. Out-of-bounds
occupations beyond tolerance stop the run; covariance clipping is disabled.

The analysis first takes each trajectory's minimum absolute modular energy,
then reports its mean and ordinary `SD/sqrt(10)` SEM. It does **not** take the
minimum after averaging spectra. Sizes with any nonfinite sample gap are
flagged rather than silently dropping those samples. No bootstrap or automatic
power-law fit is used. CSV tables, JSON provenance, and separate raw-gap and
normalized-gap PDF/PNG figures are produced after all 14 shards verify.

At fixed `T=10`, the denominator is independent of size. This tests whether a
size trend survives without the earlier `T proportional to Ny` normalization.
It does not establish a converged infinite-time Lyapunov exponent. Increasing
both width and circumference also changes wall separation, so this is not a
fixed-width circumference sweep. A physical cycle remains a full spatial sweep,
not a single local measurement.

No intermediate spectra, contours, eigenvectors, record weights, or covariance
histories are saved. Full covariances exist only in rolling resume checkpoints.

## Running in Colab

1. Put this complete folder (including `src`) at
   `MyDrive/final_production_new_designs/25_square_hard_wall_gap_pilot/`.
   Open `run_square_hard_wall_gap.ipynb` on an A100 40-GB-class runtime.
2. Mount Drive and run the configuration cell. `REPORT_ONLY=True` inventories
   existing results/checkpoints without simulation. The complete scientific
   configuration is visible but locked; scientific changes need a new revision.
3. Set `REPORT_ONLY=False` and run the staging/runner cell. Optionally set
   `MAX_NEW_EXECUTION_BATCHES=1` for the first `L=44` case. The largest geometry
   runs first; one resident batch holds all ten trajectories. The allocator
   ceiling is 35 decimal GB. Timings and peak reserved memory are printed.
4. Relaunch the same notebook to resume. Never run two writers on this output.
5. After 14/14 verified shards, enable `RUN_ANALYSIS`. The last cell releases
   the Colab runtime.

Code and scratch are staged under `/content`. The separate output collection is
`MyDrive/classA_final_production_outputs/square_hard_wall_gap_l20-44_s10_t10_v1`.
The notebook streams text through Jupyter-compatible stdout, with outer shard,
physical-cycle, and endpoint sample progress bars.

## Resume and storage

Each case runs two five-cycle canonical-engine segments. The rolling checkpoint
contains the complete centered covariance, completed cycle, NumPy/Torch CPU/all
CUDA RNG states, sample IDs, and configuration/source identity. Continuation
sets `G_init_prepared=True`, skips repeated exterior preparation, and restores
RNG immediately before the next canonical call.

At cycle ten, the checkpoint is verified **before** endpoint diagonalization.
Endpoint extraction handles one sample at a time. If interrupted there, resume
skips dynamics and repeats only endpoint extraction. The two immutable
five-sample result/completion pairs are verified before checkpoint deletion.
An interruption between those publications retains the first verified shard.
Incomplete or mismatched pairs are not counted as completed.

Publication uses local scratch, a temporary DriveFS copy, size/SHA-256 readback,
atomic replacement, and completion JSON last. It does not claim independent
Drive-server verification; a broken mount fails visibly. No Drive API or
transaction framework is included. A partial checkpoint pair is rejected and
the case reruns deterministically from zero.

The largest uncompressed ten-sample covariance is about 2.40 GB; replacing it
temporarily requires roughly twice that space on Drive. Completed endpoint
spectra are small (tens of MB for the campaign). Allow at least 8 GiB free in
both scratch and the mounted output filesystem. Local scratch copies remain
until the Colab runtime is released.

## Validation status

The maintained test is `tests/test_square_hard_wall_gap_pilot.py`. It covers the
contract, legacy spectral parity, exact uninterrupted versus checkpoint-resumed
small CPU-backed GPU-class runs, RNG preservation, corrupt/partial pairs, failed
readback, endpoint/publication/cleanup interruption recovery, notebook structure,
and byte identity of the bundled canonical GPU sources. These small tests do
not constitute a production A100 timing or memory benchmark. No CUDA device was
available during implementation; production completion is recorded below.

## Completed production data — downloaded 2026-09-28

All **14/14 result shards** were downloaded from `abhuiyan2398@gmail.com`:
70 trajectories, samples 0–9 exactly once at each of the seven sizes, all at
cycle 10. The original result NPZs and completion JSONs are retained unchanged
under `gpu_data/square_hard_wall_gap_l20-44_s10_t10_v1/results/`.

`gpu_data/square_hard_wall_gap_l20-44_s10_t10_v1/DOWNLOAD_VERIFICATION.json`
records Drive file IDs, byte counts, SHA-256 hashes, task/source/configuration
checks, spectral reconstruction checks, and per-size timings. All 70 sample gaps
are finite. The 28 result/receipt files total **713,447 bytes** (approximately
0.71 MB compressed), substantially below the conservative pre-run estimate.
The Drive checkpoint directories were empty at verification; no Drive files
were changed or deleted during import.

The local-only `import_drive_results.py` handles authenticated connector download
references and verifies the saved campaign contract without rerunning dynamics.
Raw modular gaps and rates divided by `2T=20` are both available for analysis.
