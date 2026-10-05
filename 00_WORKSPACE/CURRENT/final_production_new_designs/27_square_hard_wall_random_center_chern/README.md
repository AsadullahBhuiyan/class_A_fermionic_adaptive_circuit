# Square hard-wall Chern dynamics, 40 cycles

Self-contained A100 40-GB-class notebook, adapted from the completed bundle 23.
It does not import bundle 23 or reuse its samples. Old code and results are unchanged.

| Nx=Ny=L | Trajectories | Saved cycles | Radius R=0.2L | Canonical interfaces |
| --- | ---: | --- | ---: | --- |
| 20 | 100 | 0–40 | 4 | 5,15 |
| 30 | 100 | 0–40 | 6 | 8,22 |
| 40 | 100 | 0–40 | 8 | 10,30 |

## Run in Colab

Upload this folder to `MyDrive/final_production_new_designs/`, open
`run_square_hard_wall_random_center_chern.ipynb`, select an A100, and run in order.
The configuration cell exposes the complete contract and paths. `REPORT_ONLY`
prints the resume inventory without launching work; `MAX_NEW_BATCHES` optionally
limits a session. Only one runtime should write to the output folder.
Code and scratch work live under `/content`. Both levels of tqdm output are
relayed into the notebook. The final cell disconnects the runtime.

## Scientific contract

Alpha1=1, alpha2=30, nshell=1; hard support truncation; slab-only raster-y
measurements; periodic boundaries; perfect correction; no postselection;
pure random half-filled initialization; complex128. The canonical GPU
`classA_U1FGTN_gpu.run_markov_circuit` performs Born-conditioned exterior
preparation before cycle zero. This is not the full-layer purification protocol.

At every cycle including zero, each trajectory independently draws ten distinct
integer y centers uniformly without replacement. A separate deterministic RNG
is keyed by root seed, geometry, sample ID and cycle. x0=L/2; R=0.2L. Periodic
minimum-image coordinates keep disks intact across y=0. Both orbitals of a cell
belong to the same counterclockwise sector. The unchanged frame estimator uses
Gamma=(V V†)^T and

    C_G = Re{12 pi i [tr(Gamma_CA Gamma_AB Gamma_BC)
                     - tr(Gamma_AC Gamma_CB Gamma_BA)]}.

GPU contractions batch trajectories and centers without a full covariance.
`center_chunk_size` can reduce temporary memory; fix it before the first run.
Centers are averaged within each trajectory before forming ensemble mean and
SEM (ddof=1 over 100 trajectories), never treated as independent samples.

The canonical integer slab boundaries use L//2 ± L//4. Thus L=30 has walls
at 8 and 22, not noninteger coordinates 7.5 and 22.5. All disks are strictly
contained. This sweep changes slab width, Ny and disk radius together; it is
not an isolated Nx or fixed-R test. Exactly 40 cycles are used for every size,
not 2Ny. Whether 40 cycles suffices at larger width is a result to check.

## Calibration and simple resume

No batch sizes are prequalified. For each size, benchmark ascending batches
5,10,25,50,100 with the real observer: one warm-up plus five timed cycles.
Stop increasing the batch at a measured memory/time failure; if none qualify,
try 2 and 1. Choose the highest measured trajectory throughput among tested
candidates with peak reserved GPU memory <32 GiB and a full-40-cycle forecast
including setup and 25% margin <45 minutes. Separate calibration seeds never
contribute production samples. Compression and Drive transfer are not included
in this forecast, so it is an estimate, not a hard wall-time guarantee.

The saved `execution_plan.json` freezes selected batches, task boundaries and
seeds before production. Resuming verifies NPZ/JSON identities, bytes and hashes
and skips completed batches. An interrupted batch reruns from its beginning
with its original seed. Larger batches follow the user's existing calibrated
batch-resume design; no additional checkpoint protocol is introduced.
If a completed batch exceeds one hour, save it and stop the queue, including
on restart, before more long tasks run. Any re-batching must preserve completed
results and explicitly change the execution contract, not edit the frozen plan.

Results are written locally, copied to a temporary DriveFS path, reopened and
checksummed, atomically renamed, then followed by a completion JSON.
DriveFS readback is not independent cloud-server verification. Keep adequate
Drive space and allow synchronization before disconnecting. At least 12 GiB
free is required on scratch and mounted Drive before starting work.

## Products and independent identity

One compressed NPZ plus completion JSON per execution batch stores:

- samples, cycles 0–40, and all ten center coordinates and Chern values
  `(batch,41,10)`;
- per-trajectory center means and integer global charge `(batch,41)`;
- final complex128 occupied frames and ranks (no state/covariance histories);
- geometry, fixed scientific settings, batch seed and source/configuration hashes.

Total: 300 trajectories and 123,000 scalar Chern evaluations. Half-filled
endpoint frames alone total approximately 11.30 decimal GB before compression
and rank padding. Actual compressed storage and run time are measured in Colab.
Root seed: `2026092801`. Independent output collection:
`classA_final_production_outputs/square_hard_wall_random_center_chern_l20-30-40_s100_t40_r0p2_v1`.
The L=20 ensemble is new, not silently pooled with the old L=20 data.

## Rebuild and tests

Run `python build_notebook.py --sync-sources` in this directory. This copies the
canonical engine/helper byte-for-byte and regenerates notebook/config/manifest.
From the repo root run `pytest tests/test_square_hard_wall_random_center_chern.py`.
CPU tests do not establish actual A100 timing/memory qualification.
Building the bundle does not upload to Drive or launch production.

Local validation (2026-09-28): 34 new-bundle tests plus 29 original-bundle
regressions passed together. Coverage includes all three disk radii and seam
wrapping, explicit-covariance parity, heterogeneous ranks, observer RNG
isolation, physical-projector equivalence on deterministic CPU reruns,
40-cycle calibration forecasts and fallback, completion resume and failed
readbacks, canonical source synchronization, and executable notebook cells.
Actual A100 calibration remains pending.
