# Hard-wall Chern dynamics: ten random centers

Self-contained A100 40-GB-class Colab bundle. No other bundle is imported.
This is a new ensemble, not an extension or import of an older campaign.

## Run

Upload this entire folder to `MyDrive/final_production_new_designs/`, open
`run_hard_wall_random_center_chern.ipynb`, select an A100, and run in order.
All settings and paths are visible in the configuration cell. The scientific
contract is fixed for this revision; session controls are `REPORT_ONLY` and
`MAX_NEW_BATCHES`. Drive upload and execution are not performed by building this bundle.
Code is staged under `/content`; only completed products go to Drive.
Use one runtime/writer at a time for this output folder.

## Science and products

Nx=20, Ny=20/30/40; S=100 per geometry; cycles=40/60/80.
Alpha1=1, alpha2=30, nshell=1, hard support truncation, periodic boundaries,
interfaces x=5,15, slab-only raster-y measurements, perfect correction, no
postselection, pure random half-filled initialization, complex128. The canonical
engine performs Born-conditioned exterior preparation before observed cycle zero.
This is NOT the newer full-layer measurement/purification protocol.

Each sample and cycle draws ten distinct integer y centers without replacement
using a separate deterministic RNG keyed by root seed, geometry, sample ID and
cycle. x0=10 and R=4. Minimum-image y coordinates wrap the disk at y=0. Both
orbitals share sector membership. Three counterclockwise sectors use angular
intervals [0,2pi/3), [2pi/3,4pi/3), [4pi/3,2pi). With Gamma=(V V†)^T,

    C_G = Re{12 pi i [tr(Gamma_CA Gamma_AB Gamma_BC)
                      - tr(Gamma_AC Gamma_CB Gamma_BA)]}.

The frame implementation gathers sector rows and contracts with leading GPU
dimensions (trajectory, center); no full covariance is constructed. Optional
center chunking only bounds temporary memory, not the RNG or estimator.

One NPZ and completion JSON per execution batch. NPZ fields:

- `sample_ids`, `cycles`; `centers_y`, `real_space_chern`: (B,T+1,10).
- `center_average`, `global_charge`: (B,T+1), with integer charge = frame rank.
- `final_frame`: complex128 (B,2NxNy,max_rank), zero-padded;
  `final_ranks`: integer (B,).
- `metadata_json`: geometry/protocol/config, sample IDs, batch seed, source hashes,
  config hash and canonical entry point.

There are 300 trajectories and 183,000 individual Chern values. Final frames
take roughly 3.7 decimal GB before compression/padding; no frame/covariance
history is saved. Statistics first average the ten centers within each trajectory,
then take the trajectory mean and SEM (ddof=1). Centers are NOT independent samples.

## Calibration and resume

No batch size is prequalified. Each geometry tests B=25,50,100 with the actual
observer, one warm-up cycle and five timed cycles. Test B=10,5 if none qualify.
Select highest measured samples/cycle-second among candidates with peak reserved
memory <32 GiB and full-trajectory forecast (including setup and a 25% margin)
<45 minutes. Calibration uses separate seeds and never supplies production data.
The forecast excludes compression/Drive transfer and is an estimate, not a bound.
All candidates and measured timing/memory are saved in `execution_plan.json`.
The selected sizes, task boundaries and batch seeds are frozen before production.
Batch-dependent circuit RNG streams are never silently changed on restart.

Restart the same notebook/config to verify and skip completed batches. Missing,
partial or corrupt result/receipt pairs rerun with the original batch seed.
Scientific/source identity mismatch stops instead of mixing campaigns.
Each result is created locally, copied to a temporary DriveFS path, reopened and
checksummed, atomically replaced, then followed by its completion JSON.
**DriveFS readback is not independent server/cloud verification.** Keep Drive
space available and allow synchronization before disconnecting.

The approved exception to the usual small-shard policy is calibrated execution
batches with completion-only resume: interruption loses only the in-flight batch.
If a batch exceeds one hour, it is saved and the queue stops, including on future
restarts. Do not hand-edit the frozen plan to shrink pending tasks: ask for an
explicit revised execution plan preserving all completed results/seeds first.
This bundle does not implement automatic repartitioning or state checkpoints.

Root seed: 2026092701. Output collection:
`classA_final_production_outputs/hard_wall_random_center_chern_nx20_ny20-30-40_s100_r0p2_v1`.

## Rebuild and test (repository)

Run `python build_notebook.py --sync-sources` here to regenerate the notebook,
configuration and manifest and copy the two canonical GPU sources byte-for-byte.
Run `pytest tests/test_hard_wall_random_center_chern.py` from the repository root.
Local dynamics tests use the canonical CPU engine; CUDA timing/memory qualification
must be performed on the actual A100 by the notebook. No A100 qualification is
claimed by the CPU tests.

Local validation on 2026-09-27: 29 tests passed, covering all production geometry
partitions, explicit-covariance parity, padding, center/RNG invariance, canonical
CPU dynamics and interrupted rerun, calibration selection/fallback, queue resume,
publication failure paths, source synchronization and notebook generation.
Actual A100 timing/memory calibration remains pending. No production data have
been generated and no Drive files have been uploaded by this implementation.
