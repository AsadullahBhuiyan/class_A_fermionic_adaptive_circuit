# Wall-pump width endpoint acquisition

This standalone A100 bundle supplies the missing endpoint-state ensembles for
the width-controlled wall-pump analysis. It does not thread flux or analyze a
pump. Existing CPU endpoints and all older output folders remain unchanged.

Status on September 7, 2026: the first Colab launch stopped while constructing
the dense benchmark model, before benchmark dynamics began. It produced no
benchmark receipt, endpoint shard, or scientific data. The operational bug
was a fixed `backend="local"` argument applied to `nshell=None`; the corrected
runner maps `nshell=1` to the canonical local backend and `nshell=None` to the
canonical dense backend. Production has not been launched successfully.

## Bundle contents

- `run_wall_pump_width_endpoints.ipynb`: the drop-in Colab interface;
- `campaign_config.json`: the locked editable scientific and benchmark grid;
- `run_campaign.py`: benchmark, resume, checkpoint, and endpoint publisher;
- `src/classA_U1FGTN_gpu.py` and `src/occupied_frame_gpu.py`: byte-identical
  copies of the canonical GPU engine and occupied-frame helper;
- `build_notebook.py`: the canonical notebook generator; and
- this README: the operator and durability contract.

## Locked campaign

- `Ny=24`, `Nx=20,24,28,32`, walls at `Nx/4` and `3Nx/4`
- soft wall: `dw_truncation=False`, `meas_slab_only=False`
- hard wall: `dw_truncation=True`, `meas_slab_only=True`
- `nshell=1` with engine backend `local`, and dense (`nshell=None`) with engine
  backend `dense`
- pure half-filled initialization, `alpha_1=1`, `alpha_2=30`
- 48 raster-y cycles, perfect correction, no postselection, complex128
- canonical `classA_U1FGTN_gpu.run_markov_circuit`
- root seed `2026090701`
- revision `wall_pump_width_endpoints_s100_v1`

The primary S100 cells are `nsh1/Nx=28,32` and
`dense/Nx=24,28,32`, for both walls: 1,000 endpoints. The bridge uses S25 for
`nsh1/Nx=20,24` and `dense/Nx=20`, also for both walls: 150 endpoints. These
independent bridge trajectories are backend cross-check data; they are not
silently pooled with legacy CPU samples.

## Benchmark gate

Before production, the runner measures 5-cycle throughput and reserved memory
for execution batches `10,20,40,60,80,100` on the worst-case `Nx=32`
dense/soft arm. It chooses the fastest candidate retaining at least 8 GiB of
A100 headroom. It then runs one full 48-cycle, five-trajectory nonproduction
timing for each of the ten primary protocol/width/wall arms. Production starts
only if their conservative S100 projection sums to at most 40 hours, half the
recorded 80-hour 56-core CPU forecast. That forecast comes from the summed
per-trajectory timings in the completed `N20x24` shell-one, `N20x24` dense,
and `N24x24` shell-one S100 endpoint campaigns, extrapolating the measured
shell-one and dense worker times as `Nx^4` to the five missing cells and then
dividing by 56. The unrounded estimate is about 80.5 hours, so the locked
40-hour cutoff is slightly stricter than the requested twofold speedup. The
accepted receipt fixes the backend and execution-batch map for all restarts.

Production also requires at least 25 GiB free in the output filesystem after
the benchmark gate. Report-only and benchmark-only runs do not apply this
production-space gate.

## Results and resume

Every durable product contains exactly five trajectories. Primary paths are:

`endpoints/{nsh1|dense}/N{Nx}x24/{soft|hard}/shard_XX.npz`

Bridge paths have the same suffix under `bridge/`. Each NPZ contains sample
IDs, the padded final occupied `frames` and `ranks`, every-cycle global charge,
half-filling offsets, endpoint x-density and regional charges, Gram residuals,
configuration/source identities, and execution metadata. A matching
`.completion.json` is written last and binds the result byte count and SHA-256.

Execution batches may be larger than five for GPU efficiency, but their
boundaries and RNG seeds are fixed by the accepted benchmark. A rolling
`checkpoint.npz`/`checkpoint.json` pair is published every five cycles and
contains the native frame/ranks, NumPy and Torch CPU/CUDA RNG states, and charge
observer state. A restart loses at most five cycles. The checkpoint is removed
only after all of its five-sample results verify.

Publication uses the simple repository policy: compute under `/content`, copy
to a temporary DriveFS path, reopen and verify byte count/SHA-256, atomically
rename, then write the completion JSON. There is no Drive API, lease, pointer,
dashboard, remote-status subprocess, or migration layer.

## Colab

Upload this directory unchanged to
`MyDrive/final_production_new_designs/11_wall_pump_width_endpoints`, open
`run_wall_pump_width_endpoints.ipynb` in a 40-GB A100 runtime, and run top to
bottom. Outputs go to
`MyDrive/classA_final_production_outputs/wall_pump_width_endpoints_s100_v1`.

The configuration cell exposes `REPORT_ONLY`, `BENCHMARK_ONLY`, and
`MAX_NEW_EXECUTION_BATCHES`. The notebook stages executable code under
`/content`, streams both tqdm levels, and disconnects only after the runner
returns successfully.

Use this order in a fresh A100 runtime:

1. Leave `BENCHMARK_ONLY=True`, run the notebook top to bottom, and retain the
   printed accepted/rejected benchmark result.  A rejection is a valid result
   and means to use the documented CPU fallback instead of endpoint
   production.
2. After an accepted benchmark, set `BENCHMARK_ONLY=False` and
   `REPORT_ONLY=True`; rerun the configuration and launch cells to inspect the
   exact verified/pending shard inventory without starting dynamics.
3. Set `REPORT_ONLY=False`.  Optionally limit the first production test with
   `MAX_NEW_EXECUTION_BATCHES=1`; remove that limit only after the resulting
   five-sample shards and checkpoint behavior are visible on Drive.
4. Rerun the same notebook after any disconnect.  The accepted benchmark fixes
   the execution-batch map, verified five-sample results are skipped, and a
   matching rolling checkpoint resumes the interrupted execution batch.

The complete matrix contains 230 durable endpoint shards: 200 primary shards
for 1,000 endpoints and 30 bridge shards for 150 endpoints.  The selected GPU
execution batch may contain 10--100 trajectories, but durable results are
always split into five-trajectory products.  The notebook requires at least
25 GiB free after the benchmark; expected endpoint storage is about 13--16
GiB.

## Failure and recovery semantics

- Before the first five-cycle checkpoint, an interruption repeats only the
  current execution batch.
- After a verified checkpoint, restart restores the exact occupied frames,
  ranks, observer prefix, NumPy RNG state, Torch CPU RNG state, and all CUDA
  RNG states.  At most five completed cycles are lost.
- A result is durable only after its DriveFS temporary copy has been reopened,
  byte-counted, checksummed, and atomically renamed.  Its completion JSON is
  published last.  An incomplete or mismatched pair is pending and reruns.
- A checkpoint is removed only after every five-trajectory shard belonging to
  its execution batch verifies.  A failed final write therefore preserves the
  checkpoint.
- The bundle has no Drive API, remote file IDs, leases, dashboards, migration
  ledger, or background Drive log.  If Drive is disconnected during a write,
  the write fails visibly and the same notebook is rerun after remounting.

Downstream ingestion is strict: copy the completed output tree without
renaming directories into
`frozen_record_flux_charge_pilot/imported_endpoints/wall_pump_width_endpoints_s100_v1`.
The spectral runner rechecks shard/completion hashes, source identity, GPU
backend, complex128 dtype, wall flags, sample IDs, and configuration before it
uses a frame.

The exact NPZ and JSON keys, shapes, axes, schemas, and downstream analysis
products are defined in
`frozen_record_flux_charge_pilot/WALL_DIABATIC_DATA_DICTIONARY.md` in the
repository.
