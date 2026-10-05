# Uniform bulk validation: S100 every-cycle campaign

This is the simple, independent rerun of the completed uniform-topological P1
campaign. It records the legacy real-space Chern estimator and global charge at
every physical cycle, including the initial state.

## Locked scientific contract

- square sizes: `L = 12, 16, 20, 24`
- OW support: `nshell = 1, 2, None` (`None` is the dense construction)
- independent trajectories: `S = 100` per size/support pair
- initialization: random pure state at half filling
- geometry: `DW=False`, `alpha_1=alpha_2=1`
- dynamics: random serial order, perfect correction, no postselection
- duration: exactly `40` physical cycles for every system size
- arithmetic: `complex128` on an NVIDIA A100 with 40-GB-class memory
- entry point: `classA_U1FGTN_gpu.run_markov_circuit`

There are 12 cases, 1,200 trajectories, and 24 durable batch tasks. The
locked batch sizes are `100, 100, 50, 25` for `L = 12, 16, 20, 24`,
respectively. Legacy A100 timings project roughly 16--38 minutes of dynamics
per batch before the added every-cycle observable cost. This size-dependent
exception to the usual five-trajectory shard limit avoids hundreds of tiny,
GPU-underfilled tasks while keeping the restart unit below about one hour.

The prior S25 campaign implies about 12.6 total A100-hours after scaling to 40
cycles and S100, before accounting for the denser observer evaluation. Budget
roughly 14--18 A100-hours including the every-cycle observable.

## Run on Colab

1. Upload this entire folder to
   `MyDrive/final_production_new_designs/01_uniform_bulk_validation_maxl24`.
2. Open `run_uniform_bulk_validation.ipynb` in an A100 runtime.
3. Review the single configuration cell. Leave its production values unchanged.
4. Run the notebook from the top.

The runner writes to
`MyDrive/classA_final_production_outputs/uniform_perfect_correction_40cycle_s100_maxl24_batched_v1`.
Rerunning the notebook verifies and skips completed batches. A disconnect can
lose only the batch currently in memory.

Set `REPORT_ONLY = True` in the configuration cell to inspect completion without
starting GPU work. `MAX_NEW_TASKS` may be set to a positive integer for a bounded
session; `None` runs until the campaign completes or Colab disconnects.

## Result format

Each batch produces one `batch_NNN_samples_AAA-BBB.npz` and one matching
`batch_NNN_samples_AAA-BBB.complete.json`. The NPZ contains only compact
arrays; trajectory-valued arrays have shape `(batch_size, 41)`:

- `cycles`: integer `0..40`
- `normalized_cycles`: `cycles/L`
- `real_space_chern`: the fixed central three-sector estimator with radius `0.4L`
- `global_charge`: occupied-frame charge
- `particle_number`: integer occupied rank
- `half_filling_offset`: `particle_number - L**2`

The completion JSON is written last and binds the result byte count and SHA-256,
the exact batch/configuration identity, global sample indices, batch seed,
canonical entry point, and source hashes. A failed batch is rerun from its
deterministic seed; earlier verified batches are not repeated. There are no
covariance histories, state histories, Drive API calls,
leases, dashboards, migration ledgers, or archive containers.
