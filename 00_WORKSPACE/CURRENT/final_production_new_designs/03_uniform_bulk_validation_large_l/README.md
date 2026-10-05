# Uniform bulk validation: S100 large-L add-on

> **Campaign closed on 2026-09-09.** The bulk-validation claim is accepted on
> the complete balanced `L=12,16,20,24,28,32` dataset (`S=100` for each of
> `n_shell=1,2,dense`). The unfinished `L=36,40` extension is retired and is no
> longer required. Existing partial `L=36` outputs remain preserved as
> non-headline supporting data; they must not be represented as a complete
> `S=100` size point. Do not resume this notebook for the current paper plan.
> See
> [`CAMPAIGN_CLOSURE.md`](../../experiment_review/uniform_bulk_validation_analysis/CAMPAIGN_CLOSURE.md).

This is the independent large-system extension of the completed uniform bulk
campaign. It records the legacy real-space Chern estimator and global charge at
every physical cycle, including the initial state. It does not alter, overwrite,
or pool with the completed `L=12,16,20,24` output collection.

## Locked scientific contract

- square sizes: `L = 28, 32, 36, 40`
- OW support: `nshell = 1, 2, None` (`None` is the dense construction)
- independent trajectories: `S = 100` per size/support pair
- initialization: random pure state at half filling
- geometry: `DW=False`, `alpha_1=alpha_2=1`
- dynamics: random serial order, perfect correction, no postselection
- duration: exactly `40` physical cycles for every system size
- arithmetic: `complex128` on an NVIDIA A100 with 40-GB-class memory
- entry point: `classA_U1FGTN_gpu.run_markov_circuit`

There are 12 cases, 1,200 trajectories, and 36 durable batch tasks. The locked
batch sizes are `100, 50, 30, 20` for `L = 28, 32, 36, 40`, respectively. These
sizes were extrapolated from the measured completed `L<=24` campaign, whose
per-trajectory runtime scaled approximately as `L^4.8`. The projected full-batch
runtimes are approximately 1.77, 1.69, 1.78, and 1.97 hours, respectively.

This approximately two-hour restart unit is an explicit campaign choice. It is
larger than the repository's usual one-hour checkpoint threshold, but was chosen
to favor fewer, larger GPU batches. There is deliberately no intra-batch state
checkpoint: a disconnect can repeat up to about two hours of the active batch,
while every previously completed batch remains checksum-resumable.

The same timing extrapolation gives approximately 63 A100-hours for the full
extension: about 5.3, 10.1, 17.8, and 29.6 hours across all three OW choices at
`L = 28, 32, 36, 40`. Treat this as a planning estimate; the two largest sizes
have not yet been timed in production.

## Run on Colab

The instructions below are retained for provenance only. The current campaign
decision is **do not resume**.

1. Upload this entire folder to
   `MyDrive/final_production_new_designs/03_uniform_bulk_validation_large_l`.
2. Open `run_uniform_bulk_validation_large_l.ipynb` in an A100 runtime.
3. Review the single configuration cell. Leave its production values unchanged.
4. Run the notebook from the top.

The runner writes to
`MyDrive/classA_final_production_outputs/uniform_perfect_correction_40cycle_s100_l28_l40_batched_v2`.
Rerunning the notebook verifies and skips completed batches. A disconnect can
lose only the batch currently in memory.

Set `REPORT_ONLY = True` in the configuration cell to inspect completion without
starting GPU work. `MAX_NEW_TASKS` may be set to a positive integer for a bounded
session; `None` runs until the campaign completes or Colab disconnects.

## Result format

Each batch produces one `batch_NNN_samples_AAA-BBB.npz` and one matching
`batch_NNN_samples_AAA-BBB.complete.json`. The NPZ contains only compact arrays;
trajectory-valued arrays have shape `(batch_size, 41)`:

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
covariance histories, state histories, Drive API calls, leases, dashboards,
migration ledgers, or archive containers.
