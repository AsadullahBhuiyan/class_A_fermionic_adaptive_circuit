# A100 wall-diabatized spectral pump: primary campaign

This standalone Colab bundle ports the expensive dense complex128 eigensolver
steps of the frozen-record wall-diabatized spectral pump to an NVIDIA A100. It
consumes existing endpoint states and writes the same primary result arrays and
completion schema as the CPU reference. It never runs the separate 2,050-task
sensitivity suite.

Production is aggressively concurrent. The mandatory A100 benchmark measures
one, two, four, and eight simultaneous endpoint lanes on the largest 1536-mode
matrix, rejects choices that leave less than 8 GiB of aggregate device-memory
headroom, and locks the highest-throughput accepted lane count into its receipt.
Each lane owns one endpoint continuation, so GPU eigensolvers overlap with the
other lanes' CPU overlap selection, SVDs, validation, and result preparation.

## Files

- `run_wall_diabatic_spectral_pump_gpu.ipynb`: Colab operator interface;
- `run_campaign.py`: input inventory, parity gate, resume, and publication;
- `gpu_backend.py`: Torch/CUDA complex128 eigensolver substitutions;
- `spectral_cpu_reference.py`: unchanged scientific continuation, validation,
  source-contract, and result-schema implementation;
- `campaign_config.json`: unchanged locked primary scientific configuration;
- `build_notebook.py`: notebook generator; and
- this README.

This is a spectral diagnostic of already generated endpoint states. It does not
simulate the monitored circuit, so it does not call a dynamics engine.

## Required Drive inputs

The full primary campaign has two source collections:

1. The output of bundle 11, normally at
   `MyDrive/classA_final_production_outputs/wall_pump_width_endpoints_s100_v1`.
   It contains 1,000 newly generated primary endpoints and 150 optional bridge
   endpoints in five-sample shards.
2. A copy of the local `frozen_record_flux_charge_pilot` directory containing
   these three existing per-sample burn-in trees:
   - `results/N20x24_state_projector_pump_s100_v1/burnins`;
   - `results/N24x24_state_projector_pump_s100_v1/burnins`; and
   - `results/N20x24_state_projector_pump_s100_dense_v1/burnins`.

The notebook defaults the second root to
`MyDrive/frozen_record_flux_charge_pilot`. Change the path if the repository or
pilot was uploaded under a different Drive directory. Report mode verifies all
hashes and prints the first missing paths without starting GPU work.

## Scientific and numerical contract

The primary calculation remains M=256, uniform twist, signed `1e-7` regulator,
rank-two wall diabatization, separate CW/CCW paths, forward continuation, and
true reverse undo. All result arrays and the numerical gates are inherited from
the CPU reference. There are 1,600 primary endpoint tasks; enabling the
independent backend bridge adds 150 tasks.

Before production, the runner requires a 40-GB-class A100 and compares three
representative complex128 GPU eigensystems with a CPU reference. Maximum
eigenvalue and stable-subspace-projector errors must both be at most `1e-10`.
The gate compares the isolated two-mode edge-cluster projector at the nearly
degenerate flux crossing and the occupied projector away from it. The
receipt records GPU timing, peak allocation, identities, and an explicitly
labelled eigensolver-only lower-bound projection. A changed bundle invalidates
the receipt.

The GPU backend intentionally retains the mature within-endpoint continuation
and schema code. It accelerates full Hermitian eigendecompositions and endpoint
defect diagonalizations while running several endpoints concurrently. The first
complete concurrent batch therefore serves as the authoritative end-to-end
throughput measurement. If an endpoint exceeds one hour, stop: this
task-boundary-only version does not provide a safe sub-task rolling checkpoint
and should not be used for the full campaign. The bounded launch deliberately
selects the largest pending `Nx=32` tasks, records per-task and batch wall times,
and projects the 1,600-task runtime. Unlimited mode is blocked until that
receipt exists and every task in it finished within one hour.

## Resume and Drive publication

One endpoint pair is the deterministic unit of work. A restart verifies and
skips every result/completion pair. The current endpoint repeats after an
interruption; completed endpoints do not.

Each result is first written and validated under `/content`. It is then copied
to a temporary DriveFS path, reopened to verify byte count and SHA-256, renamed
to its stable name, and followed by its completion JSON. Input endpoint files
are read-only. Output goes by default to
`MyDrive/classA_final_production_outputs/wall_diabatic_spectral_pump_gpu_primary_v1`,
so CPU campaign outputs are never overwritten or silently merged.

## Colab sequence

Upload this directory unchanged to
`MyDrive/final_production_new_designs/12_wall_diabatic_spectral_pump_gpu`, open
the notebook on an A100 40-GB runtime, and follow its four-step sequence:

1. `REPORT_ONLY=True` for complete endpoint inventory;
2. `BENCHMARK_ONLY=True` for the mandatory parity/timing receipt;
3. `MAX_NEW_BATCHES=1` for one complete concurrent-batch timing; and
4. `MAX_NEW_BATCHES=None` only after accepting that timing.

The last notebook cell disconnects the Colab runtime and prints `done`.
