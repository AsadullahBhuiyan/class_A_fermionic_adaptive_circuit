# Wall-diabatized width-sweep data dictionary

This document defines the durable files consumed and produced by
`N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1`. It is a field-level
companion to `WALL_DIABATIC_WIDTH_SWEEP_RUNBOOK.md` and the two-column methods
note. Array axes are never inferred from filenames.

Notation used below:

- `S=5`: trajectories in one durable endpoint shard;
- `B`: trajectories in one benchmark-selected execution batch;
- `N=2*Nx*Ny`: physical one-particle dimension;
- `r` or `r_max`: occupied rank and padded maximum rank;
- `C=48`: endpoint-preparation cycles, giving `C+1=49` observations;
- `D=2`: direction axis ordered exactly as `directions=["ccw","cw"]`, with
  `sigma=[+1,-1]`;
- `M`: flux intervals (`256` primary), giving `M+1` observations; and
- `k`: crossing-block rank (`2` primary, `4` sensitivity).

## 1. A100 endpoint files

### Result path and schema

Primary shards use
`endpoints/{nsh1|dense}/N{Nx}x24/{soft|hard}/shard_XX.npz`; bridge shards use
the same suffix below `bridge/`. The schema scalar is
`wall_pump_width_endpoint_shard_v1`.

Identity scalars and strings:

| Key | Meaning |
|---|---|
| `schema`, `bundle`, `sampling_revision` | Product schema, bundle ID, and endpoint sampling revision |
| `canonical_entry_point`, `execution_backend` | Canonical GPU dynamics call and `gpu` backend label |
| `stage`, `collection`, `cell`, `protocol`, `wall` | Task coordinates; `collection` is `endpoints` or `bridge` |
| `Nx`, `Ny`, `wall_locations` | Geometry and the two walls at `Nx/4,3Nx/4` |
| `nshell`, `nshell_label` | `1,nsh1` or `-1,dense`; `-1` is the serialized sentinel for `None` |
| `alpha_1`, `alpha_2`, `cycles_total` | Locked controller values `1,30,48` |
| `shard_index`, `sample_ids` | Durable shard number and its five global sample IDs |
| `execution_batch_id`, `execution_batch_seed` | Larger GPU execution batch that produced the shard |
| `selected_execution_batch_size` | Accepted candidate from `10,20,40,60,80,100` |
| `base_config_sha256`, `resolved_execution_sha256` | Scientific config and config-plus-execution-map identities |
| `source_hashes_json`, `metadata_json` | Canonical JSON copies of source and full task identity |

Scientific arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `cycles` | `(49,)` | Cycle numbers `0,...,48` |
| `frames` | `(S,N,r_max)` complex128 | Padded final occupied frames; only columns below `ranks[s]` are physical |
| `ranks` | `(S,)` | Final occupied rank per trajectory |
| `frame_capacity` | scalar | Stored padded column count `r_max` |
| `global_charge` | `(S,49)` | Integer occupied rank at every cycle, including the initial state |
| `half_filling_offset` | `(S,49)` | `global_charge-Nx*Ny` |
| `initial_total_charge`, `final_total_charge`, `net_injected_charge` | `(S,)` | Endpoint charge accounting |
| `minimum_rank`, `maximum_rank` | `(S,)` | Minimum and maximum charge reached over the 49 observations |
| `density_x` | `(S,Nx)` | Final density summed over `y` and orbital |
| `N_left`, `N_right` | `(S,)` | Final half-system regional charges |
| `charge_partition_residual` | `(S,)` | `N_left+N_right-final_total_charge` |
| `gram_residual` | `(S,)` | Final occupied-frame orthonormality residual |
| `elapsed_execution_batch_seconds` | scalar | Wall time for the containing execution batch |
| `benchmark_projected_missing_1000_seconds` | scalar | Accepted full missing-matrix projection |

### Completion JSON

The sibling schema is `wall_pump_width_endpoint_shard_completion_v1`. It
repeats the exact task, geometry, wall, sample IDs, execution-batch map and
seed, canonical entry point, backend, base/resolved configuration hashes, and
source hashes. `result_filename`, `result_bytes`, and `result_sha256` duplicate
the nested `result={name,bytes,sha256}` record. `completed_utc` and
`elapsed_execution_batch_seconds` are operational metadata. The result is not
complete unless every repeated identity matches the resolved task and the
actual final-path bytes hash to the recorded SHA-256.

### Rolling checkpoint

Each execution batch owns
`checkpoints/<execution-batch-id>/checkpoint.npz` and `checkpoint.json`.
The NPZ schema payload contains:

- scalar `completed_cycle` and `elapsed_seconds`;
- `frame` with shape `(B,N,r_max)` and `ranks` with shape `(B,)`;
- the exact `sample_ids`;
- `seen_cycles` with shape `(49,)` and `global_charge` with shape `(B,49)`;
- per-sample `gram_residual`; and
- RNG fields prefixed by `rng__`: NumPy algorithm, key array, position,
  Gaussian-cache flag/value, Torch CPU state, CUDA-device count, and one Torch
  state per CUDA device.

The JSON schema is `wall_pump_width_endpoint_checkpoint_v1`. It binds the
checkpoint to the full batch identity, accepted execution map, configuration
and source hashes, completed cycle, elapsed time, NPZ filename, byte count,
SHA-256, and update time. Only cycles divisible by five (or the final cycle)
are restorable.

### A100 benchmark receipt

`benchmarks/a100_endpoint_benchmark.json` has schema
`wall_pump_width_endpoint_a100_benchmark_v1`. Its immutable identity records
the source/config hashes, GPU name, six candidate batch sizes, five-cycle
short test, 8-GiB headroom gate, ten production-shaped full timing arms,
80-hour local reference, and 40-hour A100 cutoff. Data fields include the full
candidate rows, selected batch, resolved execution map/hash, ten 48-cycle
timing rows, projected seconds/hours for the missing 1,000 endpoints, status,
and completion time. A rejected receipt is evidence for the CPU fallback; it
does not authorize A100 production.

## 2. CPU-fallback endpoint files

The fallback writes the same `wall_pump_width_endpoint_shard_v1` NPZ interface
under a separate import root, so the spectral loader can consume it without a
format conversion. Its completion identifies backend `canonical_cpu`, the
canonical CPU entry point and source hashes, and the CPU-fallback configuration
hash. It contains only the 1,000 primary endpoints; no `bridge/` tree is
created. Its rolling checkpoint contains the native CPU occupied frame, rank,
observer prefix, and exact NumPy state for each sample task, aggregated into a
verified five-sample shard only after all five members complete.

## 3. Wall-diabatized spectral pair

### Path, identity, and axes

Primary paths are
`pump/{protocol}/N{Nx}x24/{wall}/sample_NNN.npz`. Bridge paths begin with
`bridge_pump/`; exact controls with `controls/<variant>/`; and sensitivities
with `sensitivity/<variant>/`. All use schema
`wall_diabatic_spectral_pump_result_v1` and a sibling completion with schema
`wall_diabatic_spectral_pump_completion_v1`.

`metadata_json` embeds the complete task: stage, task ID, cell, protocol,
geometry, wall, sample, mesh, block rank, wall window, gauge, primary/bridge
status, source backend and collection, control kind, scientific configuration
hash, runner source hashes, and the exact endpoint dependency. The dependency
contains result/completion paths, optional shard member index, result bytes and
SHA-256, completion SHA-256, source configuration hash, and source completion
schema. The completion repeats that metadata, adds wall time, and binds the
result `{name,bytes,sha256}`.

### Directional histories

Every key below has leading direction axis `D=2`.

| Keys | Shape | Meaning |
|---|---:|---|
| `phi` | `(D,M+1)` | Signed regulated flux mesh |
| `N_left`, `N_right`, `N_total` | `(D,M+1)` | Continued-projector regional and total charge |
| `delta_N_left`, `delta_N_right`, `delta_N_total`, `q_x` | `(D,M+1)` | Change from the source projector and raw `q_x=(Delta N_R-Delta N_L)/2` |
| `density_x` | `(D,M+1,Nx)` | Continued-projector x-resolved density |
| `ordinary_delta_N_left`, `ordinary_delta_N_right`, `ordinary_delta_N_total`, `ordinary_q_x` | `(D,M+1)` | Previous-global-overlap control |
| `instantaneous_delta_N_left`, `instantaneous_delta_N_right`, `instantaneous_delta_N_total`, `instantaneous_q_x` | `(D,M+1)` | Instantaneous-refill closure control |
| `principal_overlap`, `selected_weight_floor` | `(D,M+1)` | Main continuation matching diagnostics |
| `ordinary_principal_overlap`, `ordinary_selected_weight_floor` | `(D,M+1)` | Ordinary-overlap diagnostics |
| `edge_internal_gap`, `edge_external_gap` | `(D,M+1)` | Doublet splitting and doublet-to-complement isolation |
| `edge_B_eigenvalues`, `edge_combined_wall_weight` | `(D,M+1,k)` | Wall-polarization spectrum and wall-window weight inside the edge block |
| `edge_link_min_singular` | `(D,M+1)` | Neighboring edge-cluster polar-link quality |
| `edge_active_mask`, `edge_point_valid` | `(D,M+1)` | Applied diabatization interval and pointwise qualification |
| `total_charge_residual`, `projector_residual` | `(D,M+1)` | Numerical charge and orthonormality/idempotency diagnostics |

The direction labels are stored explicitly as `directions` and `sigma`; code
must not assume a reversed order. `q_x` is always raw and directional. The
analysis additionally computes `sigma*q_x` only for correctly oriented event
fractions; it never replaces the saved CW or CCW values.

### Source, crossing, and endpoint diagnostics

| Keys | Shape | Meaning |
|---|---:|---|
| `source_N_left`, `source_N_right` | scalar | Initial regional charges |
| `source_density_x` | `(Nx,)` | Initial x-resolved density |
| `source_real_space_chern_by_y0` | `(Ny,)` | Legacy three-wedge Chern estimator at every transverse origin |
| `source_real_space_chern_mean`, `source_real_space_chern_std`, `source_real_space_chern_xref`, `source_real_space_chern_radius` | scalar | Chern summary and estimator geometry |
| `edge_minimum_gap_index`, `edge_active_start_index`, `edge_active_end_index` | `(D,)` | Crossing and active interval indices; `-1` denotes no qualified active interval |
| `edge_entering_wall_label` | `(D,)` | Backward-compatible first occupied edge label |
| `edge_entering_wall_labels` | `(D,k/2)` | All occupied wall labels for rank-2/rank-4 blocks |
| `resolved`, `unresolved_reason` | `(D,)` | Direction-specific qualification result and fixed diagnostic reason |
| `endpoint_defect_eigenvalues` | `(D,N)` | Full spectrum of the gauge-unwrapped endpoint defect `D=P_end-P_0` |
| `endpoint_particle_density_x`, `endpoint_hole_density_x` | `(D,Nx)` | Positive/negative spectral-weight density of the endpoint defect |
| `endpoint_leading_particle_mode_density_x`, `endpoint_leading_hole_mode_density_x` | `(D,Nx)` | Leading positive/negative defect eigenmode densities |
| `endpoint_leading_particle_wall_weights`, `endpoint_leading_hole_wall_weights` | `(D,2)` | Radius-two weights on the two walls |
| `endpoint_leading_positive_eigenvalue`, `endpoint_leading_negative_eigenvalue` | `(D,)` | Leading particle/hole defect eigenvalues |
| `endpoint_positive_defect_count_above_0p9`, `endpoint_negative_defect_count_below_minus_0p9` | `(D,)` | Count of near-unit positive/negative defect modes |
| `endpoint_maximum_remaining_abs_defect_eigenvalue` | `(D,)` | Largest defect magnitude after removing the leading pair |
| `multicut_positions`, `multicut_q_x` | `(Nx-1,)`, `(D,Nx-1)` | Cut locations and endpoint charge readout across every x cut |
| `center_of_charge_displacement` | `(D,)` | Endpoint defect first moment along x |
| `large_gauge_parent_error`, `continuation_undo_error` | `(D,)` | Parent closure and true forward/reverse undo errors |
| `input_frame_gram_residual`, `input_projector_residual`, `rank` | scalar | Input numerical identity |

Boolean endpoint summaries are `endpoint_near_unit_event`,
`endpoint_single_defect_pair`,
`endpoint_defect_modes_opposite_wall_localized`, and
`endpoint_near_unit_defect_validation_pass`, each with shape `(D,)`. A
near-unit scalar response is not accepted as a clean one-particle transfer
unless the independent defect-pair and wall-localization conditions also pass.

## 4. Control and sensitivity collections

Exact controls use the same spectral schema at `Nx=20,Ny=40`. For each soft
and hard wall, the six variants are topological `M128`, `M256`, `M512`,
topological seam-gauge `M256`, matched trivial `M256`, and complex-conjugated
Chern `M256`: 12 combined-direction pairs total. Their generated source frames
use schema `wall_diabatic_exact_control_source_v1` with checksum-bound
`wall_diabatic_exact_control_source_completion_v1` siblings.

These controls do **not** twist a state-derived surrogate
`I - 2*P_exact(0)`. At every saved flux the runner rebuilds the canonical OW
functions and the exact equilibrium Hamiltonian `H_exact(phi)`. Translation
symmetry is then used to diagonalize fixed-`k_y` blocks, match the complete
eigenbasis of each block by Hungarian overlap, polar-align degenerate
subspaces, and retain the initial occupation labels. The exact control is
therefore the momentum-resolved legacy spectral-flow calculation. Its global
rank-pair diagnostics are informational: the monitored-state rank-two
isolation threshold is not applied to a translation-symmetric wall band
containing adjacent momenta. Two additional scalar diagnostics,
`exact_block_off_diagonal_maximum` and `exact_seam_restoration_error`, record
translation-block leakage and common-gauge restoration.

Sensitivity pairs use the primary endpoint dependency and the same result
schema. Variant identity is explicit in `metadata_json` and in the directory
name. Five variants are evaluated on 410 unique endpoints: `M128`, `M512`,
`seam_M256`, `radius3_M256`, and `rank4_M256`, for 2,050 combined-direction
pairs. A sensitivity cannot overwrite its M256 primary pair.

## 5. Analysis products

The analysis admits only checksum-verified spectral result/completion pairs.
Its durable products are:

| File | Contents |
|---|---|
| `analysis/samplewise_raw_qx.csv` | One primary row per sample and direction, including raw endpoint charges, crossing diagnostics, source Chern number, resolution status, defect values, and provenance paths |
| `analysis/all_verified_raw_qx.csv` | Primary, bridge, control, and sensitivity directional rows |
| `analysis/raw_directional_statistics.csv` | Per-cell/wall/direction counts, raw means and SDs, trajectory-bootstrap 95% intervals, correctly signed event fractions, and resolved fractions |
| `analysis/raw_directional_curves.csv` | Mean and sample SD of `q_x`, `Delta N_L`, and `Delta N_R` at every flux point |
| `analysis/analysis_summary.json` | Pair counts, invalid-pair ledger, resolved reasons, exponential gap fits, bridge sensitivity, sensitivity completeness, source identities, and figure paths |
| `analysis/figures/wall_diabatic_raw_qx_paths.{pdf,png}` | Separate soft/hard and CCW/CW mean paths with SD bands |
| `analysis/figures/wall_diabatic_endpoint_histograms.{pdf,png}` | Separate soft/hard and CCW/CW raw endpoint distributions |
| `analysis/figures/wall_diabatic_numerical_diagnostics.{pdf,png}` | Endpoint response versus external gap, link quality, and multi-cut spread |

The exponential crossing-gap fit uses physical wall separation `W=Nx/2` and
resamples independent trajectories within each width. The CPU/GPU bridge is
an unpaired sensitivity analysis with cell--wall--direction fixed effects and
a stratified bootstrap interval. Bridge rows are excluded from the primary
S100 statistics by construction.

## 6. Status vocabulary

- **verified endpoint:** result/completion identity and checksum match and the
  frame passes shape, complex128, rank, Gram, backend, wall, and source gates;
- **verified spectral pair:** result/completion/dependency hashes match and all
  array-shape and numerical gates pass;
- **pending:** missing, partial, mismatched, corrupt, or failed work; never a
  synonym for zero response;
- **resolved:** the preregistered edge block satisfies every qualification
  threshold for that direction;
- **unresolved edge cluster:** finite scientific output whose edge block fails
  a fixed qualification threshold; not an execution failure; and
- **complete campaign:** every required pair and control verifies, all
  sensitivities exist, and the final analysis/acceptance checklist is written.
