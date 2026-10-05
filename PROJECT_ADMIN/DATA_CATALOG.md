# Data catalog

Inventory date: 2026-09-12.  Sizes are filesystem usage reported by `du`; file counts are
regular files. `Active` means code, notebooks, and generated results that should stay
grouped as one working experiment package. `Warm` means useful legacy evidence retained
locally. `Archive candidate` means a large standalone result tree that may be copied
externally after a separate decision; it never means erroneous or disposable.

## Active/current: keep local

| Path | Size | Files | Newest | Reason |
|---|---:|---:|---:|---|
| `00_WORKSPACE/CURRENT/final_production_new_designs/` | 59 GB | 3,382 | 2026-09-12 | Independent redesigned Colab campaigns and their owning active data, including completed max-mix Lyapunov, entropy/charge, purification, tangent-replay, and hard-wall x-resolved correlator imports |
| `00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/` | 26 GB | 10,441 | 2026-09-08 | Active flux-charge pilot with the checksum-verified 1,150-trajectory wall-pump endpoint import kept beside its downstream analysis |
| `00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/` | 112 MB | 479 | 2026-08-27 | Preserved prior-design production bundles |
| `00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/` | 4.5 GB | 1,611 | 2026-09-24 | Active merged Colab outputs, including the verified slot-17 tangent-gap v2 import |
| `00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall/` | 4.6 GB | 556 | 2026-08-17 | Accepted exact transverse-width campaign and provenance |
| `00_WORKSPACE/CURRENT/experiment_review/b1_controller_frame/` | 100 MB | 57 | 2026-08-17 | Current controller-frame mechanism audit |
| `00_WORKSPACE/CURRENT/validation_campaigns/` | 240 MB | 426 | 2026-08-17 | Current engine and campaign validation |
| `00_WORKSPACE/CURRENT/topological_frustration_diagnostics/results/` | 267 MB | 448 | 2026-08-16 | Recent admissible diagnostic evidence |
| `00_WORKSPACE/COLAB/colab_small_system_testing/` | 10.64 GiB | 494 | 2026-08-17 | Active Colab package; notebooks, code, GPU data, and analyses stay together |
| `00_WORKSPACE/COLAB/colab_charge_fluctuations/` | 1.03 GiB | 2,504 | 2026-08-17 | Active Colab package; notebooks, code, CPU/GPU data, and analyses stay together |
| `00_WORKSPACE/COLAB/final_production_drive_recovery_2026-09-01/` | 986 MiB | 61 | 2026-09-01 | Checksum-verified recovery of all 58 server-visible P1/H1 v3-v4 files plus recovery metadata |

The former `00_WORKSPACE/CURRENT/final_production_new_designs/` tree was frozen
on 2026-09-01 at
`00_WORKSPACE/LEGACY/final_production_new_designs_quarantine_2026-09-01/`.
The matching Drive deployment was renamed in place to
`QUARANTINED_2026-09-01_final_production_new_designs_v4`; its output folders were
not deleted or renamed. The replacement
`CURRENT/final_production_new_designs/` began as an empty rebuild boundary and
now contains the new independent campaign bundles and their active outputs.

The 2026-08-27 Downloads merge is recorded in [`downloads_merge_20260827.json`](../00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/_import_receipts/downloads_merge_20260827.json). The retained ZIP is the recovery copy; the abandoned atomic-write temporary file was excluded.

The completed max-mix many-body Lyapunov v2 A100 campaign was imported from
Google Drive on 2026-09-04 into its owning bundle at
`00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_manybody_lyapunov_pilot/gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/`.
Its 160 NPZ/completion pairs contain 800 trajectories and occupy 35,946,936
downloaded bytes. All pairs passed the campaign's checksum/configuration/source
verification; Drive provenance and the numerical audit are recorded in that
directory's `DOWNLOAD_MANIFEST.json`. The completed 2,000-replicate diagnostic
analysis is in the bundle's
`analysis_outputs/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/`.
It validates exact many-body-level reconstruction, same-window record-weight
closure, and boundary localization, but rejects `T=2Ny` temporal sufficiency
and all precision CFT-coefficient claims.

The completed hard-wall `T=4Ny` max-mix many-body Lyapunov campaign was
imported from Google Drive on 2026-09-17 into
`00_WORKSPACE/CURRENT/final_production_new_designs/13_maxmix_manybody_lyapunov_4ny/gpu_data/maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled/`.
It contains 140 NPZ/completion pairs, 700 trajectories, and 348,991,195
downloaded bytes across `Ny=20,24,30,36,44,56,60`. Every result checksum,
configuration/source identity, cycle and spectrum mask, and sample index
0--99 passed local verification. The bundle-local `DOWNLOAD_MANIFEST.json`
records Drive provenance, the aggregate raw-file inventory hash, and numerical
residual maxima. The soft-wall arm was not run or imported.

The completed hard-wall endpoint entropy/charge campaign was imported from its
two complementary Drive lanes on 2026-09-08 into
`00_WORKSPACE/CURRENT/final_production_new_designs/05_hard_wall_entropy_charge_batched_v2/gpu_data/hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint/`.
It contains 140 NPZ/completion pairs, 700 trajectories, and 12,527,731 bytes.
The campaign runner independently reports 40/40 valid lane-A shards and
100/100 valid lane-B shards with no invalid or pending work. Per-file hashes,
Drive provenance, and the exact config/source identity are recorded in the
data directory's `DOWNLOAD_MANIFEST.json`.

The full completed-Drive inventory and canonical destinations are recorded in
[`COMPLETED_DRIVE_CAMPAIGNS_20260908.md`](COMPLETED_DRIVE_CAMPAIGNS_20260908.md).
The hard/soft purification, pure tangent replay, and wall-pump endpoint imports
were completed on 2026-09-09. Across those four imports, all 382 result/completion
pairs and the wall-pump benchmark receipt match the Drive server inventory by
path, byte count, and SHA-256; each owning data root contains a concise
`DOWNLOAD_MANIFEST.json`.

The completed hard-wall x-resolved endpoint correlator campaign was imported
from Google Drive on 2026-09-12 into its owning bundle at
`00_WORKSPACE/CURRENT/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling/gpu_data/hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_endpoint_frame_halfcov_occupations_v2_30gib_batched/`.
It contains 60 NPZ/completion pairs, 300 trajectories, and 15,239,260,880
downloaded bytes across `Ny=40,50,60`. All result hashes match their completion
records, each size covers sample indices 0--99 exactly once, and the campaign's
own report-only validator finds 60/60 valid shards, seven completed execution
batches, no invalid pairs, and no pending work. Drive provenance and per-file
hashes are recorded in that directory's `DOWNLOAD_MANIFEST.json`.

The completed slot-17 hard-wall pure-state tangent-gap execution-v2 outputs
were imported from Google Drive on 2026-09-24 into
`00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2/`.
All 374 v2 NPZ/completion pairs (6,675 trajectories and 3,953,552,384 bytes)
pass the campaign runner's task-identity, configuration/source, byte-count,
SHA-256, and NPZ-schema checks. The preserved 563,231,101-byte v1 qualification
archive supplies the remaining 25 trajectories and is stored in the adjacent
versioned v1 directory. Its local byte count and pinned SHA-256 match. The
campaign runner accepts all 375 logical tasks and all 6,700 trajectories with
zero failures; exact Drive IDs and import details are recorded in both import
provenance files.

## Large standalone result trees: possible archive candidates

| Path | Size | Files | Newest | Proposed archive class |
|---|---:|---:|---:|---|
| `cache/G_history_samples/` | 663 GB | 85 | 2026-06-04 | Legacy dense covariance histories |
| `00_WORKSPACE/LARGE_RESULTS/choi_covariance_cpu/cpu_data/` | 113 GB | 1,051 | 2026-08-01 | CPU Choi campaign raw data |
| `00_WORKSPACE/LARGE_RESULTS/lyapunov_analysis_v2/cache/` | 4.4 GB | 7 | 2026-04-23 | Legacy Lyapunov raw cache |
| `00_WORKSPACE/LARGE_RESULTS/dw_convergence/` | 955 MB | 62 | 2026-06-05 | Legacy convergence campaign |
| `00_WORKSPACE/LARGE_RESULTS/experiments/` | 462 MB | 333 | 2026-06-18 | Older experiment outputs and code |

The five possible archive candidates total 837,778,719,908 bytes across 1,535
inventoried files. No source dataset has been deleted or moved. An archive attempt on
2026-08-17 was canceled at the user's request before any dataset receipt completed; see
the runbook for the preserved incomplete destination-copy inventory.

## Archive-candidate manifest status

| Dataset | Files | Exact bytes | Source manifest |
|---|---:|---:|---|
| `cache/G_history_samples/` | 85 | 711,135,758,665 | [`cache_G_history_samples.json`](archive_manifests/cache_G_history_samples.json) |
| `00_WORKSPACE/LARGE_RESULTS/choi_covariance_cpu/cpu_data/` | 1,051 | 120,497,302,728 | [`choi_covariance_cpu_data.json`](archive_manifests/choi_covariance_cpu_data.json) |
| `00_WORKSPACE/LARGE_RESULTS/lyapunov_analysis_v2/cache/` | 7 | 4,662,308,188 | [`lyapunov_analysis_v2_cache.json`](archive_manifests/lyapunov_analysis_v2_cache.json) |
| `00_WORKSPACE/LARGE_RESULTS/dw_convergence/` | 60 | 1,000,244,147 | [`dw_convergence.json`](archive_manifests/dw_convergence.json) |
| `00_WORKSPACE/LARGE_RESULTS/experiments/` | 332 | 483,106,180 | [`experiments.json`](archive_manifests/experiments.json) |

These are source inventories, not archive receipts. External copies must still be read
back and verified before any source deletion can even be proposed.

## Active Colab data integrity inventories

The following four pre-existing checksum manifests are retained as useful integrity
records, but the archive runner does not consume them. Together they cover 2,728 files
and 12,360,081,037 bytes. Their data remains in its producing Colab package.

| Active package data | Files | Exact bytes | Integrity inventory |
|---|---:|---:|---|
| `00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/` | 243 | 10,568,304,887 | [`colab_small_system_testing_gpu_data.json`](archive_manifests/colab_small_system_testing_gpu_data.json) |
| `00_WORKSPACE/COLAB/colab_small_system_testing/analysis_outputs/` | 227 | 850,581,846 | [`colab_small_system_testing_analysis_outputs.json`](archive_manifests/colab_small_system_testing_analysis_outputs.json) |
| `00_WORKSPACE/COLAB/colab_charge_fluctuations/gpu_data/` | 93 | 926,561,008 | [`colab_charge_fluctuations_gpu_data.json`](archive_manifests/colab_charge_fluctuations_gpu_data.json) |
| `00_WORKSPACE/COLAB/colab_charge_fluctuations/cpu_data/` | 2,165 | 14,633,296 | [`colab_charge_fluctuations_cpu_data.json`](archive_manifests/colab_charge_fluctuations_cpu_data.json) |

## Git storage

After explicit approval, 1,267 Git-classified `tmp_pack_*`/`tmp_obj_*` files totaling
788,075,533,657 bytes (733.953 GiB) were permanently removed.  Git now reports zero
garbage and `.git` contains 399.815 GiB of regular files: seven valid packs totaling
383.410 GiB plus approximately 16.42 GiB of loose objects.  Connectivity verification
passed and the working-tree changes remain visible.  The valid history must remain until
the clean successor is promoted or another verified historical copy exists.

## External-storage status

`/data/abhuiyan` is writable on the separate 42 TB filesystem. The canceled attempt left
an incomplete, unverified destination tree at
`/data/abhuiyan/class_A_fermionic_adaptive_circuit_cold_20260817`: 42 finalized copies,
one partial file, approximately 377.3 GB total, and zero verified receipts. It is not an
archive and will not be resumed or removed without explicit user direction.
