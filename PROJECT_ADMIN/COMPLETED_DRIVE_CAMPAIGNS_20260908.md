# Completed Google Drive campaign imports — 2026-09-08

This inventory records completed, non-bulk production campaigns visible in the
Google Drive account `abhuiyan2398@gmail.com` and their canonical local homes.
Each campaign stays with the bundle or downstream experiment that owns it. No
second monolithic copy is created.

## Import state

| Campaign | Completed scope | Canonical local destination | State |
|---|---|---|---|
| `maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled` | hard wall; Ny = 20, 24, 30, 36, 44, 56, 60; S=100 each; T=4Ny | `00_WORKSPACE/CURRENT/final_production_new_designs/13_maxmix_manybody_lyapunov_4ny/gpu_data/maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled/` | Downloaded and checksum-verified: 140 pairs, 700 trajectories, 348,991,195 bytes; soft-wall arm was not run |
| `maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2` | Ny = 20, 22, 24, 26, 28, 30, 36, 40; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_manybody_lyapunov_pilot/gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/` | Already local and checksum-verified (800 trajectories) |
| `domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25` | completed hard-wall lane: Ny = 20, 24, 28 and 21 alpha values; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/gpu_data/domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25/` | Downloaded and checksum-verified: 210 pairs, 6,300 trajectories, 3,193,042 bytes; soft-wall lane remains separate and incomplete |
| `domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1` | completed hard-wall lane: Ny = 24, 28, 32; alpha1 = 1, 3; nshell = 1, 2, dense; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/08_domain_wall_correlator_scaling/gpu_data/domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1/` | Downloaded and checksum-verified: 72 pairs, 1,800 trajectories, 236,177,281 bytes; soft-wall lane remains unrun |
| `hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint` | Ny = 30, 35, 40, 45, 50, 55, 60; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/05_hard_wall_entropy_charge_batched_v2/gpu_data/hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint/` | Downloaded and checksum-verified (700 trajectories) |
| `soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint` | completed lane A: Ny = 40, 60; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/10_soft_wall_entropy_charge_batched_v2/gpu_data/soft_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint/` | Downloaded and checksum-verified: 40 pairs, 200 trajectories, 3,926,385 bytes including the lane-A A100 benchmark; lane B remains incomplete |
| `maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2` | hard wall, Ny = 20, 30, 40; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification/gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2/` | Downloaded and checksum-verified: 60 pairs, 300 trajectories, 8,175,889,380 bytes |
| `maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3` | corrected soft wall, Ny = 20, 30, 40; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification/gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3/` | Downloaded and checksum-verified: 60 pairs, 300 trajectories, 8,175,884,825 bytes |
| `pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1` | hard/soft, Ny = 24, 28, 32, alpha1 = 1, 3; S=100 each | `00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/gpu_data/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1/` | Downloaded and checksum-verified: 32 pairs, 1,200 trajectories, 25,280,965,143 bytes |
| `wall_pump_width_endpoints_s100_v1` | 1,000 primary plus 150 bridge endpoints | `00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/imported_endpoints/wall_pump_width_endpoints_s100_v1/` | Downloaded and checksum-verified: 230 pairs plus benchmark receipt, 1,150 trajectories, 16,905,617,976 bytes |
| flattened ground-state reference | full exact reference task matrix | `00_WORKSPACE/CURRENT/final_production_new_designs/06_domain_wall_flattened_ground_state_reference/results/` | Already local and completion-verified |

The hard and soft purification revisions are deliberately kept in separate
revision directories. The scientific comparison uses hard-v2 and corrected
soft-v3; their result identities must not be rewritten or silently pooled.
The four large imports comprise 382 NPZ/JSON result pairs plus one benchmark
receipt and exactly 58,538,357,324 bytes (decimal 58.54 GB). They were
downloaded on 2026-09-09 without changing the Drive sources, then independently
verified against the server inventory with zero missing or mismatched files.
Every expected result and receipt is pinned by relative path, Drive file ID,
byte count, and SHA-256 in
`PROJECT_ADMIN/drive_import_manifests/completed_campaigns_20260908.remote.json`.
Revalidate the local import at any time with:

```bash
python scripts/verify_completed_drive_imports.py
```

The verifier never repairs or deletes data. It reports exact missing or
mismatched paths and exits nonzero until all imports are complete.

The hard-wall bipartite-MI lane was imported separately on 2026-09-09 after it
reached 210/210 durable macros. Its bundle-local `DOWNLOAD_MANIFEST.json` pins
all 420 Drive objects by relative path, file ID, byte count, and downloaded
SHA-256. The campaign runner independently recognizes all 210 pairs with zero
invalid or partial tasks. The unfinished soft-wall lane was not copied into the
repository snapshot.

The completed hard-wall correlator lane and completed soft-wall entropy lane A
were imported on 2026-09-10. Their bundle-local `DOWNLOAD_MANIFEST.json` files
pin the Drive object identities, downloaded byte counts, and SHA-256 values.
The campaign runners independently recognize 72/72 hard correlator tasks and
40/40 lane-A entropy shards with zero invalid or partial products. No
incomplete soft-correlator or lane-B entropy result was copied.

## Drive provenance

| Drive object | File/folder ID |
|---|---|
| output root `classA_final_production_outputs` | `1nK13sAEG5-fT9vPDOv-NtJym4q9ZMHIb` |
| max-mix Lyapunov hard-wall 4Ny v4 campaign | `1yhfUxvmi7NWRdCLGjvpK7HS4OnvHOvpI` |
| max-mix Lyapunov hard-wall 4Ny v4 results | `122Gg2_EF-ss33JlFMnOdWrBuvQwMeaay` |
| max-mix Lyapunov v2 campaign | `1D6DZ69-shgbjKo2cQY3EzLh1UEhBI_r9` |
| bipartite-MI v2 campaign | `1VWs45gj0VzcH5K52e3CZl0UJZkUTzbEg` |
| hard-wall correlator campaign | `1vQaltRyATGK2oPcedK4vC0R5FSM7U8x7` |
| hard-wall correlator results | `1hKLSgZxxiZzuCp94-iGiq6qYdjSBdgQ1` |
| soft-wall entropy campaign | `1-nL0baf0Z1PWowsJ0vCmREj3KX7ucB8m` |
| soft-wall entropy lane-A results | `1BcMss61Niyn6h900nhb7K7CFnohH3kRu` |
| hard-wall endpoint lane B campaign root | `1ylJBW6zjWeHzWL5gYO56b8ijK8SWSx_H` |
| hard-wall endpoint lane B results | `1Gy0nwk1QJyZG2t70Soevj4QVeFS2-l6I` |
| hard-wall endpoint lane A campaign root | `1rKTLJDequkMAoMcmng4CDsL_NsH9efDq` |
| hard-wall endpoint lane A results | `1r3yn78oYf9SBlG_TkXlqZFmtXAG_kl7o` |
| hard purification v2 campaign | `1Ln4h1tr02buhaGkKKnmnVHPEsvgNOkAb` |
| hard purification v2 results | `1cFOznXqXV_Z0zJO0uZNDx5R__DSe3dwc` |
| corrected soft purification v3 campaign | `1riTCIrTsngqWqXtyb1p67wGdwJnvc4HA` |
| corrected soft purification v3 results | `1tKa_yFyGBBNWPw6JDbAvGSV-vCWJTaq8` |
| pure tangent replay acquisition | `18OtDAft97_D1WRrx4S2cGTxsqkOU1UKS` |
| pure tangent replay results | `1yN7jWmS96dzPGWjcC9XfcVqn3T8dD5za` |
| wall-pump endpoint campaign | `1AtSJCDCEZQwdKv7zPLr63PBw7WYflAOJ` |
| wall-pump primary endpoints | `1hEZh46kbiFDfNXOTxeBTS7arnzK1yXoE` |
| wall-pump bridge endpoints | `1Fnc8HRXBomiRPhrzJW6_U4k542ESHzpz` |
| wall-pump benchmarks | `1HE58buAgb-RdpDw944BLkUz-Zx9T5t2-` |

## Scope boundary

This inventory ordinarily excludes incomplete campaigns. The bipartite-MI
campaign was 276/420 result/completion pairs (9,000/12,600 trajectories) at the
original inventory time. Its scientifically complete 210-macro hard-wall lane
has since been imported as an explicitly partial campaign snapshot; the
unfinished soft-wall lane is excluded. The correlator and soft-wall entropy
campaigns are likewise retained as explicit partial snapshots containing only
their completed hard-wall and lane-A subsets, respectively. The new
modular-spreading S100 campaign had not produced a complete result matrix.
