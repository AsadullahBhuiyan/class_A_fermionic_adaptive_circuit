# Browser import: square hard-wall Chern campaign

## Current status: complete, 2026-09-29

All 25 result/completion pairs are now present and verified: **100 independent
trajectories each at L=20, L=30, and L=40**, with cycles 0–40 and ten Chern
centers per trajectory/cycle (123,000 individual Chern values in total).
The final increment adds L=40 samples 20–99 in 16 batches.

Downloaded through Chrome into four original ZIPs retained in Downloads:
`drive-download-20260929T060210Z-1-001.zip` through `-004.zip`.
`IMPORT_COMPLETION_20260929.json` records their paths/checksums, exact members,
the new imported file hashes, and validation of all 25 campaign batches.
The original `IMPORT_MANIFEST.json` is preserved. The local-only helper
`../../import_completed_download.py` reproduces the incremental import and
validation; production scripts and Drive outputs were not changed.

Validation includes byte counts/SHA-256, source/configuration/task metadata,
sample indices 0–99 for each size, cycle coverage, deterministic distinct
centers, center averages, complex128 endpoint frames, finite values and zero
padding, and final frame rank/charge/norm agreement.

The older snapshot and relaxation analysis directories describe the partial
snapshot; their L=40 plots still use S=20 and have not been overwritten.

## Full-ensemble figure, 2026-09-30

`../../analysis_outputs/convergence_and_endpoint_marker_S100/` uses all 100
trajectories at each size. It contains a 2-by-1 figure: cycle-resolved absolute
Chern deviation with a linear inset, and the L=30 endpoint local Chern marker
averaged over all 100 samples. Each nonlinear observable is computed per
trajectory before averaging; the covariance matrices are not averaged first.
`../../plot_convergence_and_marker.py` reproduces the validation, figure,
CSV/NPZ plot data, provenance JSON, and caption. The raw position-commutator
marker retains periodic-seam artifacts and is not the finite-radius disk
estimator in the convergence panel. No dynamics were rerun.

## Original partial import, 2026-09-28

Imported 2026-09-28 from
[Drive output folder](https://drive.google.com/drive/folders/1xb0KlsPuwj7nT24Y_4tsDh1QJftCFMcG).
The browser downloaded the 19 files selected at 22:20 UTC into three ZIPs:
`/home/abhuiyan/Downloads/drive-download-20260928T222020Z-1-001.zip`,
`-002.zip`, and `-003.zip`. All originals remain untouched.

Contains nine NPZ/receipt pairs and the original execution plan:
100 trajectories at L=20, 100 at L=30, and 20 at L=40 (samples 0–19).
Every saved trajectory contains cycles 0–40. The L=40 campaign is incomplete
in this snapshot; no output or receipt has been manufactured for pending tasks.

`IMPORT_MANIFEST.json` records ZIP paths, ZIP hashes, exact archive members,
and imported file hashes. Every NPZ was verified against its original
completion receipt and independently passed the runner's scientific checks.
Analysis lives in `../../analysis_outputs/snapshot_L20S100_L30S100_L40S20/`.
The production bundle, deployed manifest, Drive files and running Colab
session were not modified. `analyze_download.py` is a local-only import and
analysis helper, not a notebook dependency.
