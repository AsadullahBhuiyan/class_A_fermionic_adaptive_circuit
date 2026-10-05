# Drive import: hard-wall random-center Chern dynamics

Import date: 2026-09-28. Account: abhuiyan2398@gmail.com.
[Original Drive output folder](https://drive.google.com/drive/folders/1f01t1WI6B-FZ3B6w0WzDYYgKhpx2AygA).
Nothing was changed or deleted on Drive; no dynamics were launched.

## Campaign inventory

Drive contains the execution plan and all 13 expected result/completion pairs,
covering 300 declared trajectories: 100 each at 20x20, 20x30 and 20x40. The
selected execution batches were respectively 100, 50 and 10. Remote filenames
and byte sizes match the receipts; every receipt matches the frozen task and
source/configuration identity. This inventory check is distinct from full
download/checksum/scientific verification of the archives.

## Complete local download and verification

All 13 result/completion pairs and the execution plan are preserved here.
The result archives total 3,525,965,086 bytes (approximately 3.53 GB).

| Geometry | Trajectories | Saved cycles | Individual Chern values |
| --- | ---: | --- | ---: |
| 20x20 | 100 | 0..40 | 41,000 |
| 20x30 | 100 | 0..60 | 61,000 |
| 20x40 | 100 | 0..80 | 81,000 |
| Total | 300 | | 183,000 |

Each archive contains ten sampled centers and their Chern values per
trajectory/cycle, center averages, integer charge histories, and final
complex128 occupied frames.

The ten 20x40 archives were downloaded through the Drive connector. Its
256-MiB file limit prevented downloading the three larger 20x20/20x30
archives. These were imported from the user's manual browser download,
`hard_wall_random_center_chern_nx20_ny20-30-40_s100_r0p2_v1-20260928T153628Z-1-002.zip`,
with bytes and SHA-256 checked against the original completion receipts.
The original Downloads ZIPs were left untouched.

`DOWNLOAD_MANIFEST.json` records source Drive IDs/URLs, local checksums, and
manual ZIP/member provenance. `DRIVE_INVENTORY.json` preserves the original
listing. `VERIFICATION.json` records successful full verification of all
300 trajectories: result/receipt checksums, frozen task/configuration/source
identities, samples 0..99 per geometry, exact cycle coverage, ten unique
deterministic periodic centers, finite Chern values, center averages,
integer charge, and final-frame rank/norm/dtype/padding.

To repeat verification, run the local-only `../../import_drive_results.py verify`
from this directory. No dynamics need to be rerun.

The canonical Colab bundle and its deployed manifest remain unchanged. The
local import helper and this data directory are not notebook dependencies.
