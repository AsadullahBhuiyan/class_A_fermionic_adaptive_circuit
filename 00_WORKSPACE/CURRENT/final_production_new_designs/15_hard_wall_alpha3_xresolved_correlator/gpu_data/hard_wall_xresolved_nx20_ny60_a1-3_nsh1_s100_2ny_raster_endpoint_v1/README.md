# Completed alpha-3 hard-wall Ny60 correlator import

Downloaded from Google Drive on 2026-09-15. The original Drive files are unchanged.

- Campaign: `hard_wall_xresolved_nx20_ny60_a1-3_nsh1_s100_2ny_raster_endpoint_v1`.
- Source: https://drive.google.com/drive/folders/1JTBMudFpBVgRymHSkfNXjffxlqGIUXWY
- Data: `results/Ny060/`, 20 original NPZ/completion-JSON pairs, 100 unique samples (0–99).
- Geometry/protocol: Nx=20, Ny=60, alpha_1=3, alpha_2=30, nshell=1, hard/support-terminated walls at [5,15], raster_y, perfect correction, complex128.
- Endpoint: 120 physical cycles (2Ny); independent sampling revision, not pooled with alpha_1=1.
- Saved: sample-resolved x-resolved squared correlators, their x averages, charge, half-filling offsets, and identity metadata. No endpoint occupied frames or covariance matrices are part of this campaign.
- Final completion record: 2026-09-15 09:36:55 UTC.
- Import checks: all 20 result SHA-256/byte counts match their completion records; all 40 byte counts match Drive metadata; samples cover 0–99 exactly once; endpoint/scientific metadata checked; recorded executable source hashes match the local bundle; saved x averages exactly equal the mean over the 20 x columns.
- Original downloaded files total 714,043 bytes. No rolling checkpoints remained on Drive at completion.

File IDs, checksums, source identity, and validation summary are recorded in
`PROJECT_ADMIN/drive_import_manifests/hard_wall_alpha3_ny60_20260915.remote.json`
(repo-relative path).

Recheck imported file integrity from the repository root:

```bash
python scripts/verify_completed_drive_imports.py --manifest PROJECT_ADMIN/drive_import_manifests/hard_wall_alpha3_ny60_20260915.remote.json
```
