# Downloaded bundle-19 postselected data

## Latest: complete v2 every-cycle contour campaign

`postselected_maxmix_hard_soft_alpha1_3_nx20_ny40_s1_4ny_contours_gpu_v2/`
contains all four completed trajectories: alpha1=1 and 3, hard and soft walls.
Imported originals total 209,905,149 bytes (about 210 MB), four NPZ/completion
pairs, all through cycle 160. The folder layout is
`alpha1_<1|3>/results/<hard|soft>/postselected_trajectory.npz` plus `completion.json`.

Every run includes `entropy_contour` of shape (161,20,40) and
`entropy_contour_x` of shape (161,20), as well as total entropy and gap histories.
The full endpoint covariance, occupations and eigenvectors remain included.
These are one fully postselected trajectory per configuration, not a Born ensemble.

`DOWNLOAD_MANIFEST.json` pins all eight original file hashes and Drive IDs,
executed source hashes, resolved per-alpha configurations, and validation results.
Byte counts and SHA-256 match the original completion records. Every-cycle
contour sums agree with total entropy, and endpoint contours agree with those
reconstructed from the saved eigensystem. The four existing v2 numerical
contracts, time coverage and endpoint residual spot checks passed.
Drive data and the older v1 collections below were not modified.

## Historical v1 imports (without every-cycle contours)

Imported from Google Drive on 2026-09-24. These are single fully postselected
trajectories, not the 100-sample Born ensemble of bundle 20. All are Nx=20,
Ny=40, maximally mixed initialization, nshell=1, raster-y, alpha2=30, T=160,
complex128, postselect=True, postselect_probability=1, perfect_correction=False.

| Collection | alpha1 | Completed constructions | Trajectories |
|---|---:|---|---:|
| `postselected_maxmix_hard_soft_nx20_ny40_s1_4ny_gpu_v1` | 1 | hard, soft | 1 each |
| `postselected_maxmix_hard_soft_nx20_ny40_alpha3_s1_4ny_gpu_v1` | 3 | hard only | 1 |

Each collection retains the original `results/<construction>/postselected_trajectory.npz`
and `completion.json`, plus an import `DOWNLOAD_MANIFEST.json` recording the
remote file IDs, checksums, executed-source hashes and hash-verified runtime config.
The original NPZ files are kept intact, not expanded into separate NPY files.

Every trajectory contains cycles 0 through 160, total entropy, active charge,
occupation-derived gap histories and numerical diagnostics. Endpoints retain
the full active covariance, centered spectrum, occupations, complex eigenvectors,
and active basis indices: dimension 880 for hard walls and 1600 for soft walls.
The per-time Lyapunov gap is undefined (NaN) at t=0; later values are finite.
Result sizes and SHA-256 values agree with the remote completion records;
source hashes match the repository at import, configuration hashes were reproduced,
and eigenvector residuals for 16 endpoint columns per result are below 8e-15.

The alpha3 run uses the shipped source config file unchanged but overrides
alpha1, sampling revision, and construction_order in the notebook. Its resolved
runtime configuration is preserved explicitly in its download manifest; do not
mistake the source-config hash for an alpha1=1 execution.

Three completed pairs total, about 127.3 MB. Remote files and earlier campaigns
were not altered. No soft-wall alpha3 result was present at import.
