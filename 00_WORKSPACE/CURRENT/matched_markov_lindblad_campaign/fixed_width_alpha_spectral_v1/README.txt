Fixed-width hard-wall channel spectral sweep (2026-10-01)

Approved grid: Nx=20, Ny=20,40,60,80,100 and alpha_1=1.0,1.1,...,3.0.
105 cases; 21 single-core CPU workers, one alpha per worker. Each worker
sweeps Ny in ascending order. All numerical library thread counts are one.
Existing square-size campaigns and all unrelated jobs remain unchanged.

alpha_2=30, nshell=1, FIXED inclusive walls x=5,15, all slabs active,
hard-wall support truncation, X trial orbitals, periodic boundaries, zero
twist, raster-y Ap/Am/Bp/Bm, perfect correction, measurement dephasing,
complex128. Alpha=2 is included exactly using the canonical zero-norm
prescription (no alpha offset). Canonical CPU OW construction and the
validated square_large_spectral_v1 block-product/eigensolver are reused.
No sampled trajectory, covariance evolution, or cycle horizon is needed.

Both the interior and the entire periodic exterior are diagonalized. The
gap is -2 log rho(A), with rho maximized over the full union of both spectra.
No slab is frozen or discarded. Independent full-system action checks and
dominant eigenpair residuals verify each calculation; unresolved unit modes
are flagged without manufacturing a positive gap. Near-zero eigenvalues
of this nonnormal product are not claimed to be individually forward-accurate.

This is a fixed-width/variable-wall-length limit, NOT simultaneous 2D size
scaling. A finite covariance-sector gap does not prove a full many-body gap.

Each case saves spectrum.npz and a completion.json with configuration,
source hashes, bytes, SHA-256, numerical diagnostics, timing and peak RSS.
Resume skips only verified pairs; interrupted cases rerun. Per-case and
per-alpha logs expose progress. After all 105 cases verify, the queue writes
analysis/gaps.csv and channel_gap_vs_alpha.png/.pdf with five Ny curves.

Launch: python run_scan.py launch
Status: python run_scan.py report --root RESULTS_DIRECTORY
Resume: python run_scan.py launch --root RESULTS_DIRECTORY
The default cores were checked free at launch: 0-7,40-44,46-53. Override
--cpus with 21 distinct allowed IDs when resuming on a differently loaded host.
The coordinator requires 126 GiB available headroom for the 21-worker launch;
each child also checks its geometry-scaled headroom before allocating.
Do not modify scientific source files while the campaign is running.
