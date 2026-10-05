# Bundle-20 alpha1=3 Born purification control

The `hard_wall_alpha3_maxmix_nx20_ny40_s100_4ny_v1` collection is the
hard-wall control at Nx=20, Ny=40, alpha1=3, alpha2=30, nshell=1,
100 independent Born trajectories, and T=4Ny=160 cycles. Initialization is
maximally mixed on the active slab, raster-y ordering, perfect correction,
complex128, and **no postselection**. It must not be pooled silently with
bundle 19's single fully postselected trajectories or bundle 07's alpha1=1 data.

Raw NPZs and completion records are retained under `results/hard/Ny040`.
Each immutable result shard contains five trajectories. Twenty shards cover
sample indices 0..99; execution used resident batches of 40,40,20 trajectories.
All observations include cycle zero and then every cycle through 160.

Saved data include full physical occupation spectra, cell entropy contours,
cell charge-variance contours, total entropy/charge/variance, per-cycle and
cumulative realized Born log probabilities, record-event counts, numerical
diagnostics, and the final physical centered covariance for each trajectory.
No covariance histories or endpoint eigenvectors are stored; eigenvectors
can be extracted later from the saved final covariance.

The collection's `DOWNLOAD_MANIFEST.json` records verified completion,
byte counts, SHA-256 checksums, Drive file IDs, resolved configuration,
executed-source hashes, coverage, and numerical consistency checks.
Original compressed NPZ bytes are preserved. Import changes nothing on Drive.
Reproduce import checks using `import_drive_results.py --staging <download-directory>`.
