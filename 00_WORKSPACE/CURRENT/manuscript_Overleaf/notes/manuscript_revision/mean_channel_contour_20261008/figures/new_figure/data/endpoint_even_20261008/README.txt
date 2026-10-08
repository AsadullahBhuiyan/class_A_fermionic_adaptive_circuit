Current endpoint inputs for Figures 6(a,b) and 8(b,c), imported 2026-10-08.

Nx=20; Ny=24,28,32,40,50,60; 100 independent trajectories per size; T=2Ny; alpha_1=1, alpha_2=30, n_shell=1, slab-only hard-wall measurements, perfect correction, Born-prepared pure exterior. No simulation was performed for this manuscript revision.

The compact NPZ contains per-trajectory origin-averaged entropy, quantum charge variance, and left/right contour sums for all widths Ay=0..Ny/2. The Ny32 half-strip contour is stored as (sample, relative dy, x). Source NPZ shards remain unchanged in experiment_review/endpoint_figures_import_20261008. provenance.json binds the compact arrays to all 120 source shards and receipts. fits.json and curves.csv are exact copies of the imported analysis products.

Shared half-strip anchored fits use 8<=Ay<=Ny/2 and equal total weight per size. The uncertainty propagates trajectory covariance across widths and the anchor. Both figure renderers recompute the fits from these arrays and compare with the imported results. R^2 in plot annotations retains the agreed display convention for the uncentered anchored-fit statistic.

Figure 6(c) uses the existing independent convergence histories. Figure 8(a) uses the new Ny32 alpha_1=1 contour and the existing independent alpha_1=3 control. These independent ensembles are not pooled.
