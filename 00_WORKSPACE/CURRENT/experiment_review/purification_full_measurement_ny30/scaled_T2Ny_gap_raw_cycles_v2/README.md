# Restored T=2Ny panel C

Restores the original size sweep at T=2Ny, NOT fixed T40 or T60.
Nx20, Ny20,24,30,36,44,56,60; 100 trajectories per size. Independently
revalidated all 140 Campaign 13 result/receipt pairs and exactly matched
the original seven mean gaps and SEMs. Each sample gap is the minimum
absolute modular energy divided by 2T, before sample averaging.
Fit: SEM-weighted log-space WLS, ordinary propagated sampling errors.
This remains a finite-time scaled-duration fit, not a proven infinite-time law.

Panel C is slab-only with Born-conditioned exterior; A/B/D retain distinct
full-measurement Campaigns 21/22 at Ny30. All are hard-wall alpha2=30,
nshell1, perfect-correction raster-y covariance dynamics. Caption retains
protocol and estimator differences. A/B show raw cycles through60;
B has alpha1=1. C is log-log with clearer ticks. D retains the same T60
mode-density heatmap; its time appears only in the caption.
Earlier fixed-time versions and manuscript are unchanged.
Reproduce: python restore_gap_2ny.py.
