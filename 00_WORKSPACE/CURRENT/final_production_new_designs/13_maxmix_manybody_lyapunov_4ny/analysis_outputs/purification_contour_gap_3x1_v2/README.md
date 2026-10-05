# Hard-wall purification and endpoint gap (revised figure)

Reproduce with `python plot_purification_gap_summary.py` from bundle 13.

Figure caption: Hard-wall purification with Nx=20, alpha1=1, alpha2=30,
nshell=1, raster-y measurements and perfect correction, for 100 independent
Born trajectories per size and T=4Ny cycles. The active slab starts maximally
mixed with product-state exterior modes. (a) Total entropy <S(t)>/Ny for
Ny=20,30,40, exclusively from the bundle-07 hard-wall ensemble. (b) Spatial
contour <s_x(t)>/Ny for the exact same trajectories and sizes at x=5,15
(walls) and x=10 (center), where s_x is the sum of the trajectory's entropy
contour over y. Both panels save every cycle, average the trajectory-resolved
observable, and display one sample SEM as shading. Time axes are linear in
t/Ny and entropy axes are logarithmic. (c) Bundle-13 endpoint Lyapunov gap
versus Ny=20,24,30,36,44,56,60, using only T=4Ny. For each trajectory,
Delta=min_j |log(nu_j)-log(1-nu_j)|/(2T); cap modes remain infinite in
magnitude and do not lower a finite minimum. Points average this gap across
100 trajectories, with sample SEM error bars (std(ddof=1)/sqrt(100)). No
averaging over time, bootstrap, inset, or fitted line is used. Connecting
lines are guides to the eye. Bundles 07 and 13 are independent ensembles.

The earlier mixed-dataset/late-segment version is preserved under
`purification_contour_gap_3x1_v1/`. This version uses one dataset for both
entropy panels and only the endpoint gaps for panel (c).
