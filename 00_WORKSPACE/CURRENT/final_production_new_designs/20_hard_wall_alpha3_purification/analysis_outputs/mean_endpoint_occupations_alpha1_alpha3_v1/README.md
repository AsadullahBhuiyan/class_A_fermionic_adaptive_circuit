# Mean endpoint occupation spectra

Hard wall, Nx=20, Ny=40, T=160, alpha2=30, nshell=1; 100 independent Born trajectories per alpha1, maximally mixed initialization and perfect correction. Alpha1=1 is bundle 07; alpha1=3 is bundle 20. Restrict to x=5,...,15 (880 active modes), diagonalize each sample, sort ascending, then average at fixed rank. Shading is sample-wise SEM, not bootstrap. Panel (b) magnifies the same rank transition. No fitting or averaged covariance is used. Input hashes and numerical checks are in summary.json; unrounded sample spectra are preserved in the NPZ. Only sub-1e-9 roundoff outside [0,1] is clipped for plotting.

A broadened rank-averaged step can reflect sample-dependent charge, and does not by itself establish mixed modes or gaplessness in individual trajectories.
