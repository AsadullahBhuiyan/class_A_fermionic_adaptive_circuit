# Single-column purification and endpoint-gap figure

Reproduce using `python plot_purification_gap_summary.py` in bundle 13.
PDF size: 3.375 by 6.5 inches; PNG: 300 dpi. Earlier v1/v2 figures are preserved.
Use an ordinary `figure` with `\includegraphics[width=\columnwidth]{hard_wall_purification_contours_gap_3x1.pdf}`
in two-column RevTeX; neither `wide text` nor `figure*` is required.

## Caption

Hard-wall purification, Nx=20, alpha1=1, alpha2=30, nshell=1,
raster-y measurements and perfect correction, starting from a maximally
mixed active slab with product-state exterior modes. Each size has 100
independent Born trajectories evolved for T=4Ny cycles. (a) Total entropy
averaged sample-wise, divided by Ny, for Ny=20,30,40. (b) Spatial entropy
contours summed over y within each trajectory and then averaged, divided
by Ny; x=5,15 are the walls and x=10 is the center. Panels (a,b) use the
same bundle-07 trajectories and every-cycle measurements. Shading is one
sample SEM. (c) Endpoint Lyapunov half-gap from independent bundle-13 data,
Ny=20,24,30,36,44,56,60. For each sample compute
Delta=min_j |log(nu_j)-log(1-nu_j)|/(2T) before ensemble averaging;
exact-cap modes remain infinite. Error bars are sample standard deviations
divided by sqrt(100). The dashed curve fits the seven size means to
Delta=A Ny^(-z) in log space, weighted by SEM/mean. No bootstrap or
time-point resampling is used. Fit errors are one-standard-error parameter
uncertainties from the absolute weighted least-squares covariance.

## Fit interpretation

This reuses the earlier endpoint-gap fit: A=1.60416 +/- 0.20516,
z=1.11395 +/- 0.03545, chi-squared=2.84653 for five degrees of freedom.
The power law is a finite-size fit to finite-time endpoints T=4Ny, not
an independent determination of an asymptotic dynamical exponent. Quoted
fit errors do not include finite-depth or finite-size systematic effects.
Machine-readable fit results and source verification are in analysis_summary.json.
