# Squared correlator of the trajectory-averaged channel state

This is a no-fit analogue of technical-report Figure 12, using the new
raster-y hard-wall endpoints at alpha_1=1 and 3. It does not replace Figure 12.

For the occupation matrix `Cbar_ij = Tr(rhobar c_i^dagger c_j)`, compute

`C_Gbar(x,r) = sum_(y,mu,nu) |Cbar[(x,y,mu),(x,y+r,nu)]|^2 / (2 Ny)`

and `C_Gbar^av(r) = mean_x C_Gbar(x,r)`. Here the plot label `C_Gbar`
denotes the squared correlator of the mean occupation matrix, following
the repository's correlator notation; it does not mean squaring the
centered engine covariance `G=2C-I`. The NPZ key `G_final` stores physical
occupations `C`, despite its name. Separations use periodic y, and both
orbital indices are summed. The factor 1/2 matches Figure 12 exactly.

Born averaging is already contained in the channel endpoint. Squaring
comes next, followed by the spatial origin average, optional x average,
and logarithm. We do **not** spatially twirl the covariance before squaring.
The additionally twirled result is saved separately as a diagnostic.

This differs from Figure 12's `mean_x,y,mu,nu E_xi[|C_xi|^2]/2`.
For a common ensemble and initialization,
`E[|C_xi|^2] = |E[C_xi]|^2 + E[|C_xi-E[C_xi]|^2]` elementwise.
An endpoint mean covariance alone cannot reconstruct the second term.
Also, the mixed ensemble need not be Gaussian: this squared two-point
function is not automatically its connected density-density correlator.
No Wick reduction of the averaged state is assumed.

## Inputs and comparison limits

- Source: `results/raster_y_channel_endpoints_v1_20260915T025739Z/alpha{1,3}_hard/`.
- Nx=20, Ny=64, alpha_2=30, nshell=1, walls x=5,...,15 inclusive.
- Full-system maximally mixed initialization; all slabs evolve; hard-wall
  support truncation; perfect correction and number dephasing; complex128.
- Canonical `classA_U1FGTN.run_markov_channel`, fixed raster_y every cycle,
  within-cell order Ap,Am,Bp,Bm; endpoint cycle 128=2Ny.
- Outcomes are averaged analytically, not estimated from S trajectories.
  There is no schedule or temporal average and no sampling error bar.
- Last ten cycle-boundary covariance increments are below 1.1e-16 in
  normalized Frobenius norm. This is a global convergence diagnostic, not
  a certified relative-error bound for extremely small matrix elements.
- Figure 12 instead uses Ny=60 in (a,b), with S=100 pure half-filled
  trajectories and Born-conditioned frozen exteriors. Therefore a direct
  numerical difference cannot be attributed to averaging order alone.

## Figure and interpretation

`plot_mean_channel_squared_correlator.py` creates the files under
`analysis_outputs/raster_y_mean_channel_squared_correlator_v1/`:

- (a) full-x average for alpha_1=1 and 3;
- (b) alpha_1=1 at x=5,6,10,14,15;
- (c) the same spatial cuts for alpha_1=3.

All horizontal coordinates are natural log chord distance,
`log[(64/pi) sin(pi r/64)]`, for 1<=r<=32. All vertical coordinates are
natural log squared correlators. Symbols are computed values; joining
lines are guides. There are no fits or claimed decay exponents.
Panel (c) is not a size collapse: only Ny=64 exists in this matched
raster-y comparison. No additional evolution was launched.

The main view retains Figure 12's 1e-8 **display-only** cutoff. Only
r=1,...,4 survive it in these curves. At r=5 the full-x averages are
7.7897e-10 (alpha_1=1) and 7.4317e-10 (alpha_1=3). At r=1 they are
0.022570 and 0.0035637, respectively. Thus the most visible alpha contrast
is short-range; both mean-state correlators fall rapidly. This is not a
failure to reproduce the trajectory-wise observable: estimator and
preparation are different. The data do not by themselves establish an
asymptotic decay law.

The `_uncut` figure displays every positive raw value, including very tiny
tails. It is a diagnostic, **not** a certification of their relative
precision or evidence for a universal tail. No small value is clipped or
replaced in the saved NPZ/CSV; r=0 is saved but excluded from the figures.

Result hashes, embedded configurations, saved schedule words, Hermiticity,
and stationarity are checked on load. Unit tests compare the estimator
with explicit index sums and the legacy occupied-frame observer, and check
its normalization, translation behavior, and averaging-order distinction.
