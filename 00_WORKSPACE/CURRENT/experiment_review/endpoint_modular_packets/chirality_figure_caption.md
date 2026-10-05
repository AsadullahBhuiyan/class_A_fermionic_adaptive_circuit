# Modular-handedness figure v2

## Main: conditional propagation and control comparison

Hard-wall endpoints at Nx=20, Ny=32, nshell=1, alpha2=30, with 100 independent
trajectories for each alpha1=1,3. Random pure initial states with Born-conditioned
exterior product preparation undergo 64 raster-y physical cycles with perfect
correction and complex128 arithmetic. The saved occupied frames define the
restricted occupation matrices on all x and 16 consecutive periodic y rows.
Two independent charge-2 packets occupy both local orbitals at (5,8) and (15,8)
in translated subsystem coordinates. Modular evolution is the exact exponential
of -2 atanh(2 C_A - I_A), clipping centered eigenvalues to
[-1+epsilon,1-epsilon], epsilon=1e-10. No circuit simulation is rerun.

(a,b) Left/right wall longitudinal conditional probabilities for alpha1=1.
For each trajectory/cut separately, sum density over columns within periodic
x distance 2 of the source wall, then normalize by instantaneous retained
charge. Average over 32 translated cuts within each trajectory, then trajectories.
Maps share a linear 0..1 color scale. Vertical coordinate y_rel-8 is position
relative to injection, not the cut origin. Dashed guides mark zero. Colored
overlays are mean displacements, equal to map first moments to numerical precision.

(c) Mean wall-window displacement relative to injection for both ensembles;
orange/blue denote alpha1=1, gray/black the alpha1=3 control. Solid/dashed identify
x=5,15. (d) Wall contrast D_chi=(Delta y_x5-Delta y_x15)/2, formed within each
trajectory before computing mean and uncertainty. Shading is +/- one SEM over
100 independent trajectories, retaining correlation between walls within a
trajectory. Different alpha ensembles are not paired. The contrast is not a
topological invariant. Times span 0..1 at spacing 0.01, without smoothing,
unwrapping, or a velocity fit.

## Companion: five spatial snapshots

Top row: alpha1=3 control. Bottom: alpha1=1. Columns: times 0,0.1,0.2,0.5,1.
Panels show two independently evolved packets, averaged over cuts within each
trajectory and then trajectories. Circle AREA is linear in mean density with
the same scale throughout; cells below 1e-4 are omitted only from rendering.
Vertical coordinate y_rel is position within the subsystem. Dashed guides and
open source rings identify injection at y_rel=8; black ticks give averaged
wall-window centers using the main-figure estimator, not the COM of pooled density.

## Supporting numerical comparisons

The cutoff comparison reuses saved epsilon=1e-8,1e-10,1e-12 displacement results
on the same source ensembles. The control axis is explicitly labeled as an
expanded scale. Opposite signs persist, but magnitude and oscillation period
depend on cutoff. The control is weakly mobile, not exactly stationary. No
regularization-independent velocity or proof of chirality from snapshots alone
is claimed.

Retention curves show instantaneous wall-window charge / initial charge 2,
averaged over cuts then trajectories, with +/- trajectory SEM. This exposes
leakage instead of hiding it behind conditional normalization. All reported
times are modular times, not physical circuit times.
