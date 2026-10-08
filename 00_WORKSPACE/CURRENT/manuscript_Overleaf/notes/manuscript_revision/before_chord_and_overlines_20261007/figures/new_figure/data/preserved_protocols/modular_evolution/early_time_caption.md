Early-time modular propagation from the same hard-wall Nx=20, Ny=32 pure-state
endpoints as the previous figure: S=100 independent trajectories for each
alpha1=1,3; alpha2=30, nshell=1, random pure initialization with Born-conditioned
exterior preparation, perfect correction, raster-y order, and endpoint cycle64.
Each of the 32 translated half-cylinder cuts retains all x and 16 y rows.
Packets contain two incoherent local orbital occupations and begin at
(x,y-y0)=(5,8),(15,8). The reduced centered covariance is diagonalized separately
for every trajectory and cut, and h_A=-2 atanh(G_A) generates unitary modular
evolution without division by circuit depth. The main figure uses the original
covariance-eigenvalue clipping threshold epsilon=1e-10.

(a,b) Mean evolved density at t_mod=0,0.1,0.2 (purple, orange, green), for
alpha1=3 (control) and alpha1=1 respectively. Density is averaged only after independent
evolution, first over cuts and then over trajectories. Marker area is
proportional to sqrt(mean density/2), identically normalized in both panels;
the rendering threshold 1e-4 does not enter numerical calculations. Dotted
vertical lines mark x=5,15.

(c) Alpha1=1 signed wall-window COM displacement over t_mod=0..1, spacing0.01. The
radius-2 periodic-x window is normalized independently before averaging.
Solid orange: x=5; dashed blue: x=15. Shading is one SEM across the 100
independent trajectory-level origin means. There is no smoothing, unwrapping,
wall-sign reorientation, Hamiltonian pre-averaging, or velocity fit.

The current stacked figure has three rows and one column, no titles, and
in-axes alpha1 labels. A light zero guide is retained in (c), with the legend
above it. The previous horizontal rendering is preserved separately.

The companion clipping-sensitivity figure uses the same trajectories and cuts
at epsilon=1e-8,1e-10,1e-12 for both alpha1 values. Solid/dashed lines distinguish
the two walls; colors distinguish thresholds. All curves include trajectory
SEM. Paired differences relative to epsilon=1e-10 and their SEM are saved in
displacement.csv. Spectral medians describe individual reduced generators,
not a generator constructed from an averaged covariance. A stable sign of
motion does not imply that its magnitude or oscillation frequency is independent
of regularization, nor does a short-time displacement alone prove chirality.
