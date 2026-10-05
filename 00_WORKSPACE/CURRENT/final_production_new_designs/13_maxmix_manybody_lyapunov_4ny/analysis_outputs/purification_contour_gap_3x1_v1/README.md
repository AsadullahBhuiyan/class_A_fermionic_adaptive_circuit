# Hard-wall purification, contours, and Lyapunov gaps

Reproduce from bundle 13 with `python plot_purification_gap_summary.py`.
The script reuses the existing bundle-13 and bundle-07 data validators and
writes PDF, 300-dpi PNG, sample/summary CSV tables, and JSON provenance here.

## Figure caption

Hard-wall maximally mixed purification at Nx=20, alpha1=1, alpha2=30,
nshell=1, with raster-y measurements, perfect correction, and complex128
dynamics. The active slab is maximally mixed after exterior product-state
preparation; walls are at x=5,15. Every curve uses S=100 independent Born
trajectories per circumference and total depth T=4Ny.
(a) Trajectory-averaged total entropy per Ny versus cycle/Ny, for
Ny=20,24,30,36,40,44,56,60. Bundle 13 supplies all sizes except Ny=40,
which comes from the independent bundle-07 hard-wall ensemble (marked 07).
Bundle 13 saves spectra/entropy every four cycles and at the registered
Ny multiples; bundle 07 saves entropy every cycle. Only saved times are
used, connected with lines. Overlapping ensembles are not pooled.
(b) Trajectory-averaged x-resolved contour contributions per Ny at the
two walls x=5,15 and the slab center x=10, for Ny=20,30,40 in bundle 07.
The entropy contour is first evaluated per trajectory and summed over y,
then averaged over trajectories. Shading in (a,b) is one sample SEM.
(c) Occupation-derived finite-time single-particle Lyapunov half-gap in
bundle 13, Delta(t)=min_j |log(nu_j(t))-log(1-nu_j(t))|/(2t).
The minimum is taken independently within each trajectory and checkpoint;
exact-cap modes remain at infinite magnitude and cannot lower a finite gap.
The blue curve uses t=T. The green and red curves average the two saved
endpoint gaps within [T-4,T] and [T-8,T-4], respectively, inside each
trajectory, before averaging over trajectories. These are two-point
approximations to segment averages, not measurements at every intervening
cycle or slopes of a four-cycle propagator. Each time uses its own 2t
normalization. The three curves share trajectories/checkpoints and therefore
are correlated comparisons, not independent ensembles.
Error bars are sample SEM, std(ddof=1)/sqrt(100), computed from the
trajectory-level endpoint or segment values. No bootstrap, exponential fit,
power-law fit, or inset is used in this figure.

The vertical axes of (a,b) are logarithmic; both normalized-time axes are
linear. Panel (c) has linear axes. Curves in the final two four-cycle
segments concern a short late-time interval and are not a test of
convergence across entire Ny-long windows.
