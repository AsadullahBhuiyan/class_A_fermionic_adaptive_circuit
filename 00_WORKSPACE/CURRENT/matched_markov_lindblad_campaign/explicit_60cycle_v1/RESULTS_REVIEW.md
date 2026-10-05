# Review of completed fixed-60-cycle sweep

Run: `results/20260929T193023Z/`. All 16 cases and the analysis manifest
were checksum-verified during the square-sweep relaunch review. Both
`relaxation_vs_spectral.png` and `full60_increment_diagnostic.png` were
visually inspected; the raw NPZ increments and local ratios were checked.

At fixed Nx=20 and Ny=20,...,50, the normalized relaxation curves nearly
collapse by alpha. The RMS successive covariance increment falls below
1e-13 at cycles 15--16 for alpha_1=1, and cycle 11 for alpha_1=3. All
60 cycles were run, without early stopping. By cycle 20 the remaining
increments are at roundoff-scale plateaus, approximately 1e-16 and
1.5e-17 respectively. Thus 60 cycles are ample to achieve numerical
cycle-boundary stationarity for this maximally mixed initial state and
these finite geometries; the plateau is not a physical closing gap.

The finite-time decay need not reproduce the asymptotic spectral rate
exactly before roundoff. For example, at Ny=50 the local increment rate
at cycle 10 is 1.9478 for alpha_1=1 versus spectral Delta_C=1.8298, and
2.8852 for alpha_1=3 versus spectral Delta_C=3.1837. These are direct
consecutive-increment ratios, not fitted rates. Nonnormal transients and
initial-state mode overlap remain relevant. The spectral guide curves
are not pointwise bounds or fitted amplitudes. No claim is made that
the observed transient rates equal the exact slowest covariance rate.

This fixed-width sweep does not establish a nonzero thermodynamic or
many-body gap. The separately relaunched Nx=Ny sweep changes the slab
width and wall separation together and is the appropriate next comparison.
