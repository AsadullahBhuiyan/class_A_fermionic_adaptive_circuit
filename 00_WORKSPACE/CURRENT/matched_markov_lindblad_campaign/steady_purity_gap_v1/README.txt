Stationary occupation gaps matched to the fixed-width channel-gap sweep
====================================================================

User-approved grid: Nx=20; Ny=20,40,60,80,100; alpha_1=1 and 3 (10 cases).
All other physical parameters and engine/helper sources match the checksum-
verified fixed_width_alpha_spectral_v1/results/20261001T220546Z campaign.
The source comparison is required before each case. Previous outputs untouched.

Canonical CPU classA_U1FGTN.run_markov_channel, maximally mixed initialization,
hard-wall OW support truncation, inclusive walls x=5..15, all slabs active,
alpha_2=30, nshell=1, X trials, periodic x/y, zero twist, complex128,
perfect correction and measurement dephasing, raster-y Ap/Am/Bp/Bm order.
No random trajectories: this is exact outcome-averaged dynamics.

Run 60 cycles and one additional cycle to check the stationary endpoint.
Record charge and absolute Frobenius increments every cycle. Diagonalize the
occupation matrix at cycles 0,1,5,10,20,40,60,61. Save the two final covariance
snapshots (60 and 61), all observed occupation spectra, and compact diagnostics.
Both spectra below retain all slabs; no mode is removed near half filling.

1. Actual spectrum: combine complete eigenvalue spectra of the two EXACT hard-
   wall blocks after checking that the inter-block covariance vanishes.
2. Translation-twirled spectrum: take the momentum-diagonal blocks of F C F^dagger.
   This is exactly the y-translation twirl, not an assumption of translation
   invariance and not necessarily the fixed point of the ordered channel.

Purity gap = min_j |1-2n_j|; distance to half filling = purity gap/2.
At t=0 the purity gap vanishes trivially because the initial state is maximally
mixed. A finite-size small minimum does not by itself establish a thermodynamic
closing. The covariance gap g_C=1-rho(A)^2 is read from the matched spectral run.
These are two-point gaps, not a reconstruction of a non-Gaussian density matrix.

Stationarity acceptance requires the last five absolute Frobenius increments
below 1e-10 AND both endpoint purity-gap changes below 2e-10. Unconverged results
are saved explicitly as not_converged and excluded from the stationary plot.
This is numerical convergence evidence, not a rigorous distance-to-fixed-point
bound for an arbitrary nonnormal map. No positive gap is forced or fitted.

Execution: run_purity.py launch --cpus 9,10
One detached tmux queue, two single-core workers (one per alpha), one numerical
library thread each, nice=10, case-level logs and tqdm progress. CPU numbers must
be checked for availability before launch. Existing jobs remain untouched.
Results are atomic NPZ/completion-JSON pairs with source/configuration checksums.
Restart: run_purity.py launch --root ABSOLUTE_EXISTING_ROOT --cpus 9,10
Verified completed cases are skipped. Only the interrupted case is repeated.
Report/plot: run_purity.py report --root ABSOLUTE_EXISTING_ROOT
The queue automatically generates CSV, three-panel PDF/PNG, caption, and
manifest after the workers exit. Partial reports disclose the completed count.

Tests: tests/test_stationary_purity_gap.py compares small-system canonical
dynamics to independent dense projector sweeps, the block spectrum to a full
dense eigensolve, and the Fourier-diagonal estimator to the existing explicit
translation-twirl implementation; it verifies unchanged reference sources.
