# Origin-averaged central spectral weight, 20 by 32

All 100 hard-wall alpha1=1 pure endpoints at cycle 64, subsystem widths 1–16, all 32 periodic origins, L=0.5. All x columns and both orbitals are retained. No simulation or origin subsampling is used.

At each origin, normalize the pooled trajectory mixed-mode density after excluding |lambda| >= 1 - 1e-8. Then average the 32 densities equally. Its central-window integral is the mean over origins of sum_s n(s,origin) / sum_s m(s,origin). The raw count averages all origins inside each trajectory, then averages the 100 independent trajectories. Error bars retain the correlations among cuts and widths; origins are not additional independent samples. CSV columns also provide all-mode normalization and the alternative orders of conditional normalization.

At half width, the normalized integral is 0.05609329 +/- 0.00012049 (one SEM), the raw mean count is 6.453750 +/- 0.013979, and the all-mode normalized integral is 0.01008398.

The descriptive fit n = a + b log[(32/pi) sin(pi Ay/32)], over widths 4–16, gives a=5.910840, b=0.235704 +/- 0.008568, R-squared=0.992932. Changing the lower width to 2 gives b=0.241131 +/- 0.004986; to 8 gives b=0.226408 +/- 0.028779. These finite-size fits do not establish an asymptotic law. A normalized fraction has an additional width-dependent denominator and need not have the same scaling as a count.

Validation: source receipts, configuration, bytes, hashes, exact sample IDs 0–99; finite matrices, frame orthonormality, Hermiticity before symmetrizing, spectral bounds and trace sums. Origin-zero spectra match the preceding calculation within 1.74e-14; half-width complementary cuts give identical central counts. Unmodified spectra are cached, allowing L to change without new diagonalizations. Fixed-origin products remain in the adjacent `centered_spectral_window_vs_subsystem_n20x32_hard_alpha1_v1` directory.

## Files

- `spectral_window_vs_subsystem.ipynb`: editable CPU notebook; plotting commands immediately above figures.
- `subsystem_spectra.npz`: eigenvalues indexed by width, with axes trajectory, origin, mode; sample IDs, origins and widths included.
- `window_integrals.csv`: central spectral weights, mean counts, SEMs and normalization alternatives.
- `sample_window_counts.csv`: all sample/width/origin counts.
- `window_statistics.npz`: count arrays and full covariance matrices across widths.
- `spectra_diagnostics.json`, `window_diagnostics.json`, `cross_checks.json`: provenance, numerical and statistical checks.
- `spectral_window_vs_log_chord.pdf/png`, `spectral_window_vs_width.pdf/png`: vector and 300-dpi figures.

Figure caption: S=100 independent trajectories initialized as pure states with the exterior prepared as a product frame; hard-wall production endpoints at cycle 64. Nx=20, Ny=32, alpha1=1, alpha2=30, nshell=1. Each periodic subsystem contains all x and both orbitals, with Ay=1–16. Centered covariance eigenvalues are computed separately for each trajectory and origin. At each origin the mixed-mode density is normalized after pooling trajectories, then averaged equally over origins. Raw counts average origins within trajectories before ensemble averaging. Error bars are one trajectory SEM (delta method for normalized ratios). Fits use widths 4–16; full cross-width trajectory covariance is propagated to coefficient uncertainties.
