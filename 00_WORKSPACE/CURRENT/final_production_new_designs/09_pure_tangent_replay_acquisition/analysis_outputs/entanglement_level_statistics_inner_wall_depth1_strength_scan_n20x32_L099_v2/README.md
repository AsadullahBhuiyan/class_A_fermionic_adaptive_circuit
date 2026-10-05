# Inward-extended wall disorder: strength scan and distribution comparison

Nx20 Ny32, Ay16, origin y0=0, |lambda|<=0.99. IID zero-mean diagonal
Gaussian disorder only at x=5,6,14,15, independently on the two orbitals.
100 half-filled ground states per variance 1,2,3,4,6,9,12,16,25; compared to 100
pure nonequilibrium endpoints. No reflattening after adding the potential.
The original weak wall-disorder experiment shared potentials across the two
orbitals; this scan preserves the independent-orbital convention of the
subsequent full-diagonal and wall-band experiments.

Adjacent-gap ratios are calculated per realization, then pooled.
Mean errors: one-SE whole-realization jackknife. CDF area (Wasserstein)
and maximum (Kolmogorov) distances use unbinned ratios. Distance intervals:
1000 whole-realization bootstrap replicates, 95% percentile intervals.
No independent-level p-values, parametric shape fit, or equivalence claim.
The closest tested strength is selected on these same data.

Saved: executed notebook, source spectra provenance, raw ratios, histograms,
spacing distributions, window sensitivity, CDF curves, distance table,
bootstrap replicates, figures in PDF and 300-dpi PNG, and checksums.
Reproduce with build_notebook.py; acquisition belongs to the equilibrium bundle.
