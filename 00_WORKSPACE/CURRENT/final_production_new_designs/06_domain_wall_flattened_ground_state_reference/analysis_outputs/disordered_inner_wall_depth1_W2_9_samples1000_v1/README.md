# Equilibrium sample-count convergence: W^2=9, walls + one inward column

Nx20 Ny32, hard walls x5,15, disorder mask x5,6,14,15, independent Gaussian
potential for every selected orbital. Fixed 640 particles, Ay16, y0=0, L=.99.
1000 realizations; first 100 reproduce the preceding six-strength scan.
Seeds [2026092703,5,sample_id] preserve that prefix exactly.

Ratios are computed within each realization before pooling. Compare nested
prefixes 100,250,500,1000, a disjoint new-900 set, and ten independent 100-state
blocks. Mean uncertainties use a whole-realization jackknife. CDF distance
intervals resample whole realizations (500 replicates, 95% percentile).
No independent-level p-values or additional nonequilibrium simulations.

GUE density/CDF reference: folded Atas et al. 3x3 approximation (mean .60266).
The mean plot separately labels the large-matrix GUE value .5996.
Neither increasing sample count nor this finite geometry establishes a
thermodynamic/sector-resolved universality classification.

Reproduce acquisition with run_analysis.py; execute analysis via build_notebook.py.
Saved outputs include potentials, spectra, energies, sample-resolved ratios,
histograms, prefix/block statistics, bootstrap distances, PDF/PNG figures,
executed notebook, full input provenance, receipts and checksums.
