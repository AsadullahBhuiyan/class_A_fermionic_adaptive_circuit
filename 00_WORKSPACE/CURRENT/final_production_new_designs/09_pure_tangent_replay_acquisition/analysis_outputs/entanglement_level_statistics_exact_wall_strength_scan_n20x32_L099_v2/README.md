# Exact-wall adjacent-gap ratio comparison

Nx20 Ny32, Ay16, fixed y0=0, |lambda|<=0.99. IID Gaussian chemical potential
only at x=5,15; independent along y and between orbitals. Variances
1,2,3,4,6,9,12,16,25, 100 realizations each. Pure nonequilibrium reference
is unchanged. Equilibrium states use globally fixed half filling.

The acquisition belongs to the equilibrium exact-wall central-charge campaign.
Run build_notebook.py after Ny032 completes. It verifies every saved sample,
forms spacings and ratios inside each realization, then pools ratios for a
unit-area density histogram. The grid highlights just the two wall columns.
Errors on the reported pooled mean use a delete-realization jackknife.

Saved outputs: executed notebook, provenance, raw ratios, histogram counts and
densities, sensitivity tables, diagnostics, PDF and 300-dpi PNG.
