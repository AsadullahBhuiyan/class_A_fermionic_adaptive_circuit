# Ascending hard-wall endpoint occupation spectra

Slot07, Nx=20, Ny=20,30,40, hard walls, alpha1=1, alpha2=30, nshell=1,
maximally mixed initialization, perfect correction, raster-y, no postselection,
complex128, T=4Ny. All 100 independent trajectories per size are plotted.

Top panels: each trajectory's complete active-slab occupation spectrum in
ascending order, with 22Ny eigenvalues. Pure exterior modes are excluded.
Each translucent curve is one sample, not the spectrum of an averaged
covariance and not an average over sorted ranks. Changes in rank/charge
between samples must not be mistaken for extra mixed modes within a sample.

Bottom panels: only that sample's eigenvalues satisfying 1e-9 < nu < 1-1e-9,
again sorted ascending. The logit vertical scale resolves proximity to both
0 and 1. No clipping manufactures finite eigenvalues. Horizontal index is
the ordinal within the unsaturated subset, not the full-spectrum index.
Lines guide the eye within a single trajectory; no fits or error bars are used.
All raw spectra, sample IDs and unsaturated counts are saved in three NPZs.

Count distributions (number of trajectories with 2, 3, 4 unsaturated modes):
Ny20: (15,40,45), mean3.30; Ny30: (12,49,39), mean3.27;
Ny40: (12,44,44), mean3.32. These are numerical counts at the stated tolerance.

This is an occupation spectrum, not a tangent-cocycle singular spectrum.
The latter does not imply an occupation matrix without an additional physical
construction. All 300 extraction hashes, ascending order, IDs and finite
masks are verified. Source data are unchanged.

Reproduce from the parent directory: `python plot_ordered_occupations.py`.
