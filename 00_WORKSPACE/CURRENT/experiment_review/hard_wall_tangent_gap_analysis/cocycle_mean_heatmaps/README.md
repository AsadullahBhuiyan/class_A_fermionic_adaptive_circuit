# Sample-averaged cocycle singular-mode densities

Completed for 200 saved slot17 full chronological cocycles: Nx=20, Ny=40,
hard walls, alpha1=1 and 3, alpha2=30, nshell=1, 100 trajectories each,
pure random half-filled initialization, perfect correction, raster-y ordering,
no postselection, complex128, T=2Ny=80, no burn-in. The smaller completed
slot17 sizes saved gaps only, not matrices or eigenvectors. This figure does
not silently pool the different, partially completed slot09 CPU replay dataset.

## Operators and averaging

The preserved matrix convention is

\[
K=e^\ell U\operatorname{diag}(s)V^\dagger,
\qquad H_T=K^\dagger H_0 K.
\]

Thus U diagonalizes KK^dagger and lives on INITIAL coordinates, while V
diagonalizes K^dagger K and lives on ENDPOINT coordinates. Both operators have
the same nonzero eigenvalue logs 2*(log(s)+ell), but their eigenvectors and
spatial profiles differ. The 880 active input rows are embedded using the
saved `active_input_indices` into the 1,600-coordinate physical lattice.
Endpoint vectors already use all 1,600 coordinates. Exterior zero-padding is
applied to U only; V is not projected, truncated, or renormalized to the slab.

For each trajectory average the orbital-summed normalized densities over the
selected vectors, then average equally over the 100 trajectories. Every
sample density and displayed mean integrates to one. This is NOT an average
of complex vectors, matrices, or generators before diagonalization. All-mode
means are normalized diagonals of spectral-subspace projectors and do not
depend on the eigenvector basis within included degenerate subspaces.

The main figure includes all numerically resolved singular modes according
to the preexisting extraction threshold eps_float64*max(K.shape)*s_max.
There are 14--20 per sample at alpha1=1 (mean 16.97), and 696--715 at alpha1=3
(mean 707.63). Null vectors are excluded, not interpreted as finite modes.
The companion selects the four one-leg rates (log(s)+ell)/T closest to zero,
including any boundary ties within 1e-10; all current selections have four
modes. This is a declared diagnostic window, not a physical tangent-gap filter.

## Findings and crucial limitation

For alpha1=1, all-mode endpoint densities put 95.71% +/- 0.24 percentage
points near the walls (x=4,5,6,14,15,16); the four closest-rate mode densities
give 96.39% +/- 0.26 percentage points. Uncertainties are trajectory SEM.
Initial-direction densities are spread through the slab, with wall-window
weights 37.43% and 36.76%, respectively (a uniform slab has 4/11=36.36%).

For alpha1=3, initial directions are also spread through the slab, while the
resolved full-matrix endpoint directions are concentrated in the exterior.
This is the result for the persisted unrestricted one-leg product, not a
claim that the physical hard-wall tangent pair modes occupy the exterior.
The spatial plot alone does not identify the origin or physical meaning of
that sector. The saved Ny40 matrices lack the initial occupied frame needed
to reconstruct the physical occupied--empty restriction. As documented in
`../ENDPOINT_SINGULAR_MODES.md`, these modes must NOT be identified with the
occupied--empty pair modes underlying the published tangent gaps. Missing
basis labels are not reconstructed by matching rates or fabricating frames.
No physical failure, new topological classification, or chirality conclusion
is inferred from this unrestricted-matrix comparison.

These tangent products use pure initialization and T=2Ny; slot07 purification
uses maximally mixed initialization and T=4Ny. Similar wall density does not
make their spectra or eigenmodes identical.

## Figures and caption

`cocycle_mean_density_all.pdf` and `.png`: sample-averaged probability density
of every numerically resolved full one-leg cocycle singular direction. Top
row: KK^dagger initial eigenvectors; bottom row: K^dagger K endpoint
eigenvectors. Columns: alpha1=1 and 3. Hard walls, Nx=20, Ny=40, S=100
independent trajectories per column, pure initialization, T=80, scientific
settings above. Average modes within samples, then samples with equal weights.
Common linear probability scale; no smoothing, wall alignment, fitting,
or error overlay. Physical walls lie at x=5 and 15. Initial rows and endpoint
rows are distinct operators even though their nonzero eigenvalues agree.

`cocycle_mean_density_nearest4.pdf` and `.png`: same layout and estimator,
but restricted to the four resolved one-leg rates closest to zero per sample.
This is not a reconstruction of the previously computed slow tangent pair modes.

Eight NPZs save means, per-pixel trajectory SEMs, all sample densities, mode
counts and sample IDs. `manifest.json` records SHA-256-verified input
extractions, output hashes, per-sample wall/exterior weights and summary data.
All 200 input checksums pass; probability normalization and agreement with
the previously extracted x profiles are checked for every selected subspace.
Tests check coordinate indexing, subspace basis invariance, equal-sample
weighting, tied rates and numerical-null exclusion. No simulation or new SVD
was run, and existing data remain unchanged.

Reproduce from the parent directory:

```bash
OPENBLAS_NUM_THREADS=2 python -m unittest -v test_cocycle_mean_heatmaps.py
OPENBLAS_NUM_THREADS=2 python plot_cocycle_mean_heatmaps.py
```
