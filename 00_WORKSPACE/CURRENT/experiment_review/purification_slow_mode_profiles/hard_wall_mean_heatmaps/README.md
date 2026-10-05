# Slot-07 hard-wall sample-averaged mode density

`slot07_hard_wall_sample_averaged_density.pdf` and `.png` show Nx=20,
Ny=20,30,40, with 100 independent trajectories per panel. For each sample xi,
average over all m_xi numerically resolved endpoint domain-wall modes:

\[
p_\xi(x,y)=\frac{1}{m_\xi}\sum_{j=1}^{m_\xi}\sum_\mu
|u_{\xi j}(x,y,\mu)|^2,\qquad
\overline p(x,y)=\frac1{100}\sum_{\xi=1}^{100}p_\xi(x,y).
\]

Every sample and ensemble density sums to one. This is a density average,
not an average of complex wavefunctions; samples with more resolved modes do
not receive extra weight. The resolved-subspace average is invariant under
unitary rotations within that subspace. No spatial localization threshold,
y translation, wall alignment, smoothing, or color renormalization is applied.

**Caption.** Sample-averaged endpoint probability density of resolved
purification modes, hard walls only, for (a) Ny=20, (b) Ny=30 and (c) Ny=40,
Nx=20, S=100 independent trajectories each. Maximally mixed initialization,
perfect correction, alpha1=1, alpha2=30, nshell=1, raster-y ordering,
no postselection, complex128, and T=4Ny cycles. Modes satisfy
1e-9 < nu < 1-1e-9 in the endpoint occupation spectrum; 2--4 modes per sample.
Average orbital-summed probabilities first over modes within each trajectory,
then over trajectories with equal sample weight. All 100 samples contribute
at each size. Both walls are at x=5 and x=15; the exterior is zero-filled from
the verified hard-wall active slab. Panels share one linear probability color
scale. No error overlay or fit is shown; per-pixel trajectory SEMs are saved
with the data. Smaller per-cell densities at larger Ny partly reflect the
normalization over a longer wall, not weaker localization.

The NPZs retain sample densities, means, SEMs, sample IDs and mode counts.
The manifest verifies all 300 source extraction hashes. Each density is
normalized and its x marginal is checked against the prior extraction.
No dynamics or eigendecomposition is repeated; original inputs are unchanged.

Run from the parent directory:

```bash
OPENBLAS_NUM_THREADS=2 python plot_hard_wall_mean_heatmaps.py
```
