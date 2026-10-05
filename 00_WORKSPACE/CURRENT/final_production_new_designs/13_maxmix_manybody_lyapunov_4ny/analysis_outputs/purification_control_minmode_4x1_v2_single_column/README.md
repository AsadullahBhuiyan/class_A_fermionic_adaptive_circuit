# Purification figure with alpha3 control and the minimum-|lambda| mode

The PDF/PNG remains 3.375 x 6.8 inches. Old figure revisions and all raw data
are untouched. Reproduce with `plot_purification_control_minmode.py` in bundle 13.

- Panel a: Ny=40 only, independent alpha1=1 (bundle 07) and alpha1=3 (bundle 20)
  Born ensembles, 100 trajectories each, arithmetic means and sample SEM.
  Entropy is divided by Ny=40 and cycles by Ny=40. The alpha3 observer clips
  occupations at epsilon=1e-12 only when evaluating entropy, producing a
  near-zero numerical floor. This tail does not establish physical residual
  mixedness. Both ensembles use the same entropy estimator.
- Panels b,c: identical tables/fit to the previous figure. Panel b is alpha1=1,
  bundle 07; panel c uses bundle 13 endpoint gaps. No refitting or pooling.
- Panel d: Ny=30, alpha1=1, bundle 07 endpoints at T=120. For each sample xi,
  choose j*=argmin_j |lambda_(xi,j)| using
  lambda=[log(1-nu)-log(nu)]/(2T), then form
  p_xi(x,y)=sum_mu |u_(xi,j*)(x,y,mu)|^2 and average these 100 densities equally.
  It is NOT the average over all finite modes and NOT a mode of the averaged
  covariance. All 100 minima are unique and individually separated according
  to the saved extraction diagnostics. No degenerate-sector convention was
  needed. The selected signed rates include 56 negative and 44 positive
  values; selecting the most negative rate would be a different observable.

The extraction uses finite occupations 1e-9 < nu < 1-1e-9; the argmin is
checked against the entire saved active spectrum. Each selected density sums
to one. Orbitals are summed, not averaged; hard-wall exterior cells are zero.
The heatmap has a linear color scale using this new density's own maximum.
Per-sample selected indices, occupations, rates and eigen-residuals are saved
in selected_minimum_modes.csv. The NPZ retains all 100 densities and per-pixel
sample SEM, although no uncertainty overlay is shown in the heatmap.

All new raw inputs and extraction files are checksum-verified against their
historical manifests. analysis_summary.json pins source-table and input hashes.
See figure_snippet.tex for the complete caption; use graphicx and amsmath.
revtex_layout_check.pdf demonstrates the single-column placement with caption.
