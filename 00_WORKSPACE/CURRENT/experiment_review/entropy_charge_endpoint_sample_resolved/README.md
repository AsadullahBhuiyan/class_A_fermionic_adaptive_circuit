# Ensemble-mean endpoint entropy and charge

This directory analyzes the independent hard-wall endpoint campaign
`hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint`.
It is deliberately not pooled with the older time-dependent entropy campaign.
The input consists of 100 independent trajectories at each
`Ny=30,35,40,45,50,55,60`, evaluated at `t=2Ny`.

## Authoritative estimator

For each trajectory, periodic strip origins are averaged first.  At each
fixed `Ny`, the 100 resulting trajectory curves are then averaged and the
ensemble-mean curve is fitted on `Ay=8,...,floor(Ny/2)`:

```text
c1 = 3 m1,  c2 = 4 m2,  c3 = (9/2) m3,  k = pi^2 mF.
```

Uncertainties are ordinary trajectory sampling SEMs.  If `a` is the linear
OLS slope projection and `Sigma_y` is the sample covariance of the full strip
curve across trajectories, the reported variance is

```text
Var(m) = a.T @ (Sigma_y / 100) @ a.
```

Thus correlations between different strip widths are retained.  No bootstrap
and no regression-residual error are used.  Since all trajectories at a fixed
size share the same unweighted fit grid, the central slope from fitting the
mean curve equals the mean of the individually fitted slopes.  The code checks
this identity, but the mean-first construction is the authoritative estimator.

The entropy/charge ratios are formed from coefficients fitted to the two
ensemble-mean curves.  Their SEMs use the joint entropy--charge covariance and
the delta method.

## Canonical figures

The title-free, single-column two-panel combination of von Neumann entropy
and intrinsic charge variance is `figures/endpoint_entropy_charge_mean_curve_collapse_2x1`
(PDF, 300-dpi PNG, provenance JSON, and a companion LaTeX figure snippet).
Regenerate it with `python make_entropy_charge_collapse_2x1.py` from this directory.
It uses the same verified data, anchored mean fits, covariance SEMs, and size
encodings as the separate collapse figures, at 3.375 by 4.6 inches.

Run from the repository root with:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/entropy_charge_endpoint_sample_resolved/make_contour_scaling_figures.py
```

This revalidates all 140 result/completion pairs and exact sample IDs before
writing:

- `endpoint_entropy_mean_curve_size_collapse_3x1`: ensemble-mean endpoint
  entropy collapses for `q=1,2,3`.  Each trajectory is first anchored by
  subtracting its measured half-strip endpoint
  `Ay*=floor(Ny/2)`, after which the anchored curves are averaged.  One
  equal-size-weighted, through-origin slope is fitted across all seven sizes.
- `endpoint_charge_variance_mean_curve_size_collapse_1x1`: the identical
  mean-first construction for intrinsic subsystem charge variance.
- `endpoint_mean_curve_cq_k_scaling_1x1`: coefficients obtained by fitting the
  unanchored ensemble-mean curve separately at each size.  Bars are propagated
  trajectory SEMs; dashed segments are only guides between adjacent sizes.
- `endpoint_entropy_charge_contours_1x2`: cellwise ensemble means of the 100
  independently sampled spatial contours for the fixed half-strip `y0=0`,
  `Ay=Ny/2` at `Ny=60`.  Each panel uses an independent square-root `Blues`
  normalization from zero to its maximum, keeping zero white while sharpening
  weak and intermediate intensity.  Domain-wall overlays are omitted and the
  underlying trajectory-resolved arrays remain unchanged.

The fit panels follow the legacy evidence atlas: open unconnected empirical
markers, a gray band marking the declared fit window, and a black dashed fit
extended over the displayed log-chord domain.  The anchored curves use

```text
Delta X = log[sin(pi Ay/Ny) / sin(pi Ay*/Ny)]
Delta Sq = Sq(Ay) - Sq(Ay*)
Delta F_A = F_A(Ay) - F_A(Ay*)
```

so both even and odd circumferences end exactly at the origin.  Each size has
equal total weight in the joint fit rather than weight proportional to its
number of strip widths.  Plotted curve bars are pointwise trajectory SEMs and
may be smaller than their markers.

The following CSVs are the canonical numerical exports:

- `data/ensemble_mean_endpoint_curves.csv`
- `data/ensemble_mean_curve_fits.csv`
- `data/ensemble_mean_anchored_collapse_fits.csv`
- `data/ensemble_mean_entropy_charge_ratios.csv`

The earlier representative-collapse and sample-wise-summary derivatives are
preserved for provenance, but are superseded and excluded from the current
figure manifest and technical report.

We denote intrinsic subsystem charge variance by `F_A` and its spatial contour
by `f_A(r)`.  The subscript labels the subsystem; `q` is reserved for the
R\'enyi index.  Existing raw array names remain unchanged for compatibility.

The older exploratory sample-resolved tables can still be regenerated with:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/entropy_charge_endpoint_sample_resolved/analyze_endpoint_sample_resolved.py
```

They are not the estimator used in Figs. 8--10.
