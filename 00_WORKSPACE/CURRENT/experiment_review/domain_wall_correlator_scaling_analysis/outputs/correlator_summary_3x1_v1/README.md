# Single-column correlator summary

`hard_wall_correlator_summary_3x1.pdf` and the 300-dpi PNG are the main
3-by-1 figure, 3.375 inches wide. They are also generated into
`technical_report/figures/`. The report caption, definitions, and interpretation
live in `technical_report/correlator_endpoint_section.tex`.

- (a) Ny60 x-averaged arithmetic trajectory means, alpha_1=1 versus 3.
- (b) Ny60 alpha_1=1 x-resolved means at x=5,6,10,14,15.
- (c) Alpha_1=1 Ny24,28,32,40,50,60 collapse. Normalize each ensemble mean
  by its own antipodal value, then take the logarithm; normalize chord by its
  antipodal value as well. The free through-origin slope is -2.2290665923,
  using r=2..floor(Ny/4), with equal total regression weight per size.
  Dashed continuation beyond each size's fit window is extrapolation.
  The r=1 points are displayed but are outside the fit. No exponent is imposed.
  The light gray band is the union of size-specific fit ranges; the darker
  region is their intersection. Colored open symbols are fitted points;
  gray crosses are excluded. Neither shade represents uncertainty.

Each size has 100 independent trajectories, Nx=20, alpha_2=30, nshell=1,
hard/support-terminated walls, raster-y, pure half-filled initialization,
perfect correction, complex128, and endpoint 2Ny. Only independent separations
through Ny/2 are plotted. The 1e-8 display cutoff in (a,b) is applied after
averaging, not to individual samples, and is not a certified numerical floor.
No SEM, bootstrap, or uncertainty bands are plotted. The mean-curve regression
is not an average of trajectory-wise power-law exponents.

`boundary_bulk_addition_diagnostic.pdf/png` is a separate supporting figure:

- (a) Exact Nx-normalized contributions of B={5,6,14,15} and its complement U,
  compared with their sum (the full-x mean). U includes interior and exterior
  columns; it is not identified with a purely exponential sector. At r=5 and
  r=30, B accounts for 99.35% and 99.95% of the total, respectively.
- (b) The central column x=10 plotted against raw r on a semilog coordinate.
  Its dashed exponential fit uses only r=2..6 and gives squared-correlator
  decay length xi_C=0.65524 cells. The visibly different longer-distance
  behavior is retained; this does not establish a purely exponential tail.

Both are alpha_1=1, Ny60, S100 means. Connecting colored lines are not fits;
the dashed black segment in (b) is the fitted exponential. The diagnostic is
not a fourth panel of the technical-report figure.

## Fit-window check

`quarter_vs_half_fit_windows.pdf/png` shows both fits with their actual selected
points and gray ranges. `fit_window_sensitivity.pdf/png` and the matching CSV
scan lower endpoints 1,2,3,4,5,6,8 and upper endpoints floor(Ny/4), floor(Ny/3),
Ny/2. Every valid fit includes all six sizes; configurations with fewer than
four points at any size are flagged invalid without dropping sizes.

With lower endpoint 2, extending quarter to half changes beta from 2.229067
to 2.228248 (0.037%). The half-window values starting at 3,5,8 are 2.215968,
2.201613,2.190333. Including r=1 instead gives 2.663366: sensitivity is much
stronger to the short-distance cutoff. The quarter window is an inherited
conservative comparison, not required by the chord formula. The anchored
normalization already uses Ny/2 even when regression stops at Ny/4, and the
near-antipodal points have small through-origin regression leverage.
The JSON also records individual-size anchored and free-intercept comparisons;
these are different estimators, not independent ensembles to pool.

`compact_curves.npz` retains sample-resolved x-resolved arrays and unmasked
regional sums. `plotted_data.csv` retains unmasked primary/decomposition
series and fit/display selection flags. `summary.json` records all source
paths, result hashes, completion records, fit diagnostics, and window checks.
No raw data, previous figures, or unrelated report sections were changed.

Reproduce from the repo root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis/build_correlator_summary_3x1.py
python -m pytest -q tests/test_correlator_summary_3x1.py tests/test_large_ny_correlator_review.py
```
