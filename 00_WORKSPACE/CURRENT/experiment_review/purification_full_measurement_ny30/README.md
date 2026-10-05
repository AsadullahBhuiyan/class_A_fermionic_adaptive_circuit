# Full-system-measurement purification at Ny=30

This figure recreates the four-panel purification figure without overwriting
the old assets or changing the technical report. Panels A/B/D use campaign 21
(alpha1=1) and campaign 22 (alpha1=3 clipped continuation), Nx=20, Ny=30,
100 trajectories, endpoint T=60. Panel C deliberately retains the existing
campaign-13 slab-only size sweep and its T=2Ny fit; protocols are not pooled.

- A: sample-mean total von Neumann entropy divided by Ny, alpha1=1 versus 3.
  The four-panel figure has no entropy-fit overlay. The separate figure
  `figures/purification_ny30_alpha1_entropy_power_law.pdf` (and PNG) shows
  only the alpha1=1 mean and a gray dashed power-law fit A(t/Ny)^(-p), using
  unweighted OLS in log space over cycles 15..60 (Ny/2 through 2Ny). Earlier
  nonzero cycles remain visible but are not fitted. Average first,
  then take the log and fit. Its sampling SEM uses delta-method propagation
  of the full within-trajectory temporal covariance (not fit residuals).
  `entropy_power_law_fits.csv` also reports starts 1,3,5,10,20,30 at fixed
  endpoint 60. These are descriptive finite-window fits: the curvature and
  window dependence must not be confused with the smaller sampling SEM.
  This title-free diagnostic is 3.375 by 2.5 inches; its caption is
  `entropy_power_law_caption.tex`. Cycle time remains t, distinct from entropy S.
  Use `python make_figure.py --entropy-fit-only` to update only this diagnostic;
  the four-panel PDF and PNG are protected by before/after checksum checks.
- B: sample-mean entropy contour summed along y at x=5,15,2,18, divided by Ny.
  The x=2 and x=18 trivial-slab controls are separate, not averaged together.
- C: unchanged old seven-size gap values, SEMs, and power-law fit.
- D: sample-mean density of the unique minimum-absolute-rate eigenmode at T=60.
  Each trajectory supplies one normalized mode; orbital probabilities are summed.

All uncertainties are ordinary sample SEM (100 trajectories), never bootstrap.
Panels A and B use log-log axes; cycle zero remains in the CSVs but is not plotted.
The stored entropy estimator clamps occupations to [1e-12,1-1e-12], so the
near-zero tails should be interpreted as numerical floors. Campaign 22 retains
unclipped observations through cycle 30 and uses stabilized dynamics thereafter.

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python make_figure.py`.
The script verifies input completion checksums, source/configuration identity,
sample and cycle coverage, contour closure, slow-mode normalization and selection.
Outputs include a single-column 3.375 by 6.8 inch PDF, 300-dpi PNG, CSV curves,
sample-resolved slow-mode densities, and a hash-bound analysis manifest.
`caption.tex` supplies the LaTeX placement and mixed-protocol caption.

`unequal_time_gaussian.tex` (and its compiled PDF) derives Gaussian two-time
Wick formulas, the adaptive-circuit operator-insertion definition, and exact
Gaussian filtering/phase-probe identities for real and imaginary density
correlations. `test_unequal_time_gaussian.py` verifies the identities against
three-mode Fock-space calculations and checks a dephasing counterexample to
naively multiplying ensemble-averaged propagators.
