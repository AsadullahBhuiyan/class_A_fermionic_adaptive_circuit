# Domain-wall correlator scaling analysis

## Matched alpha comparison

`plot_alpha_comparison.py` compares the completed `alpha_1=1` and `alpha_1=3`
ensembles at `Nx=20, Ny=28, nshell=1`, with identical hard-wall protocols,
100 trajectories each and endpoint cycle 56. Two side-by-side, shared-axis
panels show `log(mean C_G)` versus log chord at `x=5,6,14,15`. Gray marks
`r=2..7`; dashed unconstrained power-law fits are diagnostics and are not an
assertion that the alpha-3 curves follow a power law. Small positive values
are retained, without certifying a numerical-accuracy floor. No sample-index
pairing is assumed between the ensembles. Inputs are checksum-verified and
their recorded engine/observer source identities must match.

Run `python 00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis/plot_alpha_comparison.py`.
PDF, PNG, plotted CSV, compact sample-resolved endpoint arrays, fit/provenance
JSON and caption are saved under `outputs/alpha1_comparison_Ny028_nshell1/`.

## Earlier correlator figures

`plot_typical_hard_wall_x_slices.py` makes a trajectory-resolved spatial cut through
the completed hard-wall baseline at `Nx=20`, `Ny=32`, `alpha_1=1`,
`alpha_2=30`, and `nshell=1`.  The plotted state is the final observation at
cycle `2 Ny = 64`.

The representative trajectory is selected without inspecting any individual x slice.
For every one of the 100 trajectories, the script fits the archived x-averaged squared
correlator to `G(r_y) proportional to r_y^(-beta)` on the legacy declared window
`2 <= r_y <= Ny/4`.  It then selects the trajectory whose fitted beta is closest to
the ensemble median.  This makes "typical" deterministic and reproducible rather than
an aesthetic choice.

The five x slices are `x=2` (left trivial bulk), `x=4` (trivial-side neighbor),
`x=x_L=5` (left interface), `x=6` (topological-side neighbor), and `x=10`
(middle of the topological slab).  The split logarithmic y axis is intentional: the
two trivial-region curves are at the numerical floor and would otherwise be invisible.
The dashed `r_y^(-2)` line is a visual reference, not a fit constraint.

Run from the repository root with:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis/plot_typical_hard_wall_x_slices.py
```

The `outputs/` directory contains PDF and 300-dpi PNG figures, the plotted source data,
and a JSON record of the selection rule, scientific contract, input checksums, and
selected trajectory.

`plot_typical_hard_wall_topological_slab.py` uses the same trajectory-selection rule
but plots only columns inside the active topological slab.  It includes both interfaces,
their inward neighbors, and four evenly spaced interior columns:
`x = 5, 6, 7, 9, 11, 13, 14, 15`.  This view removes the projected exterior's numerical
floor and exposes the crossover from the algebraic wall correlations to the rapidly
decaying slab interior.  It displays the full periodic range `1 <= r_y <= 32`.
The simulation archives only the independent separations `0 <= r_y <= Ny/2`; the script
reconstructs `17 <= r_y <= 31` exactly from `G_x(r_y) = G_x(Ny-r_y)`, and `r_y=32`
is the periodic recurrence of the archived on-site `r_y=0` contact term.

`plot_hard_wall_flattened_benchmark.py` provides the deterministic static comparison at
the same `Nx=20`, `Ny=32`, `alpha_1=1`, and `alpha_2=30`.  It uses the repository's
hard/support-truncated OW construction (`DW=True`, `dw_truncation=True`), constructs the
Wannier functions separately at `nshell=1` and `nshell=None`, forms
`H_flat = sum_R(P_A+ + P_B+ - P_A- - P_B-)`, and fills the lowest half of its spectrum.
The plotted correlator is reduced from that exact projector with the identical
x-resolved estimator used for the stochastic data.  The occupied momentum grid carries
the B0 regulator `phi=1e-7`.  This selects one side of the finite-volume wall-mode
crossing; without it the nearly degenerate zero modes are selected by eigensolver
tie-breaking and the long-distance correlator is not reproducible.

`plot_typical_hard_wall_short_distance.py` is the recommended dynamical presentation.
It shows only `1 <= r_y <= 7` and masks values at or below `1e-8`, where the single-run
curves become dominated by finite-size floors and realization-specific residuals.  The
cutoff is purely visual: the CSV retains every raw value and marks whether it was shown.

`analyze_hard_wall_power_law_exponents.py` performs the corresponding ensemble
extraction on the completed `S=100` baseline.  Its primary estimator follows the
legacy correlator convention: for each trajectory separately, fit
`log G_xavg = log A - beta log(r_y)` on `2 <= r_y <= 8`, then take the equal-weight
mean of the 100 fitted exponents.  The reported spread is the empirical spread of
the trajectories themselves: sample standard deviation, median, interquartile range,
and central percentile ranges.  It deliberately does not report an SEM or a confidence
interval for the mean.  The script also reports
fits to the ensemble-mean curve, cylinder-chord fits, the two-wall average, and the
left and right walls separately.  The `r_y=1` point is excluded because the
short-range bulk contribution dominates it; the choice is recorded rather than
hidden.  The output JSON states the boundary-power-law plus periodic-bulk-exponential
mixture model used to interpret the x average.

`analyze_finite_size_power_law.py` performs the matched hard-wall finite-size
comparison at `Ny=24,28,30,32,40,50`, with 100 independent trajectories at each
circumference.  The `Ny=24,28,32` points come from the new frame-native,
x-resolved campaign; `Ny=30,40,50` come from the legacy streaming-covariance
campaign.  Both cohorts use `Nx=20`, `alpha_1=1`, `alpha_2=30`, `nshell=1`,
pure initialization, hard support truncation, raster-y ordering, perfect correction,
complex128, and the identical x-averaged squared-correlator normalization.  The
primary fit is trajectory-first on the cylinder chord distance over
`2 <= ry <= floor(Ny/4)`.  The single-column, vertically stacked figure shows
the raw chord-distance curves with the critical `d^(-2)` prediction and a
technical-note-style system-size transfer plot.  The size points are unconnected,
their bars show the trajectory IQR, and the compact legend distinguishes only the
`1/Ny` and `1/Ny^2` extrapolations.  It contains no SEM or confidence interval on
a mean.  A
manuscript-ready uncertainty and estimator description is stored in
`outputs/hard_wall_finite_size_power_law_caption.md`.

`analyze_wall_resolved_finite_size_power_law.py` repeats the same trajectory-first
chord fit directly at the two programmed hard-wall columns, `xL=5` and `xR=15`.
Only `Ny=24,28,32` are included because those completed frame-native archives retain
the full x-resolved tensor; the matched legacy `Ny=30,40,50` archives retain only the
x average.  The companion single-column figure shows the two-wall mean curves and the
left-wall, right-wall, and within-trajectory two-wall-average exponents.  Size points
are unconnected, and the bars show trajectory IQR rather than uncertainty on a mean.

`plot_x_resolved_log_chord.py` gives the direct straight-line view at the largest
completed x-resolved circumference, `Ny=32`.  It plots the trajectory-mean curves
at `xL=5`, `xL+1=6`, `xR-1=14`, and `xR=15` as `log C_G(x,ry)` versus
`log[(Ny/pi) sin(pi ry/Ny)]`, with `log` denoting the natural logarithm on both
axes. It shades
the declared `2 <= ry <= 8` fit window and
shows the critical slope `-2` without constraining any fit to that value.

## Half-system ground-state occupation spectra (80 x 80)

`plot_flattened_half_system_spectra.py` computes the deterministic half-filled
ground state of the same hard-wall OW parent, now at `Nx=Ny=80` and
`alpha_1=1,3`, with `alpha_2=30`, `nshell=1`, and trial orbitals `X`.
The canonical CPU constructor sets the inclusive central slab to `20 <= x <= 60`
and truncates OW support at its interfaces before normalizing each mode. The
central slab is only topological for the appropriate mass: the name of the slab
does not assert that `alpha_1=3` is topological. No adaptive trajectories are run.
Existing data, Colab deployments, and canonical engine files are unchanged.

The single-particle parent and ground-state projector are

\[
H_{\mathrm{flat}}=\sum_{\boldsymbol R,\nu=A,B}
\bigl(|W_{\boldsymbol R,\nu,+}\rangle\langle W_{\boldsymbol R,\nu,+}|
-|W_{\boldsymbol R,\nu,-}\rangle\langle W_{\boldsymbol R,\nu,-}|\bigr),
\qquad P_{\mathrm{occ}}=\sum_{j=1}^{N_xN_y}|u_j\rangle\langle u_j|.
\]

The occupied modes are the globally lowest half of the parent spectrum. The
calculation reuses the previous benchmark's `phi=1e-7` occupation-grid regulator
and periodic reconstruction convention, to resolve the finite-volume wall-mode
crossing. It is a regulated static ground-state reference, not a flux-threading
experiment. `H_flat` is the OW-parent construction above, not a newly imposed
sign function applied after the domain-wall/truncation construction.

Restrict `P_occ` to `A=[0,Nx) x [0,Ny//2)`, including both physical orbitals,
and diagonalize this restricted occupation matrix. Each alpha gives 6,400
eigenvalues `nu`. (The code calls the occupation projector a covariance; it is
not the signed engine covariance `2 P_occ - I`.) A companion plot converts
the same retained values to `epsilon=log[(1-nu)/nu]`.

Both displayed histograms discard `nu <= 1e-12` and `nu >= 1-1e-12`, following
the request to exclude roundoff-sensitive modes. Each alpha is normalized
**separately over its retained modes**, so each displayed density integrates
to one; discarded spectral weight is not displayed as endpoint spikes. This
is a conditional spectral density, not a density normalized to all 6,400 modes.
All raw eigenvalues, including discarded values, are preserved in the NPZ.
Both vertical axes are logarithmic. Empty bins stay empty, with no invented
positive floor. Fixed common bins have width `0.02` for occupation and `1`
for modular energy. A binned gap alone does not establish an exact spectral gap.

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -u \
  00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis/plot_flattened_half_system_spectra.py \
  --threads 8
```

Use `--replot` to regenerate figures from saved eigenvalues. Products live in
`outputs/flattened_half_system_nx80_ny80/`: individual raw/resolved spectra,
diagnostics, source hashes and configuration, numerical histogram counts and
densities, and vector PDF/300-dpi PNG figures. The small-system regression test
compares the momentum calculation to a directly diagonalized canonical parent;
it also checks the roundoff filter and histogram normalization.

The completed 80 x 80 calculation retains 622/6,400 modes for `alpha_1=1`
and 600/6,400 for `alpha_1=3`. Both raw subsystem traces are 3,200 to numerical
precision. The nearest occupations to one half are approximately
`0.486700, 0.513300` for `alpha_1=1`, versus `0.016339, 0.983661` for
`alpha_1=3`. Thus the latter has no intermediate-occupation modes in this
finite-size ground-state spectrum; the small endpoint peaks in its histogram
are retained near-endpoint modes, not the discarded roundoff-level eigenvalues.
The standalone requested occupation figure is
`half_system_occupation_log_density.png`/`.pdf`; the two-panel companion adds
the modular-energy transformation of exactly the same retained modes.

`half_system_entanglement_energy_log_density.png`/`.pdf` is the standalone
single-particle entanglement-energy histogram. It transforms each retained
eigenvalue using `epsilon=log[(1-nu)/nu]`, then bins those energies in common
unit-width bins and separately normalizes each alpha to unit area. This is
not a relabeling of occupation bins, and not the many-body entanglement
spectrum formed from sums of single-particle energies. The occupation cutoff
and logarithmic density axis are unchanged. Internal NPZ keys named
`modular_energies_resolved` are retained for compatibility and contain these
same single-particle entanglement energies.

## Larger-system correlator review

`build_large_ny_correlator_review.py` builds the separate review package in
`outputs/large_ny_correlator_review_v1/`. Its shared loader,
`large_ny_correlator_data.py`, verifies the smaller every-cycle and larger
endpoint-only result pairs, extracts only compact correlator members, and
resolves missing large-campaign preparation metadata from the hash-bound
configuration function and source files. The primary series is
`Ny=24,28,32,40,50,60`, with 100 trajectories each. The old streaming
`Ny=30,40,50` ensembles are separate cross-checks, never automatically pooled.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u \
  00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis/build_large_ny_correlator_review.py \
  --threads 4
```

Start with `FINDINGS.md` and `large_ny_correlator_review_atlas.pdf` in that output
directory. `FIGURES.md` identifies each atlas page and its estimator, cohort,
fit window, and uncertainty convention. All fits are trajectory-first unless
explicitly marked as a fit to the ensemble-mean curve. Primary fits retain the
old quarter-circumference window; longer-tail windows and the amplitude-cutoff
variants are separate diagnostics. Each plot has PDF, 300-dpi PNG and plotted
CSV output. The compact unmasked inputs, full sample-wise fit table, primary
size summary, extrapolations, and static-reference arrays/diagnostics are saved.

The log-chord and scaling-curve panels follow the technical report's entropy
fit presentation: open data markers, gray primary-fit windows, and black dashed
unconstrained fits. Each dashed curve fits **the displayed ensemble mean** in
log-chord coordinates; it is not the mean of the trajectory-wise exponents.
Dashed extensions outside a size's fit window are extrapolations. On multi-size
panels, gray is the envelope of the size-specific windows, not a common range
or an uncertainty band. The dotted exponent-two guide remains separate. Fit
coefficients and windows are saved with the dashed-line series in the plotted
CSVs. The lower scaling panels retain trajectory-first exponents and IQRs;
no entropy-plot SEM convention is imported.

The runner reproduces the old Ny32 representative selection, 300 saved
wall-resolved fits and 100 historical x-average fits before building new plots.
Tests in `tests/test_large_ny_correlator_review.py` cover both archive shapes,
checksum/contract/sample validation, estimator order, synthetic exponents,
invalid windows and the static correlator's dense-covariance normalization.
This is an analysis-only review: previous figures, datasets, the manuscript,
and all Colab deployments remain unchanged. Rerunning replaces only this new
review package's generated products.
