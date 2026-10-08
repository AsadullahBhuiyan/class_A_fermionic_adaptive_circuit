# Finite-range overcomplete Wannier measurements and bulk topology

Standalone one-column RevTeX note consolidating the three existing OW notes.
This task does not edit the main manuscript; all original notes, figures,
and data are unchanged.
Only deterministic band calculations are performed here; no circuit is run.

## Read

- `ow_truncation_consolidated.pdf`: complete note, including two technical appendices.
- `ow_truncation_consolidated.tex` and `references.bib`: editable sources.
- `source_audit.csv`: claim-by-claim dispositions, kept out of the physics narrative.
- `reference_audit.csv`: primary references, their role, and verification links.
- `figures/`: two final-size vector PDFs and 300-dpi PNG companions.

The central distinction is between a truncated measurement mode, its quadratic
frame operator, the auxiliary occupied band, and a conditioned trajectory.
Range-one numerical checks support stability of the auxiliary band. They are
not a continuum certificate or a theorem about every monitored trajectory.

## Reproduce

From this directory, using the repository Python environment with NumPy,
SciPy, Matplotlib, tqdm, and the system TeX distribution:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -B reproduce.py
python -B -m unittest -v test_derivations.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build ow_truncation_consolidated.tex
cp build/ow_truncation_consolidated.pdf ow_truncation_consolidated.pdf
python -B validate.py
```

`reproduce.py` reuses the existing `technical_report/analyze_ow_truncation.py`
functions. Its read-only loader suppresses exactly the two top-level output
directory creation calls and never calls the original renderer. Function
definitions are not copied or modified. The manuscript typography helper is
loaded in a separate module instance with output paths redirected here.
All generated outputs stay in this directory. Run with `-B` to avoid bytecode
cache writes in imported source directories.

The calculation uses 1024² and 2048² Fourier grids, 201² and 401² extrema grids,
and refined local loops in two gauges. The two figures are included at their
6.5-inch native width, with the manuscript's Computer Modern typography.
The first figure uses unit real-space mode norms, not independently normalized
overlaps. The second plots minimum absolute eigenvalue of the unnormalized
interpolation; its full two-band separation is twice that quantity.

## Numerical products

- `data/figure_data.npz`: every plotted curve and heatmap.
- `data/band_checks.csv`: all combinations of Fourier grid, extrema grid, and range.
- `data/table_rows.tex`: table rows generated directly from those checks.
- `data/numerical_checks.json`: normalization, frame, winding, Fourier-convergence,
  topology, and onsite-crossing receipts, including source hashes.
- `data/protected_sources.json`: before-edit hashes for 61 protected source files.
- `data/validation.json`: independent final checks and the trajectory-summary source.
- `data/output_manifest.json`: output and local-source byte counts and SHA-256 hashes.
- `figures/data/typography/`: final-size typography records.

The quoted existing Chern values come only from
`00_WORKSPACE/CURRENT/experiment_review/slab_topology_charge_study/results/v1/summary.json`:
the square cohort's saved endpoint means at cycle 40, averaged within each
trajectory before ensemble averaging. This note does not reanalyze that campaign.
Its original disk radius scales with L; the text explicitly distinguishes the
result from an exactly quantized invariant or a fixed-radius scaling test.

## Preserved source notes

1. `00_WORKSPACE/LEGACY/form_factor_analysis/docs/windowed_chern.tex`
2. `00_WORKSPACE/CURRENT/experiment_review/ow_truncation_topology_note/ow_truncation_topology.tex`
3. `00_WORKSPACE/CURRENT/manuscript_Overleaf/notes/truncation_topology/truncation_topology.tex`

The validation compares the protected files against the original snapshot;
it does not reset or overwrite them. All 60 files other than `manuscript.tex`
remain byte-identical. That file changed concurrently, outside this note's
write/build path; its original and observed hashes are recorded separately in
`data/concurrent_source_change.json`. Those changes were left alone, not
reverted or absorbed into the baseline. Further source changes are reported
by validation. The note's production scripts never write the manuscript.
