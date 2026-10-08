Channel relaxation and stationary-state occupation gaps
=====================================================

Main deliverables:
  channel_gap_consolidated.pdf     Compiled one-column pedagogical RevTeX note.
  channel_gap_consolidated.tex     Standalone source; inline bibliography.
  fixed_width_table.tex           Generated from 10 verified saved spectra.
  square_table.tex                Generated from 14 verified saved spectra.
  validation.json                Full-precision data, receipts, hashes, tests,
                                 configuration, and automated PDF checks.
  editorial_audit.json            Final mathematical and page-review record.

The one-column revision expands the derivations with a single-mode reset,
a two-mode projector product, an explicit higher-moment block matrix,
examples of operator sectors, the scalar affine recursion, and explanations
of stationary-mode exclusion, Poissonization, and reading the numerical tables.
The full proof, original references, and verified campaign data are retained.

Rebuild and validate from this repository (no simulations or fits):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build_note.py

Dependencies: Python with NumPy, SciPy and pypdf; pdflatex with RevTeX 4.2,
amsmath, amssymb, amsthm, mathtools, bm, booktabs, hyperref, microtype;
Poppler's pdftotext, pdftoppm and pdffonts.

The build reads the saved campaigns and original notes but writes only
inside this folder. It validates all 24 NPZ/completion pairs and the 10
matched fixed-width spectral pairs, regenerates both tables, runs the
29 existing proof checks, 24 consolidation checks, and checks of the new
pedagogical examples, and compiles four passes.
It also verifies the original notes against a pre-edit checksum baseline.
A concurrent manuscript edit detected during this task is recorded in
external_change_observation.json and preserved at its observed checksum.
This task modifies no old notes, data, manuscript, or source engines.
math_checks.py retains the existing proof checks locally so
the old helper is never imported or run.

For a LaTeX-only build outside the repository, copy the .tex source and
both *_table.tex files together, then run pdflatex four times. No external
bibliography, figures, campaign outputs, or manuscript files are needed.
The full provenance build requires the saved campaign results in the repo.
PDF timestamps are pinned by the build script; exact binary reproduction
also requires the same TeX distribution and fonts.
