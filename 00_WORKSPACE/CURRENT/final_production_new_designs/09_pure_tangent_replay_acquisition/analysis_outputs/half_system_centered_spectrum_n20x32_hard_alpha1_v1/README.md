# Half-system centered-covariance histogram

Open `half_system_centered_spectrum.ipynb` and run all cells. The first code cell exposes an inclusive CPU range. Inputs are the four immutable hard-wall, alpha_1=1, N20x32 endpoint batches (100 samples, cycle 64). No dynamics run.

The subsystem is all x and y=0..15, both orbitals. The notebook computes the 640 eigenvalues of 2 F_A F_A† - I per sample, then pools all 64,000 values. It verifies input hashes and metadata, frame orthonormality, Hermiticity, spectral bounds, trace closure, and an alternate eigensolver cross-check.

Outputs: `centered_spectra.npz` (raw sample-resolved eigenvalues and IDs), `histogram.csv` (bin edges, counts, density), `diagnostics.json` (provenance and numerical checks), `figure_caption.txt`, and PDF/PNG figures. Histogram bins include endpoint mass; density has unit area.

Change `BIN_COUNT` in the plotting cell and rerun that cell and the final diagnostics cell to rebin without recomputing spectra. Running all cells rechecks input hashes and reuses the cache only if its identity and checksum match. Set `FORCE_RECOMPUTE=True` to recompute.

The bundled `latex_support/type1ec.sty` is the existing repository OT1 compatibility shim; the figure uses LaTeX labels without requiring the absent EC font package.
The PNG and inline notebook preview are rendered from the vector PDF using `pdftoppm` at 300 dpi, preserving the same LaTeX typography in both exports.

The mixed-mode section excludes abs(lambda) >= 1-MIXED_TOL (default 1e-8). Its separate CSV, diagnostics, caption, PDF and PNG use conditional unit-area normalization and a log density axis. Edit MIXED_TOL or MIXED_BIN_COUNT and rerun that plot cell to explore the retained modes without recomputing spectra.
