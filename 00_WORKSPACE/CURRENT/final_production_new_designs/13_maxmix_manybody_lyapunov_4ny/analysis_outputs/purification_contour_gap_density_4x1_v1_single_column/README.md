# Four-panel hard-wall purification figure

Vector PDF and 300-dpi PNG are 3.375 x 6.8 inches. Panels (a-c) reuse the
previous single-column three-panel figure's tables and fit unchanged.
Panel (d) adds the original Ny=30 sample-averaged endpoint mode-density map,
with its original linear color range (shared previously with Ny=20,40).
The heatmap is not an entropy contour: each trajectory's resolved-mode
probabilities are normalized to one before taking an equal trajectory mean.
No image cropping, smoothing, simulations, or new eigendecompositions are used.

The source density NPZs were checked against their extraction manifest, with
100 samples per size, unit normalization, and the stored mean reproduced from
sample densities. `analysis_summary.json` records input checksums and provenance.

Use `figure_snippet.tex` with `graphicx` and `amsmath` in a two-column RevTeX document.
`revtex_layout_check.pdf` demonstrates the single-column figure plus caption;
no `figure*` or `widetext` is required. The original three-panel outputs remain
untouched. The full scientific caption is provided in the snippet.
