# Postselected total and spatial purification, alpha1=1 and 3

Hard walls only, Nx=20, Ny=40, one deterministic fully postselected trajectory
per alpha, T=160. No perfect-correction feedback and no sample averaging.
This figure is deliberately separate from the Born-ensemble figure.

Panel a: total entropy S/Ny for both alpha values.
Panel b: y/orbital-summed entropy s_x/Ny at x=5,10,15 for both alpha values.
Both use cycle/Ny on a linear horizontal axis and logarithmic entropy.
In panel b color/marker encodes x, solid lines alpha1=1, dashed lines alpha1=3.
No smoothing, fitting, clipping or uncertainty bars. Near-zero alpha3 values
are roundoff-limited and not interpreted as physical plateaus. The x=5 and
x=15 alpha1=1 curves nearly overlap; all three alpha1=3 tails nearly overlap.

Source v2 NPZ/completion pairs are checksum-verified against the historical
download manifest. Spatial contour sums reproduce total entropy. Figure
size is 3.375 x 4.6 inches, with vector PDF and 300-dpi PNG. Raw values plotted
are in curves.csv, and provenance/normalization in summary.json.
Reproduce with plot_postselected_total_spatial_comparison.py in bundle 19.
See figure_snippet.tex for the caption and revtex_layout_check.pdf for placement.
