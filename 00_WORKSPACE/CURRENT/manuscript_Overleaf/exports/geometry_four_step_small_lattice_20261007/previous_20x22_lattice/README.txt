Corrected, tightly cropped four-slice geometry

The topological slab contains exactly ten lattice columns. Five exterior
columns lie on either side (20 total). Each plane has 22 rows, keeping its
aspect ratio within about one percent of the approved 18-by-20 version.
The former eight-column slab is preserved in previous_eight_column_slab/.

The four support centers retain their offsets from the left interface and
the same local y coordinate. Supports contain 3x3, 2x3, 2x3, and 3x3 cells.
Labels, marker sizes, colors, opacity, lattice spacing, periodic seam marks,
and time-arrow alignment are retained. Excess external whitespace is removed.

The drawing is shared with Figure 1(a) through plot_schematic.py. The
standalone renderer preserves its larger preview label sizes; the manuscript
renderer uses the shared, final-print typography standard.

Reproduce: python exports/geometry_four_step_small_lattice_20261007/draw_small_lattice.py
