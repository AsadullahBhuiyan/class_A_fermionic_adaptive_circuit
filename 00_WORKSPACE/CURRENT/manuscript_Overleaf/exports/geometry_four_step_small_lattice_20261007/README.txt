Four-slice geometry: 16 columns by 18 rows

Each plane has a 4+8+4 horizontal partition: four exterior columns, eight
slab columns, and four exterior columns. The previous 20x22 version is
preserved in previous_20x22_lattice/.

The alpha labels sit just above the final slice, as in the approved reference. Both exterior labels read
alpha_2 and are centered on their trivial-region centerlines; alpha_1 is
centered over the slab. The manuscript caption states that alpha_2 is fixed
at 30. The occupation label is right of its support,
vertically centered on the Wannier center.

Four support centers retain their offsets from the left interface and the
same local y coordinate. Supports contain 3x3, 2x3, 2x3, and 3x3 cells.
The opacity, lattice spacing, periodic seam marks, and time-arrow alignment
are retained, and external whitespace is cropped.

The drawing is shared with Figure 1(a) through plot_schematic.py. The
standalone renderer preserves its larger preview label sizes; the manuscript
renderer uses the shared final-print typography standard.

Reproduce: python exports/geometry_four_step_small_lattice_20261007/draw_small_lattice.py
