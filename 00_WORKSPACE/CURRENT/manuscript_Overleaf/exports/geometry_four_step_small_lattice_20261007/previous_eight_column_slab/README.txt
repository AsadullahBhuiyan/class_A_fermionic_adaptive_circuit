Smaller lattice version of the four-step OW-support schematic

Each plane is 18 columns x 20 rows instead of 24 x 26: scale factors 0.75
and 0.769. This is the nearest even-row lattice to the requested 0.75 scale.
The plane aspect ratio changes by only 2.5 percent due to integer rounding;
the 7.05 x 7.5 inch canvas aspect ratio is unchanged.

Lattice spacing, dot sizes, support sizes (3x3 and clipped 2x3), center-marker
sizes, font sizes, line weights, colors, opacity, and time-slice offsets are
unchanged. Phase labels and dimension arrows follow the smaller boundaries.
Support centers retain the same positions relative to the left wall, at the
same local y. The operator label is nudged 0.2 cell toward its support to clear
the narrower exterior border, retaining its vertical center alignment.

The original version and manuscript remain unchanged.
Reproduce: python exports/geometry_four_step_small_lattice_20261007/draw_small_lattice.py
