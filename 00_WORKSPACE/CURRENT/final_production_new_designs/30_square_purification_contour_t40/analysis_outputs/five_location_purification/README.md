# Five-location purification contour

Both L=30 and L=40 runs completed 40 cycles. One independent Born trajectory
per size, global maximally mixed initial state, alpha1=1, alpha2=30,
nshell=1, hard support-truncated walls, full-system measurements
(meas_slab_only=False), perfect correction, raster-y, complex128.
The completed result/receipt pairs were downloaded from Google Drive on
2026-09-29 into `../../data/square_full_measurement_purification_l30-40_s1_t40_v1/`.
Original result files, production code, and Drive outputs are unchanged.

Reproduce from the bundle directory with `python plot_five_locations.py`.
The script verifies receipt checksums, bytes, config/source identities,
completion at cycle 40, sample and geometry metadata, occupation bounds,
entropy spectral reconstruction, charge, and contour closure.

## Figure and spatial convention

`five_location_purification.pdf` is a 3.375-inch-wide vector figure;
the PNG companion is 300 dpi. Each panel contains five curves over all cycles
0–40. At each selected x, the plotted value is

    s_y(x,t) = (1/L) sum_y entropy_contour[t,x,y].

The stored contour sums both orbitals of a cell and has axes (cycle,x,y).
Units are nats per unit cell. No time, trajectory, or left/right pooling is
performed. There are no ensemble error bars: S=1 at each size.

| Size | Left wall | Right wall | Left trivial center | Right trivial center | Topological center |
|---|---:|---:|---:|---:|---:|
| 30x30 | 8 | 22 | 4 | 26 | 15 |
| 40x40 | 10 | 30 | 5 | 35 | 20 |

Wall columns use the canonical engine's inclusive topological-slab endpoints
(x_L and x_R). The trivial-center columns are nearest integer columns to the
midpoints of the two displayed outer portions. Under periodic x boundaries,
those two portions are connected across the coordinate seam, not separate
physical trivial slabs.

All cycles are connected; markers are staggered every third cycle to expose
the near-coincident trivial-bulk curves without changing their values.
The logarithmic axis is not a fit. The gray reference is the approximately
5.726e-11 nats/cell regulator floor from two orbitals and entropy_eps=1e-12.
No new clipping or floor subtraction is applied to the saved contour.

## Interpretation

This is full-system mixed-state purification entropy, not subsystem
entanglement. The trivial centers reach the regulator floor by approximately
cycle four. The topological centers retain very small, fluctuating entropy,
while much larger residual mixedness remains at the two interfaces. At cycle
40 the full-system entropies are 0.44857 nats (L30) and 0.99946 nats (L40).
These one-trajectory data do not establish an ensemble purification rate or
complete interface purification by 40 cycles. This does not contradict the
earlier plateau of a subsystem-entanglement diagnostic from a different
initialization/protocol.

Caption: **Spatially resolved purification.** Full-system entropy contour,
averaged along y at the two domain-wall columns, the centers of the two
displayed trivial portions, and the topological-slab center. Panels show
30x30 and 40x40 systems, each initialized globally maximally mixed and evolved
for 40 cycles under hard-wall, full-measurement, perfect-correction dynamics.
Each curve comes from one Born trajectory (S=1); no ensemble uncertainties or
rate fits are shown. The gray dashed line marks the entropy regulator floor.

`five_location_curves.csv` contains all plotted values. `all_x_profiles.csv`
contains every y-averaged column profile. `summary.json` records source Drive
URLs, hashes, identities, selected columns, diagnostics, and output hashes.
