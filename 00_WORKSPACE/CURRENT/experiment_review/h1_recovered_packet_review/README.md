# Recovered H1-v3 Figure 15 comparison

Run `python build_figure15_comparison.py` from any directory. Generated figures,
plotted arrays, CSV, and source provenance live in
`outputs/figure15_hard_alpha1_alpha3_v1/`. No simulation is performed, no archive
is extracted to disk or modified, and the original report/Figure 15 is unchanged.

## Data

Only the recovered hard-wall H1-v3 archives are used: alpha1=1 has sample IDs
0..19 (20 trajectories); alpha1=3 has IDs 0..24 (25 trajectories). Receipt-only
entries are not counted. The loader checks each archive's size/SHA-256 against
its receipt, every packet product's internal checksum, model/run/observer
contracts, embedded configuration, source identity, numerical status, sample
uniqueness and axes. The observer source used to interpret the arrays is
byte-identical to the observer hash in each archive. The two cohorts have the
same recorded engine hash. This does not claim an independent audit of the engine.

Nx=20, Ny=40, alpha2=30, nshell=1, hard support truncation and slab-only
measurements, perfect correction, complex128, random measurement ordering,
and final monitored cycle 80. Sources use the campaign's half-filled random
pure-state initialization. Forty translated half-cylinder cuts are analyzed
inside each trajectory. The selected diagonal packets are centered at
(x,y-relative)=(5,0) and (15,19).

## Important differences from the original Figure 15

This is a Figure-15-style comparison, not an identical-estimator substitution.
The old figure used Nx=16, cycle 50, raster-y ordering, modular times through
32, and two independently occupied orbital sources. Its spatial panel used
an averaged modular generator. H1 uses normalized coherent equal amplitudes
across both orbitals and three wall-centered x columns, and evolves each
trajectory/cut independently before averaging observables. Its saved time
range is 0..8, and cycle 50 is absent (saved cycles are 40,48,56,64,72,80).

H1 retains only the primary conditional y profile, summed over the three-column
strip x=4,5,6 or x=14,15,16 and normalized by probability retained in that strip.
It does not retain the full xy density or modular eigenvectors. Thus the left
panels display **wall-projected** profiles at x=5 and x=15, not reconstructed
two-dimensional propagation. They cannot resolve leakage in x. The dot positions
in x label the strip; they do not assert that all probability lies on the wall.

The right panels use the ordinary linear conditional center, not an unwrapped
full-subsystem center: delta-y = center(t)-center(0). The 40 cuts are averaged
inside each trajectory, then trajectories are averaged. The band is ddof=1 SEM
over 20 or 25 trajectories, retained to match Figure 15; no bootstrap is used.
The saved profiles pass normalization and center/profile consistency checks.
Retained probabilities are exported separately in `plotted_data.npz` and the CSV.

The large difference between the two alpha cohorts is visible without changing
axes: peak absolute mean displacements through modular time 8 are about 11.65
and 6.55 cells for alpha1=1, versus 0.032 and 0.049 for alpha1=3. At time 2 the
means are (+10.286,-6.545) and (+0.0186,-0.0247), respectively. Late-time curves
are nonmonotonic; they are not fitted ballistic velocities. Opposite displacement
signs for two opposite starting endpoints alone are not a chirality proof because
an ordinary center is bounded by the subsystem. The paired-endpoint diagnostic
is preserved in the data export but no additional physical claim is forced.

## Suggested caption

**Modular packet spreading in recovered hard-wall ensembles.** Nx x Ny=20 x 40,
alpha2=30, nshell=1, final monitored cycle 80, random-order perfect correction
from pure half-filled random initial states. Top: alpha1=1, S=20 independent
trajectories. Bottom: alpha1=3, S=25. Left: conditional wall-strip y profiles
for coherent unit-norm sources centered at (5,0) and (15,19), shown at modular
times 0,0.5,1 in purple, orange and green. Profiles sum the three wall-centered
x columns, normalize inside each cut, and average cuts then trajectories.
They are plotted at the wall centers as a projection, not as full xy densities.
Marker area is proportional to the square root of conditional probability;
values below 1e-4 are omitted only from the display. Right: mean displacement of
the corresponding strip-conditioned centers, averaging 40 cut origins within
each trajectory before forming the ensemble mean and SEM band. The modular
occupation regulator is 1e-10. Both rows use identical scales. Modular time
is not monitored-circuit time; curves are not linear fits.
