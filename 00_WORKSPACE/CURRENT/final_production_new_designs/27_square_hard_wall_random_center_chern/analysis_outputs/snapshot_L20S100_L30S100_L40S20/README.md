# Square hard-wall Chern analysis: 220-trajectory snapshot

Browser-downloaded on 2026-09-28 from the campaign's Drive output folder.
All nine NPZ/completion pairs passed full-file SHA-256/byte verification and
the runner's scientific array checks: exact cycles, deterministic distinct
centers, means, source/config identity, integer charge, and final-frame
complex128 dtype/rank/norm/padding. This snapshot contains S=100 at L=20,
S=100 at L=30, and only S=20 at L=40; 16 five-trajectory tasks remain absent
from the local snapshot. This does not claim the live campaign has stopped.

## Results

Ten centers are averaged within each trajectory at each cycle, then cycles
21–40 are averaged within that trajectory. Reported uncertainties are the
sample-to-sample SEM, std(ddof=1)/sqrt(S), over these trajectory averages.
There is no bootstrap, and neither centers nor cycles are independent samples.

| L | R | S | Mean C_G over cycles 21–40 ± SEM | Absolute deficit, percent |
| --- | --- | --- | --- | --- |
| 20 | 4 | 100 | 0.997686 ± 0.000123 | 0.23143 |
| 30 | 6 | 100 | 0.999600 ± 0.000053 | 0.04003 |
| 40 | 8 | 20 (partial) | 0.999893 ± 0.000035 | 0.01072 |

The complete L=30 ensemble has a 5.78-fold smaller late-time deficit than
L=20. Their mean difference is 0.001914 ± 0.000134 (independent-ensemble SEM).
The partial L=40 ensemble continues the trend, but its final S=100 mean may
shift as more records arrive. No scaling exponent is fitted to these three
sizes, especially with the last ensemble incomplete.

All sizes reach about 0.969 after two cycles and approach their respective
plateaus within roughly 5–10 cycles. Paired differences between cycles 31–40
and 21–30 are -0.000085 ± 0.000264, +0.000035 ± 0.000087, and
+0.000020 ± 0.000070. They show no clear continuing late-time drift, but this
is not a convergence proof for other observables.

Mean within-trajectory center variances, also averaged over cycles 21–40,
are 1.236e-4 ± 1.502e-5, 2.931e-5 ± 9.442e-6, and
4.033e-6 ± 2.566e-6. Spatial fluctuations decrease in this snapshot too.
At cycle 40 alone the means are 0.997574 ± 0.000470,
0.999857 ± 0.000076, and 0.9999928 ± 0.0000032; the last point is unusually
close to unity relative to its time average, so the late-window comparison
is the more representative headline.

The new L=20 late mean agrees with the independent bundle23 L=20 result
(0.997772 ± 0.000115). Those ensembles remain separate.

## Interpretation and limitations

This is evidence that the interior Chern estimator becomes closer to unity
as the square geometry grows. Both slab width and disk radius increase:
Nx=Ny=L, R=0.2L, x0=L/2. It does not isolate an Nx-only effect or distinguish
finite-disk and wall-distance effects. The canonical L=30 interfaces are at
x=8,22, rather than fractional x=7.5,22.5; the R=6 disk remains contained.
The initial protocol is pure random half-filled states followed by hard-wall
exterior preparation, nshell=1, alpha1=1, alpha2=30, perfect correction,
slab-only raster-y measurements, periodic boundaries, complex128, and
exactly forty physical cycles. No dynamics were rerun for this analysis.

## Figure and reproducibility

`square_chern_convergence.pdf` is the vector figure; its PNG companion is 300 dpi.
Panel (a) shows cycles 0–10, and (b) shows the absolute deviation from unity
through cycle 40 on a log scale. Markers represent every saved cycle; shading
is one trajectory SEM (transformed through the absolute value in panel b,
with zero lower endpoints clipped to the displayed log floor). The partial
L=40 curve explicitly reports S=20. No fitted curves are included.

`cycles.csv` and `trajectories.csv` contain the plotted statistics and the
trajectory-level late averages. `summary.json` records the exact inputs,
hashes, configuration, pending tasks, statistics and product checksums.
Run `OPENBLAS_NUM_THREADS=1 python ../../analyze_download.py` to reproduce
from the imported data. New completed tasks produce a separately named
snapshot rather than overwriting this one.
