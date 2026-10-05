# Sample-resolved endpoint modular packets

## Chirality presentation v2 (current)

The current main figure is `modular_handedness_n20x32_y8_v2.pdf/png` in
`outputs/hard_n20x32_a1-1-3_y8_chirality_figure_v2/`. It shows separate left/right
space-time maps, both ensembles' raw displacements, and the within-trajectory
wall contrast `(dy_x5-dy_x15)/2`, with trajectory mean +/- SEM. The contrast is
not a topological invariant or a velocity estimate.

Map colors are wall-window conditional probability per row: sum each packet's
density over the five-column window, divide by that realization's instantaneous
retained charge, THEN average 32 cuts within each trajectory and finally the
100 trajectories. The map's first moment is therefore the displacement curve.
No matrices are averaged before evolution. `y_rel` is measured from the cut
origin; injection is at `y_rel=8`, and maps use `y_rel-8` to match displacement.

Companions: `modular_packet_snapshots_n20x32_y8_v2.pdf/png` (2x5 snapshots at
0,0.1,0.2,0.5,1), `modular_handedness_cutoff_comparison_v2.pdf/png` (saved cutoff
results with explicitly expanded control scale), and `modular_packet_retention_v2.pdf/png`
(retained charge / initial charge 2). Snapshot circle area is LINEAR in density
with common normalization; open rings mark injection and black ticks mark the
averaged wall-window centers. Full raw densities are saved even though cells
below 1e-4 are omitted from the scatter rendering.

Run from this directory:

```bash
python analyze_chirality_figure.py --workers 8
python plot_chirality_figure.py
python -m pytest -q test_endpoint_packets.py test_chirality_figure.py
```

Compute verifies eight endpoint shards and reduces 200 trajectories at epsilon
1e-10. It resumes checksum-bound trajectory NPZ/JSON pairs and checks all
trajectories against the saved early-time baseline to absolute tolerance 1e-10.
Eight available CPUs use single-threaded BLAS by default. Smoke tests require
`--sample-limit 2 --output /tmp/modular_chirality_smoke`; give the same `--output`
to plotting. Plot verifies aggregate/reference hashes and never writes analysis
caches. Styling changes do not invalidate computation. Failed, partial, or
mismatched completion pairs are retried, never counted as complete.

The NPZ contains per-trajectory probabilities, five spatial densities, raw/full
displacements, retained charge and wall contrasts, plus means and SEM.
`analysis_summary.json` records source hashes and regression/moment checks;
`figure_metadata.json` records scales and the plotted data hash.
`chirality_figure_caption.md` describes all four figures. Report drop-in and
companion snippets do not change the main manuscript automatically. All earlier
output directories and figures remain intact.

Interpretation: opposite modular drift compared with a weakly mobile alpha1=3
control. Signs persist across tested cutoffs, but magnitude and ripple period
are cutoff dependent. No regularization-independent speed is claimed.

## Latest early-time figure and clipping check

The current presentation is `modular_packets_n20x32_y8_stacked.pdf/png` in the
early-time output directory. Regenerate it with `python plot_early_time_stacked.py`
without recomputing observables or invalidating analysis caches. Top to bottom:
(a) alpha1=3 density, (b) alpha1=1 density, (c) alpha1=1 displacement. Titles are
removed, alpha1 labels sit inside the density axes, and the curve legend is
offset from the light zero guide. `stacked_figure_metadata.json` binds the plot
to the unchanged input data hash. The older horizontal rendering is preserved.

`analyze_early_time.py` produces a separate, versioned analysis at
`outputs/hard_n20x32_a1-1-3_y8_early_time_clip_v1/`. Panels (a,c) use modular
times 0,0.1,0.2 and panel (b) covers 0..1. It analyzes all 100 trajectories and
32 cuts for each alpha1 at clipping thresholds 1e-8,1e-10,1e-12. One covariance
eigendecomposition per cut is reused across thresholds; evolution and observables
remain separate. Both controls have full short-time curves. The main figure
uses epsilon=1e-10; `clipping_sensitivity.pdf/png` shows the comparison.
Run `python analyze_early_time.py --workers 16` from this directory. A separate
`--output` is required for `--sample-limit` smoke tests. The run verifies source
checksums and resumes verified per-trajectory caches. It sets single-threaded
BLAS and limits affinity to the first available worker-count CPUs.

See `early_time_caption.md` for the figure protocol and output
`clipping_assessment.md` for the measured sensitivity. Saved products include
the full per-trajectory cutoff comparison, paired displacement differences,
packet-weighted energy spreads and clipped-mode weights, and per-cut minimum
absolute energies. Original long-time outputs below remain unchanged.

## Original long-time analysis

Requested replacement of the legacy modular-propagation figure, with a trivial
alpha1=3 density control. Uses campaign 09's saved Nx=20, Ny=32 hard-wall pure
endpoints after 64 physical cycles, alpha1=1 and 3, alpha2=30, nshell=1,
perfect correction, raster-y, complex128, 100 independent trajectories per
alpha. Both packets begin at (x,y_rel)=(5,8),(15,8). Each independently evolved
packet contains two local occupied orbitals, total charge 2. No simulation is
rerun and existing data/figures remain unchanged.

For every sample and each of the 32 periodic cut origins, form the reduced
occupation matrix P_A=F_A F_A^dagger on all x and 16 consecutive y rows. Diagonalize
G_A=2P_A-I and use h_A=-2 atanh(G_A), clipping G_A eigenvalues to
[-1+1e-10,1-1e-10] as in the legacy analysis. There is no division by physical
cycle count, no ensemble averaging of matrices before evolution, and no applied
wall-orientation sign. Propagate using the exact spectral exponential.

Panel (a) averages evolved densities at modular times 0,0.5,1,1.5 for alpha1=1.
Panel (b) computes each realization's wall-window COM displacement before
averaging; the window includes columns within periodic x distance 2 of its
source. Its saved times are 0..32 with spacing 0.01; panel (b) is zoomed to 0..4.
Panel (c) repeats (a) at alpha1=3.
Average all 32 origins within each trajectory, then average trajectories.
Displacement shading is +/- SEM across 100 independent trajectories; origins
are not counted as independent samples. Full-subsystem COM and retained window
charge are saved as diagnostics. No smoothing or COM unwrapping is performed:
the reduced y coordinate runs from 0 to 15, so midpoint displacement lies in
[-8,7]. The control's numerical displacement is saved at the four density times,
but no full control time curve is calculated or plotted.

Density marker area is proportional to sqrt(mean density/2), with a common
normalization in both panels. Cells below 1e-4 are hidden only in the rendering;
all cells enter averages and charge-conservation checks. Markers overlay in the
legacy time order. The panels represent modular evolution, not monitored-circuit
time evolution or a physical injected-charge response.

Run from any directory (choose available CPU IDs):

```bash
python /home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/endpoint_modular_packets/analyze_endpoint_packets.py --workers 16 --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Outputs live in `outputs/hard_n20x32_a1-1-3_y8_observable_mean_v1/`: PDF/PNG,
displacement CSV, NPZ ensemble and per-sample means, diagnostics/provenance JSON,
and resumable per-sample caches. Input NPZ hashes must match the campaign's
completion JSON. Cache identities bind sources, scientific parameters, and
analysis script hash; cache payload hashes are checked before reuse.
Use a separate `--output` with `--sample-limit 2` for a small smoke test.

Interpretation: the packet location differs from the legacy opposite-corner
design. Motion from y_rel=8 can have either sign, whereas corner starts impose
one-sided displacement bounds. A weak/absent midpoint drift is not a code
failure or by itself evidence of absent chirality. Preserve this protocol
difference and the unmodified legacy outputs when comparing results.
