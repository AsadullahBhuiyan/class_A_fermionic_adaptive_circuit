# Full-spectrum search for pure-endpoint physical-wall modes

## Outcome and correction to the nearest-four comparison

The previous plot examined only the four lowest-absolute-modular-energy modes.
It did not justify excluding physical-wall localization elsewhere in the
spectrum. This extension checks **all 165,816 numerically resolved modes** in
the 1,200 slot09 endpoints (12 cases, 100 independent trajectories each).
All their complex eigenvectors and spatial densities are now retained.

There ARE additional wall-concentrated eigenvectors. However, being concentrated
in x near a wall is not the same as extending down the wall away from the
entanglement cut. Most additional strongly wall-concentrated modes lie within
a few y rows of that cut. The scan therefore corrects an incomplete mode
selection, not a dynamics regression or a demonstrated physical failure.

For Ny=32, using the deliberately explicit diagnostic criteria
wall weight >80%, cut weight <50%, and numerical eigenvalue separation:

| Walls | alpha1 | Modes | Trajectories / 100 | Smallest absolute epsilon |
|---|---:|---:|---:|---:|
| hard | 1 | 2 | 2 | 14.70 |
| soft | 1 | 16 | 13 | 14.81 |
| hard | 3 | 6 | 6 | 18.06 |
| soft | 3 | 22 | 21 | 15.26 |

The wall window is x=4,5,6,14,15,16. The cut window contains two y rows at
each end of the retained half: y=0,1,Ny/2-2,Ny/2-1. These thresholds are
localization diagnostics, not definitions of an in-gap state or topology.
Unfiltered wall-concentrated mode counts (without the cut/separation conditions)
are respectively 16,65,36,896. Many are corner/cut modes.

### Distance from the entanglement cut is decisive

At Ny=32, widening the cut neighborhood from two to three rows removes all
qualifying alpha1=1 modes, hard and soft. The hard alpha1=3 candidates also
disappear. One soft alpha1=3 mode remains: sample 86, epsilon approximately
19.99. It has about 50.4% joint weight in the physical-wall windows and middle
y=4,...,11 region, but falls below the tighter 1e-8 occupation resolution cap.
Thus these data do not establish an extended, low-modular-energy, physical-wall
branch from this half-y eigenproblem. They do establish additional spatially
wall-concentrated eigenvectors, mostly associated with the near-cut region.
The distinction is visible directly in the saved maps; it is not inferred only
from a scalar threshold.

At smaller sizes there are some lower-energy candidates: the smallest for
alpha1=1 is |epsilon|=7.33 (soft Ny=28). Hard Ny=28 alpha1=3 also has an
outlier at |epsilon|=2.88. Results and threshold sensitivity for ALL sizes are
in `summary.json`; Ny32 statements must not be generalized to every size.

## What was scanned, and what “in-gap” means here

Use the exact same raw data, pure initialization, perfect correction, hard/soft
wall implementations, and pinned sources as the parent analysis. Nx=20,
Ny=24,28,32, alpha1=1,3, alpha2=30, nshell=1, raster-y, complex128, T=2Ny.
For each sample take valid columns of the saved final occupied frame, restrict
to all x and the first half of y, and diagonalize G_A=F_A F_A^dagger.
The eigenvectors are also those of the single-particle modular Hamiltonian;
epsilon=log(1-nu)-log(nu). No trajectory averaging is applied before diagonalizing.

There is no independently established bulk modular-gap boundary for this
trajectory-resolved spectrum. Rather than inventing an “in-gap” window, scan
every mode with 1e-9 < nu < 1-1e-9, covering |epsilon| < approximately 20.72.
The term **resolved** is not synonymous with **in-gap**. We cannot infer a
protected edge branch or chirality merely from spatial localization, especially
when similar candidates occur at alpha1=3. Higher-energy candidates are
near-pure occupation directions and require numerical qualification.

Full-spectrum plots retain all resolved modes, including those whose
individual eigenvectors are not well separated. The count `robust_candidate`
means only the declared spatial criteria plus a singleton occupation cluster
at spacing tolerance 1e-10; it is not a claim of robustness to physical or
geometric perturbations. Near-degenerate vectors are saved with cluster flags
and excluded from this individual-vector count. Tightening the occupation cap
to 1e-8 leaves Ny32 candidate counts 2,13,3,10 in the table's row order.
`summary.json` also varies wall weights (70%,80%,90%) and cut weights (25%,50%).

Previous legacy modular-spreading reports involve translated-cut averaging
before constructing the modular kernel; some use exact or flattened-OW target
states, not individual circuit trajectories. They are not identical estimators.
Their packet propagation and parent-Hamiltonian in-gap branches must not be
silently identified with eigenvectors of the present single-trajectory G_A.
The current scan neither reruns nor refutes those older transport experiments.

## Figures

`all_resolved_modes_Ny32.pdf` / `.png`: every resolved modular eigenmode at
Ny32 for all 100 trajectories in each hard/soft, alpha1=1/3 case. Horizontal
coordinate is signed epsilon; vertical coordinate is physical-wall weight;
color is entanglement-cut weight using the two-row convention. Red rings mark
spatially qualifying, numerically separated candidates. The gray line is the
80% wall diagnostic, not a bulk-gap edge. Dots are individual modes and are
not independent statistical samples; no uncertainties, fits, or averaged
matrices are used. Pure initialization, T=64, remaining settings as above.

`wall_modes_away_from_cut_Ny32.pdf` / `.png`: in each case, the qualifying
mode with smallest |epsilon|, selected across 100 trajectories, shown as
orbital-summed density. These are **selected existence examples, not typical
trajectories**. Cyan lines mark the walls. The horizontal edges at y=0 and
15 bound the half-system. Each density sums to one and each panel has its own
color scale. They demonstrate the near-cut character that persists even when
the first two cut-adjacent rows contain less than half the weight.

## Data and validation

- `modes/`: all resolved spectra, vectors, xy densities, wall/cut/joint-away
  weights, separation flags, residuals, and source receipts, per trajectory.
- `all_resolved_modes.csv`: one row per resolved mode, including file/column
  location. No restriction to the previously saved first 16 modes.
- `manifest.json`: source and output hashes, coverage and selection conventions.
- `summary.json`: per-case counts, energy windows, and threshold sensitivity.
- `cut_distance_sample_counts.csv`, `cut_distance_summary.json`: one-, two-,
  three-, and four-row cut neighborhoods, including joint wall/away weights.
- `examples.json`: exact sample, eigenvalue index, and diagnostics for each map.
- `validation.json`: all 1,200 output hashes verified; four plotted vectors and
  the deepest candidate independently recomputed with LAPACK evr rather than
  evd. Squared overlaps agree with one to approximately 2e-15.

All 32 raw acquisition pairs passed checksums and identity checks. Maximum
eigenvector residual is 2.80e-15. Normalization checks passed for all 165,816
saved densities. This is local postprocessing, with no new circuit simulation.
The extra NPZs occupy 1.81 GB (1.69 GiB); previous analysis outputs are preserved.

Reproduce from the parent directory:

```bash
OPENBLAS_NUM_THREADS=2 python -m unittest -v test_full_scan.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u full_scan.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u check_full_scan.py
```
