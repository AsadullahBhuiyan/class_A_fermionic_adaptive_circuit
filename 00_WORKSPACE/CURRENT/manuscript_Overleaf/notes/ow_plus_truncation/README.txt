Finite-range Wannier truncation: band mixing and a five-cell topological parent
============================================================================

Read ow_plus_truncation.pdf. The seven-page, one-column RevTeX note separates
the target band, truncated measurement modes, signed auxiliary parent, and
actual monitored trajectory.

The source is ow_plus_truncation.tex; references.bib contains its bibliography.
The manuscript's "Finite-Range Truncation" appendix supplies the starting
material. The earlier ow_truncation_consolidated note and all manuscript files
are preserved. No circuit simulation or manuscript edit is part of this work.

New plus results at alpha=1
--------------------------
Support: center plus four axial nearest-neighbor unit cells, both orbitals.
The same normalized two trial families and both band targets are used.

Retained squared norm:       0.951417267888
Opposite-band mode weight:   0.0255317691574
Half-gap min|E|:             0.930845284792
Full band separation:        1.861690569584
Occupied-band Chern number:  1
Positive-alpha parent transition: 1.50006789984

The plus parent has an explicit nearest-neighbor QWZ form. The note derives
its gap minimum and an everywhere-gapped path removing the square's corners.
Its topological interval is narrower than that of the nine-cell square.

Both square and plus masks retain C4 (90-degree) rotational symmetry.
The minimality statement is restricted to C4-invariant submasks of this plus
and the same signed two-band construction. It is not a general no-go theorem
for asymmetric supports. No claim about plus-circuit preparation is made.

Reproduction from this directory
---------------------------------
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -B reproduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build ow_plus_truncation.tex
cp build/ow_plus_truncation.pdf ow_plus_truncation.pdf
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -B validate.py

Requires NumPy, SciPy, matplotlib, pandas (native helper import), tqdm,
a TeX installation with RevTeX, latexmk, dvipng, and Poppler utilities.
No GPU or remote service is used. Numerical calculation takes roughly a minute
on this workstation. All outputs stay in this note directory.

reproduce.py reuses the preceding note's side-effect-free native-loader and
Fukui-Hatsugai-Suzuki helper, and the manuscript typography helper. It suppresses
the native plotter's two top-level mkdir calls and never runs its entry point.
The new generic window evaluation includes the plus, which the original
square-only renderer explicitly excludes.

Products and checks
-------------------
data/band_checks.csv: normalization, leakage, gap, overlap, Chern, frame identity.
data/winding_checks.csv: local-gauge winding, loop refinement, boundary amplitudes.
data/critical_points.csv: transition roots on two Fourier grids.
data/finite_lattice_rank.csv: exact finite-torus synthesis ranks and singular values.
data/transition_checks.csv: Chern numbers on either side of each parent transition.
data/corner_path.csv and plot_data.npz: exact arrays plotted in the figures.
data/numerical_checks.json: coefficient values, conventions, source hashes.
data/validation.json and output_manifest.json: final validation and artifact hashes.
source_audit.csv: what was retained, extended, or qualified.

Fourier grids are 1024^2 and 2048^2; momentum checks use 201^2 and 401^2.
Finite-grid extrema are numerical checks, not interval certificates.
The plus gap minimization and corner-removal mass argument are analytic,
with their coefficient integrals evaluated by converged quadrature.
The appendix's finite-tail estimate is not promoted to a certified continuum
bound in this note. Gap conventions are explicit to avoid a factor-of-two error.

Figures use the shared manuscript typography helper, vector PDFs and 300-dpi
PNGs. The final compiled PDF was inspected page by page. Source baselines for
36 pre-existing files are retained in data/protected_sources.json.

Two pre-existing files changed in parallel during this task: manuscript.tex
and the manuscript OW figure's typography receipt. They were reread/reviewed,
not overwritten. data/concurrent_source_changes.json records both identities;
34 other baseline files remain byte-identical. No file outside this new note
directory was edited by this task.
