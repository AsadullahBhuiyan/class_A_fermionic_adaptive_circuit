# Local CPU flattened-ground-state reference

This local, deterministic CPU calculation evaluates the half-filled ground state
of the overcomplete-Wannier flattened parent

```text
H_flat = sum_R(P_A+ + P_B+ - P_A- - P_B-)
```

for `Nx=20`, `Ny=20,24,28`, `nshell=1`, `alpha_2=30`, the locked 21-point
descending `alpha_1` grid, and both hard/support-truncated and soft/untruncated
wall constructions. It is separate from the dynamical Colab campaign.

The two observables are:

- translation-averaged mutual information between opposite full-`x` strips of
  width `w=Ny//4`;
- the finite-size central-charge fit `c_fit=3m`, where `m` is obtained from
  `S(Ay)=m log[(Ny/pi) sin(pi Ay/Ny)]+b` over `Ay=2,...,Ny//2`.

The completed local CPU products are under `results/flattened_ground_state_reference/`:

- `flattened_ground_state_reference.npz`: full arrays and diagnostics;
- `flattened_ground_state_reference.csv`: one row per wall/size/alpha case;
- `flattened_ground_state_reference.pdf` and `.png`: the 2-by-2 comparison;
- `flattened_ground_state_reference.complete.json`: checksums and provenance.

To verify the existing products without recomputing:

```bash
python run_flattened_ground_state_reference.py \
  --output-root results \
  --report-only
```

To recompute locally using eight MKL/OpenMP threads, omit `--report-only` and add
`--force`:

```bash
MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
python -u run_flattened_ground_state_reference.py \
  --output-root results \
  --force
```

The calculation uses Torch `complex128` on CPU. The saved projector,
translation, restricted-spectrum, and checksum diagnostics all pass.

## Large-`Ny`, three-shell extension

`run_flattened_ground_state_large_ny.py` is a separate local-CPU campaign for
`Ny=40,50,60`, `nshell=1,2,inf`, both wall constructions, and the same
descending 21-point alpha grid. It uses the canonical NumPy/SciPy CPU class
`classA_U1FGTN`. Translation symmetry along `y` block-diagonalizes the exact
flattened parent into `Ny` momentum blocks of size `2*Nx=40`; filling the lowest
half of their combined spectrum gives exactly the same projector as a full
dense diagonalization.

The mutual-information widths are `w=Ny//4`, namely `10,12,15`. Thus `Ny=50`
uses the nearest lower integer width because an exact quarter strip would have
12.5 lattice rows. The opposite displacement remains exactly `Ny//2`.

Run or resume with:

```bash
python -u run_flattened_ground_state_large_ny.py \
  --workers 12 \
  --threads-per-worker 4
```

The runner publishes one checksum-bound task pair per deterministic case, then
creates a separate large-size NPZ/CSV/figure and a geometry schematic under
`results/flattened_ground_state_large_ny/`. A report-only verification is:

```bash
python run_flattened_ground_state_large_ny.py --report-only
```

For two equal opposite strips on a circle, the free-Dirac CFT reference is

```text
x = sin^2(pi*w/Ny),   I_ab = -(c_eff/3)*log(1-x).
```

At exact quarter width and `c_eff=1`, this is `log(2)/3`.

The reference is the conformal-vacuum free-Dirac result. Periodic finite-size
fermion sectors can have additional zero-mode dependence, so the campaign uses
it as a continuum benchmark rather than a sector-independent identity. At the
exact grid point `alpha_1=2`, the Bloch vector vanishes at `k=(0,0)`; the saved
calculation uses the explicit symmetric prescription `h(0)=0` and
`P_+(0)=P_-(0)=I/2`. Open plot markers distinguish that midpoint convention
from the neighboring gapped spectral projectors.

## Two-column campaign note

The standalone REVTeX write-up is
`docs/flattened_domain_wall_mutual_information.tex`; its published PDF is
`docs/flattened_domain_wall_mutual_information.pdf`. The note derives the
free-Dirac two-interval result, explains the fixed-cross-ratio plateau, and
presents the small- and large-circumference results in full-page figures.

The documentation figure layer is read-only. It verifies the aggregate NPZ
byte counts and SHA-256 digests against their completion JSON files before
loading any data, and it does not modify the production results or runners.
Validate without plotting with:

```bash
python docs/make_note_figures.py --check-only
```

Regenerate the vector PDFs and 300-dpi PNGs with:

```bash
python docs/make_note_figures.py
```

The PDF figure backend embeds searchable TrueType outlines (`pdf.fonttype=42`)
while preserving the repository's CMU Sans Serif plotting style.

Build the note while keeping auxiliary files in `docs/build/`:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build flattened_domain_wall_mutual_information.tex
cp build/flattened_domain_wall_mutual_information.pdf .
```

## Wall-projected `Nx=20`, `Ny=60` ground state

`analyze_wall_projected_ground_state.py` isolates the boundary content of the
hard-wall, `nshell=1`, `alpha_1=1` flattened-parent ground state.  It uses the
additive Gaussian entropy contour of the full pure-state interval, rather than
the entropy of a narrow spatial window.  The latter has artificial extensive
entropy from tracing across its two x boundaries and is not a valid central-
charge estimator.

The analysis also resolves the two lowest-energy states at each `ky` into left-
and right-wall combinations.  Those spectral branches are marked as trusted
only while both states retain at least 50% of their norm in their own five-cell
wall window.  This avoids pretending that a unique two-band boundary model
exists after the edge states merge into the bulk continuum.

Run the deterministic analysis with:

```bash
python analyze_wall_projected_ground_state.py
```

It writes data, CSV tables, a JSON summary, and a double-column PDF/PNG figure
under `analysis_outputs/wall_projected_ground_state_nx20_ny60_v1/`.  The
entropy-contour convention gives `c_fit=1/2` per chiral wall and `c_fit=1` for
the two-wall cylinder.  The x-resolved wall correlator is compared with the
finite-ring chiral form `abs(cot(pi*d/Ny))`.
