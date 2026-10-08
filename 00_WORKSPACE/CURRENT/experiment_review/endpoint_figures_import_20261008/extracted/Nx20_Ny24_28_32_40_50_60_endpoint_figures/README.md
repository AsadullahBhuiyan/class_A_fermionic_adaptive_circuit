# Endpoint entropy, charge variance, and wall entropy figures

This is a self-contained data and figure-reproduction package for the fresh
Nx=20, Ny=24,28,32,40,50,60 campaign. Both figures use the same 600 independent
trajectories: 100 per circumference. All 120 five-trajectory shards passed
the original analysis verification. No new simulation is needed to make the
figures from this package.

## Quick start: regenerate the figures on a CPU

Extract the ZIP and open a terminal in this directory. Python dependencies are
NumPy, Matplotlib, Torch, and tqdm. `requirements.txt` lists the versions used
by the installed analysis environment when this archive was created; a CPU-only
Torch installation can analyze these data. `environment.json` records Python
and package versions. The analysis imports the archived dynamics module for
source identity checking, but does not run dynamics or require a GPU.

The original manuscript renderer requires `latex`, `dvipng`, and `pdffonts`
(Poppler) on PATH. The TeX installation must provide Computer Modern fonts,
amsmath, amssymb, bm, and the packages required by Matplotlib's LaTeX backend
(including type1cm/type1ec). The original run used TeX Live 2023 and Poppler.

```bash
python regenerate_figures.py
```

The outputs go into `regenerated_analysis/`, leaving the supplied `analysis/`
reference outputs untouched. To verify data and recompute curves and fits
without TeX, Poppler, or figure rendering:

```bash
python regenerate_figures.py --data-only
```

Equivalent direct command, from this directory:

```bash
python source/analyze_campaign.py --output-root . --output-dir regenerated_analysis
```

Unset CLASSA_ENGINE_DIR if using the direct command; the convenience wrapper
does this automatically so the bundled engine is used. Keep `source/` unchanged:
configuration, source, schedule, and shard identities are verified by the
original analysis code. Regeneration writes new provenance/font reports.
PDF metadata and toolchain versions can change PDF bytes; the scientific
curves and fitted results should reproduce the supplied CSV and JSON.

To check every packaged file's SHA-256 before analysis:

```bash
python verify_archive.py
```

## Package contents

- `results/NyNNN/samples_000-004.npz`, etc.: all 120 endpoint NPZ shards and
  their completion JSON receipts, covering IDs 0..99 exactly once for each Ny.
- `analysis/`: the original two vector PDFs, their 300-dpi PNGs, `curves.csv`,
  `fits.json`, full analysis completion/provenance report, and typography reports.
- `source/`: exact frozen observer, analysis, storage, scheduling, production,
  benchmark, validation, submission, typography, and canonical engine sources.
- `manifest.json`, `schedule.json`, `metadata/`, worker receipts, submission
  and resource records: configuration, seeds, source identity, actual task
  assignment, and completed run provenance.
- `validation/gpu_complete.json`: GPU versus CPU parity and exact-resume results.
- `benchmarks/NyNNN/complete.json`: measured batch choices and timing evidence.
- `SHA256SUMS.json`: checksums and sizes for all packaged files except itself.

No temporary native-state checkpoints, histories, spectra, or individual-origin
products are needed to reproduce these figures. They were not retained as
scientific endpoint outputs. Benchmark trajectories are separate from the
production ensemble. The frozen `source/README.md` and Slurm templates describe
the original submission request; `resource_request.json` records the operational
adjustment to 56 GiB host memory per GPU worker under the shared account quota.

## Simulation protocol

Nx=20; Ny in {24,28,32,40,50,60}; 100 trajectories per size. Endpoint cycles
are respectively 48,56,64,80,100,120 (T=2Ny). Geometry is periodic. There are
two physical orbitals per unit cell. Each trajectory begins as a random pure
half-filled state, followed by Born-conditioned product preparation of the
exterior before cycle zero. Preparation outcomes and circuit measurement
outcomes follow the Born rule; there is no postselection.

The campaign uses hard support-truncated domain walls at x=5,15, alpha_1=1,
alpha_2=30, n_shell=1, trial_orbitals="X", filling_frac=n_a=0.5, slab-only
measurements, raster_y ordering, and perfect correction. The exact serialized
configuration is in `source/campaign_config.json` and `manifest.json`.

Dynamics calls the canonical `classA_U1FGTN_gpu.run_markov_circuit` entry point
using complex128 native occupied frames, with no covariance materialization,
no Choi tracking, and frame reorthonormalization interval 1. GPU workers ran
on NVIDIA B200s. The root seed is 2026100733; per-task seeds and sample ranges
are frozen in `schedule.json`. The selected schedule assigned Ny=50,40,32,28,24
to worker 0 and Ny=60 to worker 1, with 100 trajectories per execution batch.
Dynamics checkpoints were taken every five cycles and endpoint progress after
every completed width. Completed checkpoints were removed after verified
publication. The canonical engine sources are preserved byte-for-byte.

## Endpoint observables and strip coordinates

For every trajectory and each Ay=1..Ny/2, calculate a full-x strip independently
at every periodic origin y0=0..Ny-1. The strip contains all x=0..19 and
y=(y0+dy) mod Ny for dy=0..Ay-1, including both orbitals mu=0,1. The physical
basis index is i=2*(y*Nx+x)+mu.

Let R be the native occupied frame. Restrict its rows to the strip and form
the single-particle occupied projector C_A=R_A R_A^dagger. The endpoint solver
Hermitian-symmetrizes C_A and diagonalizes it as U diag(nu) U^dagger. Eigenvalues
outside [-1e-8,1+1e-8] fail validation; accepted roundoff excursions are clamped
to [0,1]. Natural logarithms are used, and h(0)=h(1)=0:

    h(nu) = -nu*ln(nu) - (1-nu)*ln(1-nu)
    S_1(A) = sum_j h(nu_j)
    F_A = sum_j nu_j*(1-nu_j)
    s_i(A) = sum_j |U_ij|^2 * h(nu_j)

F_A is the intrinsic quantum variance of particle number within a trajectory's
strip, not the variance of mean charge across trajectories. The cell contour
is s(x,dy)=sum_mu s_(x,dy,mu). It is nonnegative and sums to S_1(A).

Only after these nonlinear observables are evaluated separately for each strip
are all Ny origins averaged within the trajectory. Contours are aligned by
relative dy before averaging. Independent trajectories are then kept separate.
The code never calculates entropy of the ensemble-averaged correlation matrix.

Each NPZ contains float64 scientific arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `endpoint__contour_von_neumann_y0avg` | (5, Ny/2+1, 20, Ny/2) | Per-trajectory origin-averaged contour, axes (sample, Ay, x, dy) |
| `endpoint__entropy_von_neumann` | (5, Ny/2+1) | Per-trajectory origin-averaged full-strip S_1 |
| `endpoint__charge_variance` | (5, Ny/2+1) | Per-trajectory origin-averaged full-strip F_A |

Width index equals Ay. Ay=0 is exactly zero. At width Ay, contour entries with
dy>=Ay are exactly zero padding. Metadata includes sample_ids, ay_values,
valid_dy_count, endpoint_cycle, origin_average_count, contour_coordinate, Nx, Ny.
Load with `numpy.load(path, allow_pickle=False)`. The JSON beside every NPZ
binds its byte count/checksum to configuration, source, seed, and sample IDs.

## The four plotted quantities

`Figure_06_entropy_charge`: full-strip S_1 in panel (a), F_A in panel (b).
`Figure_09_wall_entropy`: contour weight integrated over x={5,6} in panel (a)
and x={14,15} in panel (b):

    S_L^wall(Ay) = sum_(x in {5,6}) sum_dy s(x,dy;Ay)
    S_R^wall(Ay) = sum_(x in {14,15}) sum_dy s(x,dy;Ay)

These wall values are contributions to the full-strip entropy contour. They
are not entropies of isolated two-column subsystems. Integrating the saved
origin-averaged contours is equivalent to integrating each origin and then
averaging, since this step is linear.

For each observable Y and each individual trajectory a, first subtract its
own half-strip value:

    delta_Y_(n,a)(Ay) = Y_(n,a)(Ay) - Y_(n,a)(n/2),  n=Ny.

The plotted ordinate is the mean of these deltas across the 100 trajectories.
Error bars are their sample standard deviation (ddof=1) divided by sqrt(100).
Periodic origins are not counted as additional independent samples.
The horizontal coordinate is the logarithmic chord-length ratio:

    D_n(Ay) = (n/pi)*sin(pi*Ay/n)
    X_n(Ay) = ln[D_n(Ay)/D_n(n/2)] = ln[sin(pi*Ay/n)].

The half-strip point is exactly (0,0). Ay=1 is omitted for display only.
Stored data retain Ay=0,1 and every requested width.

## Joint fit, uncertainty, and reported coefficients

Use only 8<=Ay<=n/2 and fit through the origin, delta_Y=m*X. Every size has
equal total fit weight. If W_n is that size's fit widths, L_n=|W_n| and
w_n=1/L_n, then

    B = sum_n w_n * sum_(Ay in W_n) X_n(Ay)^2
    m = [sum_n w_n * sum_(Ay in W_n) X_n(Ay)*mean(delta_Y_n(Ay))]/B.

The coefficients are c=3*m_S, k=pi^2*m_F, and m_L,m_R for the wall panels.
The plotted entropy annotation calls the coefficient c_1.

The uncertainty calculation preserves all covariance between widths and the
shared anchor. Define the linear sample contribution

    z_(n,a) = (w_n/B)*sum_(Ay in W_n) X_n(Ay)*delta_Y_(n,a)(Ay).

Independent size ensembles give Var(m)=sum_n sampleVar_a(z_(n,a))/100.
The reported slope SEM is sqrt(Var(m)); coefficient SEMs multiply by the same
prefactors 3 or pi^2. This is not a fit based on independently weighted widths
or inverse-SEM weights. `R0_squared` is the uncentered statistic
1 - weighted squared residual / weighted squared mean ordinate.

Measured results (errors are trajectory SEMs):

| Quantity | Estimate | SEM |
| --- | ---: | ---: |
| c=3*m_S | 1.0438039361006386 | 0.0021648748233368937 |
| k=pi^2*m_F | 1.0418477048532033 | 0.002209121464143072 |
| m_L | 0.17263737667318393 | 0.0005272805430995304 |
| m_R | 0.16332295051317078 | 0.0005624163328670958 |

`analysis/curves.csv` contains per-size/per-width raw and anchored means and
SEMs, X, and fit-window membership for all four observables (Ay>=1).
`analysis/fits.json` contains full-precision slopes, coefficients, uncertainties,
prefactors, fit window, and uncentered goodness-of-fit values. The NPZ data,
rather than just the CSV SEMs, allow refitting with width covariance preserved.

## Validation and provenance

The original analysis verifies all 120 checksummed shards, IDs 0..99 once per
size, geometry, cycles, all widths, float64 storage, finite values, nonnegative
contours, exact zero width/padding, entropy/variance subsystem bounds, and
contour-entropy closure within 2e-8. GPU endpoint parity with an independent
NumPy CPU reference had maximum absolute error 7.105427357601002e-15. Actual
interruption/resume tests passed for native frame/rank/RNG state after cycle 5
and for endpoint progress resumed after width 1. See the validation receipt.
Both original PDFs passed embedded Computer Modern/AMS font and clipping checks.

Configuration SHA-256:
`c277ffb192ef4a88d6e30a3bd2ca38815ecdcf3dddebf816ac84e7013ac16ce5`

Frozen source SHA-256 (digest of the recorded source file hash dictionary):
`42a16fb1523f363cd2f0bcd0c0d8197a1d9a62cf2aac5ed4cf2a5dce4eb7eac2`

Frozen schedule SHA-256:
`5186c707cfc419a398d4034c18952ec58601ed585531e6d937c6297b567be1bf`

Historical absolute paths inside provenance refer to the production cluster.
The figure regeneration commands use only this extracted package.
