# Imaginary-time density correlations of the flattened-parent ground state

One deterministic, CPU-capable Colab calculation: Nx=20, Ny=30, hard walls
at x=5,15, alpha1=1, alpha2=30, nshell=1, trial orbital X, complex128,
exact half filling (600 occupied single-particle modes). There are no
trajectories, Monte Carlo errors, purification cycles, or statistical fits.
This bundle overrides the active parent's generic A100/Markov-runner defaults:
it constructs an equilibrium parent through canonical CPU
`classA_U1FGTN.construct_OW_projectors`, not circuit simulation.

## Meaning of the measurement

The signed OW parent is the sum of upper-band projectors minus lower-band
projectors. It is **not** spectrally flattened again to +/-1. Its thirty
40-by-40 momentum blocks retain the wall dispersion. The 1e-7 occupation
regulator reproduces the spatial benchmark's convention: representative OW
columns are transformed at k+phi/Ny, with untwisted periodic Bloch labels
retained when reconstructing the state. Both filling and propagation use
this same regulated Hamiltonian. This is not a flux-threading experiment.

For an observable O, its ground-state connected imaginary-time autocorrelation
is a sum of nonnegative occupied-to-empty transition weights times
exp[-tau*(epsilon_empty-epsilon_occupied)]. Local cell density sums both
orbitals, forms the autocorrelation at the same cell, then averages all y.
Column charge sums y first, forms its autocorrelation, and divides by Ny.
They are different observables. The equal-time values are their intrinsic
quantum variances, not variances across measurement outcomes.

The spatial benchmark's positive quantity is half the squared two-point
matrix norm. At *distinct* cells the physical connected density correlation
is minus twice that quantity. At contact it additionally contains the mean
cell density. Both spatial definitions are saved to make this explicit.

Imaginary time is inverse unrescaled parent-energy, **not circuit cycles**.
The calculation inserts two density operators; it does not simply evolve
and renormalize a density matrix under a postselection filter. `theory.tex`
derives these distinctions and the Gaussian particle-hole formula.

## Run

For local execution (two BLAS threads):

```sh
python run_campaign.py --threads 2
python run_campaign.py --report-only
```

Defaults are `results/` for durable outputs and `scratch/` for local temporary
files. Scientific settings are exposed but locked to the named revision.
In Colab, place this folder's executable files and `src/` under
`MyDrive/final_production_new_designs/24_flattened_imaginary_time_density/`
and open `run_flattened_imaginary_time_density.ipynb`. **No Drive upload has
been performed as part of implementation.** A CPU runtime suffices.

The notebook mounts once, stages just executable sources under `/content`,
streams text output and tqdm safely through Colab's OutStream, and exposes
REPORT_ONLY, CPU_THREADS, paths and the full scientific config. Its last cell
disconnects the runtime. It includes one editable plot cell per figure.

One checksum-bound NPZ/completion-JSON pair is the resume unit. On restart,
matching config/source identities, filename, size and SHA-256 must verify
before computation is skipped. Missing/invalid pairs are recomputed. Files
are written in local scratch, copied to a Drive temporary path, reopened,
checked, atomically replaced, and checked again; completion JSON is last.
This verifies **DriveFS readback**, not independent cloud visibility. Use one
writer. Interrupted calculations restart; this small deterministic task has
no scientific checkpoint. Figures are reproducible derivatives and can be
regenerated from the verified NPZ.

## Saved products

- `tau`: zero plus 240 logarithmic points from 1e-3 to 1e3.
- `energies`, `eigenvectors`, `occupied`, `parent_blocks`: momentum-space
  eigensystem, assignments, and the exact regulated parent used throughout.
- `local`, `column`: shape (241,20); the latter is already divided by Ny.
- `local_normalized`, `column_normalized`: divided by the corresponding
  equal-time value; variance <=1e-12 is explicitly undefined (NaN), with
  Boolean validity masks. Raw values are never floored.
- Cell/column means, equal-time variance references, `spatial_connected`
  and `spatial_legacy_positive` (x,separation), config/source identities and
  numerical diagnostics. No covariance histories or many-body matrices.
- `figures/`: local log-log, local semilog, column log-log, and normalized
  x-versus-tau map; each 3.375-by-2.6-inch vector PDF and 300-dpi PNG.
  Also `curves.csv` and a figure hash manifest. Curves show walls x=5,15
  and trivial bulk x=2,18. Exact zeros are omitted on log axes. The map's
  color range saturates below 1e-12, without changing the stored data.
  Curve figures show the range 1e-12 to 1 for readability; smaller raw values
  remain in the NPZ/CSV and are not floored. The notebook axis limits are editable.

No automatic power-law interpretation: short-time microscopic structure,
finite-size crossover, and weak transition matrix elements must be inspected
before selecting a fit window. A tiny energy gap alone need not control a
given density autocorrelation if that transition has negligible weight.

## Validation and rebuilding

From the repository root: `pytest -q tests/test_flattened_imaginary_time_density.py`.
Tests compare regulated momentum blocks with a directly constructed dense
parent, density correlators with dense and small exact Fock-space references,
equal-time variances, charge conservation, normalization and monotonicity,
completion skipping, corruption/partial pairs and failed readback.
The notebook is schema-validated and its code cells compile; canonical CPU
source copies are checked byte-for-byte. `python build_notebook.py` regenerates
the notebook/config. Scientific source files are not changed by this command.

## Completed local benchmark (2026-09-27)

The full default calculation has been run locally with two BLAS threads:
7.02 seconds for construction/correlations/validation and about 4--5 seconds
for the figures. The verified NPZ is 893,811 bytes. A second execution skipped
the completed calculation and regenerated only derivative figures. These are
local timings, not a measured Colab runtime. The notebook has been schema-
validated and compiled cell-by-cell, but not executed on a live Colab runtime.

- 600 occupied modes, 20 at every momentum; projector idempotency error
  2.0e-15; total-number transition weight 8.72e-29.
- Ground-state projector matches the existing regulated spatial-benchmark
  implementation exactly on this same Nx20, Ny30 input (maximum difference 0).
- Half-filling gap 1.410739e-7. The minimum transition's local density weight
  at either wall is 3.142319e-8. Thus the long-time wall plateau over the saved
  time range is consistent with this weak, extremely slow finite-size
  transition; it is not evidence for an asymptotic power law.
- Eight new tests and two active layout tests pass (10/10); all four PDFs
  have the specified dimensions and their derivative checksums verify.
- A broader legacy run gave 13 passes and four unrelated failures: bundle 06
  has stale historical source/completion identities, and two old queue tests
  refer to removed `colab_bundle_runner.py` files. Those campaigns were not
  modified to repair their historical provenance.
- The two-page RevTeX note compiles to `theory.pdf` without overfull boxes.

No Google Drive deployment was attempted; existing scientific data and
manuscript figures were left unchanged.
