# Spectral-only hard-wall channel gap

This independent campaign constructs one raster-y product `A=Q_M...Q_1`
from canonical normalized CPU OW vectors and diagonalizes it. It never
calls a dynamics routine. The represented dynamics is the perfect-correction,
number-dephasing branch of `classA_U1FGTN.run_markov_channel`.

Grid: alpha_1=1,3; Nx=20; Ny=20,24,28,32,36,40,44,50; alpha_2=30;
nshell=1; hard support truncation; inclusive slab x=5,...,15; all slabs
active; periodic x/y; zero twist; X trial orbitals; complex128.
Within each cell the order is Ap,Am,Bp,Bm; y advances before x.

The covariance multiplier radius is `rho(A)^2`; the covariance decay rate
is `-2 log rho(A)` per complete cycle. The full mixing-channel decay rate
cannot exceed this rate in the same physical symmetry sector. This is not
an occupation-spectrum gap, a singular-value estimate, an inferred
trajectory Lyapunov exponent, or a continuous-Lindblad rate.

After checking CPU utilization, from this directory:

```sh
python run_sweep.py launch --cpus 0,7
python run_sweep.py report --root results/RUN_STAMP
python analyze_sweep.py --root results/RUN_STAMP
```

The launch creates two detached tmux sessions, one alpha per CPU. Each
worker uses single-threaded numerical libraries and completes its eight
sizes in increasing order. Case logs expose model configuration, product
construction progress and eigensolver boundaries; worker logs expose case
progress. No existing session or data is modified. To resume, supply the
same `--root` to `launch` after the previous workers have stopped.
Only checksum-, source-, and configuration-matching result/receipt pairs
are skipped. Failed cases remain pending and get a traceback file.

Each case stores the complete eigenvalue spectrum, dominant left/right
eigenvectors, dominant eigenvalue, spectral radius, raw covariance decay
rate, multiplier gap, schedule word, diagnostics and provenance. No dense
product or covariance history is saved. Independent full-vector projector
sweeps check the compact-support product and dominant eigenpair. The
left/right overlap helps flag eigenvalue sensitivity in this non-normal
problem; these are floating-point diagnostics, not certified intervals.
Unit-modulus modes within 1e-10 are explicitly unresolved; they are not
discarded to pick a faster, positive rate.

The analyzer requires all 16 verified cases and produces a CSV/Markdown
table, vector PDF and 300-dpi PNG plot, caption, and one-page RevTeX source
with measured results. Compile `gap_note.tex` from the run's `analysis`
directory. The requested one-page note deliberately omits a table of
contents and keeps the plot separate. The main technical report is untouched.

## Completed run

The canonical completed run is `results/20260929T191448Z/`: 16/16 verified
cases, computed with single-threaded workers pinned to CPUs 0 and 7.
Restart verification skipped all eight cases for each alpha without another
eigensolve. Five tests in `tests/test_ordered_channel_spectral_gap.py` pass.
The comparison figure, CSV/Markdown table, checksummed analysis manifest,
and compiled one-page `gap_note.pdf` are in its `analysis/` directory.

From Ny=20 to 50, Delta_C decreases from 1.904385 to 1.829759 for alpha_1=1
and from 3.250052 to 3.183716 for alpha_1=3 (inverse cycles). These are
covariance relaxation rates and upper bounds on the full mixing-channel
rate, not evidence by themselves for a nonzero many-body gap.

The initial `results/20260929T191100Z/` run is preserved for provenance.
Its spectra preceded a report-only NumPy-boolean serialization fix; the
canonical run repeats the calculation with the final runner source hash.
No earlier campaign outputs or running jobs were changed.
