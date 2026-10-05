# Fixed 60-cycle hard-wall channel evolution

Explicit-dynamics companion to `../spectral_gap_v1/`, not a replacement.
Run 16 cases: Nx=20; Ny=20,24,28,32,36,40,44,50; alpha_1=1,3.
Every case runs exactly **60 physical raster-y cycles**, including observations
at cycle zero, regardless of convergence. All previous spectral outputs are
preserved. No trajectory sampling or continuous Lindblad evolution is used.

The initialization follows the previous raster-y endpoint campaign:
full-system maximally mixed correlation C0=I/2 (engine raw matrix zero).
Keep alpha_2=30, nshell=1, inclusive slab x=5,...,15, hard support
truncation, all slabs active, X trial orbitals, periodic boundaries,
zero twist, complex128, perfect correction and measurement dephasing.
Within each cell the canonical order is Ap,Am,Bp,Bm. The code calls
`classA_U1FGTN.run_markov_channel` directly; the observer never updates state.

## Saved data and comparison

Each completed case has one checksummed `dynamics.npz` and a completion
JSON binding configuration, sources, spectral reference and timings.
Save all 61 cycle coordinates, charge, correlation RMS, successive-change
RMS, Hermiticity diagnostics, elapsed times, numerical-floor mask and local
logarithmic increment ratios; additionally save the full cycle-60 physical
correlation matrix and its occupation eigenvalues. No dense history is kept.
The saved eigenvalues of the correlation matrix are **not** channel multipliers.

Compare D_n=C_n-C_(n-1), which satisfies D_(n+1)=A D_n A^dagger, with
the prior spectral rate Delta_C=-2 log rho(A). Agreement of slopes requires
excitation of the leading mode and sufficiently late but numerically resolved
times. Nonnormal transients and initial-state overlaps can change the observed
finite-time rate. Dotted spectral guides are not fits, amplitude predictions,
or pointwise upper bounds. The RMS threshold 1e-13 censors plotting and local
rate diagnostics only: it does not stop evolution, clip data, or certify error.
Roundoff-limited late cycles must not be interpreted as a closing gap.

## Launch, resume and analyze

After checking that the selected CPUs are available:

```sh
python run_dynamics.py launch --cpus 0,7
python run_dynamics.py report --root results/RUN_STAMP
python analyze_dynamics.py --root results/RUN_STAMP
```

Two detached tmux workers, one alpha each, use one numerical-library thread
per worker. Outer case and inner cycle tqdm progress goes to per-worker and
per-case logs. To resume after both workers stop, repeat launch with the
same `--root`. Verified completed pairs are skipped. An interrupted case
restarts its 60 cycles; there is no mid-cycle or rolling state checkpoint.
Never launch overlapping workers on the same output directory.

The analyzer defaults to requiring all 16 results. `--allow-partial` writes
a separately labeled partial analysis. Outputs include a no-fit comparison
figure, a full 60-cycle numerical-floor diagnostic, CSV/JSON tables, captions
and a checksum manifest. Neither the original one-page spectral note nor
the manuscript is changed by this companion run.

Tests: `tests/test_explicit_60cycle_channel.py` checks the fixed grid,
canonical evolution against independent dense resets, passive observation,
all-cycle coverage, and checksum/completion resume rejection paths.

## Launched campaign

Run root: `results/20260929T193023Z/`. Two workers are pinned to CPUs 0 and 7
in tmux sessions `channel60_a1_20260929T193023Z` and
`channel60_a3_20260929T193023Z`. A one-shot companion waits for both worker
completion records and then generates the final figures/table automatically;
it writes `finalization.json` and `analysis.log`. Check these records instead
of assuming a vanished tmux session implies scientific completion.
