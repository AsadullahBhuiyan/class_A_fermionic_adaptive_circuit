# CPU CFT Extraction

This folder contains a self-contained local CPU workflow for extracting
trajectory free-energy and Lyapunov finite-size observables from the canonical
CPU dynamics engine.

## Run

Smoke test:

```bash
python cpu_cft_extraction/run_cpu_cft_sweep.py --smoke
```

Parallel smoke test:

```bash
python cpu_cft_extraction/run_cpu_cft_sweep.py --smoke --sample-workers 2 --blas-threads 1
```

Core-count benchmark:

```bash
python cpu_cft_extraction/run_cpu_cft_sweep.py \
  --benchmark-workers 8 14 20 28 40 56 \
  --benchmark-samples 10 \
  --benchmark-ny 20 \
  --benchmark-cycles 4 \
  --blas-threads 1
```

Default local production run:

```bash
python cpu_cft_extraction/run_cpu_cft_sweep.py
```

Useful overrides:

```bash
python cpu_cft_extraction/run_cpu_cft_sweep.py \
  --nx 20 \
  --ny 20 30 40 \
  --samples 10 \
  --cycles-factor 2 \
  --alpha 1 \
  --sample-workers 28 \
  --blas-threads 1
```

The driver calls `classA_U1FGTN.run_markov_circuit(...)` with
`postselect=False`, `perfect_correction=True`, `track_choi=True`, the
trajectory-weight observer, and the tangent Lyapunov observer.  Because the CPU
engine requires those observers to run serially inside one engine call,
`--sample-workers` parallelizes externally across all requested `(Ny, sample)`
tasks: each process runs one sample with `samples=1` and the parent merges the
outputs by size.  For the default `Ny=20 30 40`, `samples=10` campaign this
gives 30 independent tasks, so `--sample-workers 28` can keep roughly one
socket's worth of physical cores busy.  Keep `--blas-threads 1` unless
benchmarking shows otherwise.

## Output

Each run creates `outputs/<campaign_id>/` with:

- `manifest.json`: run parameters and estimator conventions.
- `benchmark_<timestamp>/benchmark_summary.{csv,json}`: worker-count timing
  reports when `--benchmark-workers` is used.
- `scalars_by_size.csv`: per-size estimates for `f0`, one-body tangent/Choi
  gaps, additive tangent/Choi Fock gaps, endpoint/null counts, and size-resolved
  Fock-sector slopes.
- `fit_summary.json`: finite-size fits for `c_eff`, `x_tangent_fock`, and
  `x_choi_fock`.  The Fock rank is set by `--fock-rank` and defaults to one
  finite quasiparticle.
- `N<Nx>x<Ny>/trajectory_weights.csv`: cycle-resolved `-log p_xi`.
- `N<Nx>x<Ny>/tangent_lyapunov.npz`: QR-stabilized tangent spectra.
- `N<Nx>x<Ny>/choi_rapidity.npz`: Choi eigenvalues, finite rapidities, gap
  estimates, active/censored flags, and endpoint counts.

`--fock-rank` is a convenience for the additive low-lying tower: rank `r` sums
the `r` smallest finite positive one-body excitation rapidities.  A precise
operator-sector extraction may require a different particle-hole replacement or
fixed-charge excitation pattern.

The notebook `analyze_cpu_cft_extraction.ipynb` reads a campaign folder and
follows the default structure in `ANALYSIS_NOTEBOOK_TEMPLATE.md`: a short theory
and estimator summary at the beginning, an executable campaign/run-parameter
summary, clear markdown headers between analysis stages, one output figure per
editable plotting cell, and a final raw diagnostics section.
