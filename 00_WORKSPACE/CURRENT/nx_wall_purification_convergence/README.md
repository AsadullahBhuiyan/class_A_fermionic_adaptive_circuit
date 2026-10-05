# Transverse-width wall-purification convergence

This campaign tests whether the slow maximally mixed wall-purification transient changes
when the standard domain walls are moved farther apart at fixed `Ny=20`.  It runs
`Nx=20,24,28` through the canonical CPU
`classA_U1FGTN.run_markov_circuit` entry point and never stores a covariance history.

The new result is intentionally separate from two earlier sources:

- the static B0 transverse-width calibration at `Ny=48`, which accepted `Nx=20`; and
- the cycle-resolved `Nx=16,20,24`, `Ny=20` Choi-transfer scan, which uses a different
  initial covariance and observable.

Those datasets appear only in a contextual figure. They are not pooled into the max-mix
bootstrap.

## Launch sequence

Run the three-cycle smoke profile first. It also runs the geometry, cycle-zero, and exact
interruption/resumption tests.

```bash
PROFILE=smoke ./launch_tmux.sh
```

Then launch 25 trajectories per width through cycle 100:

```bash
PROFILE=pilot ./launch_tmux.sh
```

The pilot automatically runs the analysis after all trajectories complete. If
`conclusion.json` classifies the width comparison as inconclusive, add trajectories
25--49 without changing the first 25 seeds or files:

```bash
PROFILE=topup TARGET_SAMPLES=50 ./launch_tmux.sh
```

If the wall upper confidence bound remains above `0.01` bits per cell at cycle 100,
extend every completed trajectory from its saved RNG/covariance state:

```bash
PROFILE=pilot TARGET_CYCLES=150 ./launch_tmux.sh
```

`CPU_LIST` accepts normal `taskset` syntax, `MAX_WORKERS` defaults to 48, and `WORKERS`
may be an explicit count or `auto`. Without `CPU_LIST`, the runner samples CPU load and
uses low-utilization CPUs from its scheduler affinity. Every worker is restricted to one
BLAS thread.

## Outputs

Raw outputs are written under
`results/campaigns/Nx20-24-28_Ny20_nsh1_dwtrunc1_init-maxmix_S25_C100/`.
Each trajectory has an atomic compressed observable file and an exact restart checkpoint.
The restart is retained after completion so the common cycle horizon can be extended.

Analysis products are written under the matching directory in `analysis_outputs/`:

- a double-column four-panel wall/bulk convergence figure;
- a log-scale global/absolute-difference figure;
- a contextual prior-evidence figure;
- fit, threshold, and contextual CSV tables;
- all bootstrap draws; and
- `conclusion.json`, which records the equivalence, dependence, top-up, or extension
  decision.

The five-cycle rolling mean is display-only. Fits, confidence intervals, sustained
thresholds, and the final decision use the unsmoothed trajectory data.

## Direct commands

The runner and analyzer can also be invoked without tmux:

```bash
python run_nx_wall_purification_convergence_cpu.py --samples 25 --cycles 100
python analyze_nx_wall_purification_convergence.py --bootstrap-count 2000
```

Run the maintained campaign tests with:

```bash
python -m pytest -q tests/test_nx_wall_purification_convergence.py
```
