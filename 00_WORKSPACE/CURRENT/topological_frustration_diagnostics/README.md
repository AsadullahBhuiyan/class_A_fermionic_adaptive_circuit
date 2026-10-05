# Topological Frustration Diagnostics

CPU reference implementation of static controller compatibility, trajectory activity and empirical full-counting statistics, Lyapunov and regularized-Choi slow-mode proxies, and paired domain-wall charge response.

All circuit simulations call `classA_U1FGTN.run_markov_circuit(...)`. The hard-wall case uses `DW=True`, `dw_truncation=True`, `meas_slab_only=True`, `alpha_1=1`, and `alpha_2=30`. Hard truncation excludes the exterior controller modes, so `alpha_2` is retained as the canonical geometry parameter rather than treated as a physics sweep.

Run the complete small-system validation campaign with:

```bash
python topological_frustration_diagnostics/run_cpu.py all --smoke --output-dir /tmp/frustration_smoke
```

Launch the gated characterization sequence with up to 80 CPUs using:

```bash
python topological_frustration_diagnostics/launch_campaign.py \
  --campaign-root topological_frustration_diagnostics/results/campaign_001 \
  --cpu-budget 80 --workers auto --threads-per-worker auto
```

The launcher runs tests, smoke checks, spectral cross-checks, static cases,
utilization benchmarks, pilots, production trajectories, and final analysis in
that order. A failed validation stops the sequence and cannot publish a stage
success marker. Independent geometries and trajectories use `joblib` process
workers; linear-algebra threads are capped per worker to avoid oversubscription.
The parent process merges results deterministically and is the only process that
writes output files. Resource decisions and per-task timings are persisted in
`run_summary.json`.

Run an individual diagnostic with `static`, `activity`, `spectral`, or `response`. Regenerate figures from existing outputs with:

```bash
python topological_frustration_diagnostics/run_cpu.py analyze --input-dir PATH
```

Without an explicit `--nshell`, every CLI mode uses `nshell=1`. Supply `--nshell 1 2` for an explicit cutoff-convergence campaign or `--nshell none` in static mode for the untruncated-frame control. Response mode runs randomized order as the primary sequence and `raster_y` as the ordering-bias control; an explicit `--sequence` selects only that order. Large CPU campaigns should be launched only after the smoke suite passes.

Every mode accepts `--cpu-budget`, `--workers`, `--threads-per-worker`, and
`--memory-fraction`; `--no-parallel` provides the serial reference path.
