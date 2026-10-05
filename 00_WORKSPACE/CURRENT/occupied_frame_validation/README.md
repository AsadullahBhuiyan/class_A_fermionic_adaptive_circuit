# Occupied-frame validation campaign

This bundle validates and benchmarks the frame-native CPU state adapter in the canonical `classA_U1FGTN.run_markov_circuit(...)` loop. It compares paired replays of the existing rank-one covariance backend with:

- a physical occupied frame for random pure states; and
- a doubled purification frame for the maximally mixed state.

The immutable configuration is [`campaign_config.v1.json`](campaign_config.v1.json). Its first production geometry is `Nx=Ny=16`, 32 cycles, `nshell=1`, ten paired records per initialization family, complex128, `DW=False`, `alpha_1=alpha_2=1`, and root seed `2026082201`. This uniform controller is a physics gate: every one of the four initialization/backend lanes must converge toward the same Chern-insulator target before performance benchmarking is allowed.

## Launch

Inspect the allocation and command without launching:

```bash
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh --dry-run
```

Run only initialization and the regression preflight in a detached tmux session:

```bash
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh --preflight-only
```

Launch the full campaign:

```bash
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh
```

Launch a larger square system; cycles, half-system cut, physics window, and checkpoint schedule scale automatically with `2*Ny`:

```bash
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh --size 18
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh --size 20
```

The launcher prints the tmux session, log, and timestamped output directory. It waits until it can select ten idle, non-sibling physical cores from one NUMA node. Selection can be overridden explicitly:

```bash
CPU_LIST=0,2,4,6,8,10,12,14,16,18 \
  00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh
```

`ALLOW_BUSY=1` disables only the utilization threshold; physical-core and NUMA validation remain enforced. All BLAS/OpenMP thread counts are locked to one.

Resume verified sample shards after interruption:

```bash
00_WORKSPACE/CURRENT/occupied_frame_validation/launch_tmux.sh \
  --resume YYYYMMDDTHHMMSSZ
```

Each shard is atomically replaced and marked complete only after its configuration/task identity and product hashes are recorded. An incomplete shard is rerun from its paired record.

## Stages

The full entrypoint executes:

```text
init -> preflight -> records -> correctness-gate
     -> benchmark -> analyze -> validate -> report
```

The two initialization families run sequentially. Each family launches exactly ten external single-sample workers. Even samples execute covariance then frame; odd samples execute frame then covariance. A process barrier precedes each backend phase, yielding a five/five backend mix at equal ten-way concurrency. Each worker performs a tiny untimed warm-up of both kernels before benchmark timing.

If either numerical correctness or uniform-insulator convergence fails, benchmark execution is skipped. Failure analysis, aggregate products, validation status, and the RevTeX note are still generated before the campaign exits nonzero. Failure shards include exact covariance and frame states replayed to the first localized failing channel when possible.

Individual stages can also be invoked directly:

```bash
python 00_WORKSPACE/CURRENT/occupied_frame_validation/run_campaign.py records \
  --campaign-id CAMPAIGN_ID \
  --cpu-list 0,2,4,6,8,10,12,14,16,18
```

## Products

Every timestamped result directory contains:

- `manifest.json`: seeds, affinity history, environment, Git state, canonical entry point, source hashes, and stage status;
- `raw/records/`: authoritative schedules, outcomes, branch weights, and seeded cycle-zero states;
- `raw/correctness/`: cycle-resolved real-space Chern estimators for all four lanes, regional/global entropies, declared full-state checkpoints, dense-Choi comparisons, detailed timings, and failure states;
- `raw/benchmark/`: minimally instrumented paired wall/CPU/cycle timings and memory fields;
- `processed/arrays/`: aggregate NPZ arrays retaining the sample and cycle axes;
- `processed/tables/`: per-sample, per-cycle, timing-breakdown, and benchmark CSV tables;
- `figures/`: four-lane Chern convergence, error, entropy, detailed-timing, speedup, memory/throughput, and cycle-timing figures in PNG/PDF form;
- `reports/occupied_frame_validation.tex`: the RevTeX validation note, plus a PDF when `latexmk` and RevTeX are installed;
- `status/` and `logs/`: machine-readable stage outcomes and complete launch/analysis logs.

The update-only benchmark uses `G_history=False`, returns the native frame, installs no state observer, and never materializes `FF^dagger`, `VV^dagger`, or `BB^dagger`. Pure-frame factorization and maximally-mixed frame allocation are setup timings rather than dynamics timings.

## Acceptance gates

The numerical hard ceilings are locked in the configuration: covariance `1e-8`, branch probability `1e-8`, cumulative log weight `1e-7`, entropy `1e-7`, normalized Gram residual `1e-9`, and doubled-Choi checkpoints `1e-7`. Any branch disagreement, nonfinite result, dense covariance fallback, missing cycle/sample, or frame-rank inconsistency is a hard failure.

The separate physics gate calibrates the disk-partition real-space Chern estimator on the finite-size uniform ground-state projector (approximately `0.999465` at 16×16). In all four lanes, the ensemble terminal estimator must be within `0.25` of that target, improve relative to cycle zero, and drift by no more than `0.15` between cycles 24 and 32. The maximally mixed lanes must additionally reduce global entropy and end below global entropy density `0.10`.

The legacy doubled-Choi evolution is exceptionally expensive because it must be propagated through every channel even though it is saved only at checkpoints. It is therefore a designated algebraic audit on maximally mixed sample 0 (`dense_choi_audit_samples: [0]`); all ten maximally mixed samples still undergo full physical-covariance, branch, entropy, Chern, Gram, and rank-word comparison at every declared cycle.

Performance classification uses each paired record as the independent unit, 10,000 paired bootstrap resamples, and a predeclared ±5% practical-equivalence band. Performance is descriptive; only correctness determines implementation validity.

For statistics, the ten independently seeded trajectory records are the independent units within each initialization family. Covariance and frame backends replay the same record, so all backend comparisons are paired. Reported speed summaries include the ten paired ratios and a paired bootstrap interval obtained by resampling whole record pairs; cycles, channels, and individual measurement events are not treated as independent samples. The two initialization families are separate ensembles. Runs at different sizes reuse the deterministic seed labels, which permits later common-seed size comparisons, although each size still generates its own dynamics and branch record.
