# Max-mix many-body Lyapunov campaign through 4Ny

This standalone Colab bundle acquires the long-depth Gaussian transfer-spectrum
data needed for a many-body Lyapunov/CFT finite-size analysis. The current
campaign uses two parallel hard-wall lanes with one locked scientific
configuration and disjoint circumference sets.

## Locked scientific contract

- `Nx=20`; `Ny=20,24,30,36,44,56,60`.
- 100 independent Born trajectories for every size.
- Hard/support-truncated domain walls at `x=5,15`.
- `nshell=1`, `alpha_1=1`, `alpha_2=30`, maximally mixed initialization,
  `raster_y`, perfect correction, no postselection, and `T=4Ny`.
- Canonical `classA_U1FGTN_gpu.run_markov_circuit`, covariance state,
  `complex128`, and an NVIDIA A100 with at least 35 GiB detected memory.

The revision is
`maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled`.
Outputs go to
`MyDrive/classA_final_production_outputs/<revision>`.

## Which notebooks to run

- Lane A: `run_hard_wall_manybody_lyapunov_4ny_lane_a.ipynb` runs
  `Ny=60,44,20`.
- Lane B: `run_hard_wall_manybody_lyapunov_4ny_lane_b.ipynb` runs
  `Ny=56,36,30,24`.

Run lanes A and B in two separate A100 runtimes. Their Ny sets are disjoint, so
they share the same v4 output collection without writing the same result or
checkpoint paths. Do not run the old full hard-wall notebook concurrently.
Lane A preserves the exact scientific source/configuration identity and resumes
the existing `hard_Ny060_exec000_samples000-034` checkpoint. No soft-wall lane
is part of the current campaign.

Both lane notebooks mount Drive once, show the entire editable configuration in
one cell, copy the bundle to `/content`, and execute the staged runner inside the
notebook kernel so `tqdm.auto` uses Colab's native progress display. They
disconnect only in the final explicit cell.
Set `MAX_NEW_EXECUTION_BATCHES=1` for a worst-size timing test.  Set
`REPORT_ONLY=True` to inspect completed shards and checkpoints without running
new trajectories.

## Batching and resume

The A100 resident batch sizes are 100 trajectories at Ny=20,24,30; 90/10 at
Ny=36; 60/40 at Ny=44; 40/40/20 at Ny=56; and 35/35/30 at Ny=60. Sizes run
largest first. The large-size choices approximately bound
`resident_trajectories * Ny^2 <= 126000`, compared with 144000 for the failed
Ny=60 batch of 40.
PyTorch's allocator is hard-limited to 38 GiB,
and the runner prints peak reserved GPU memory after every execution batch.
Spectrum diagonalization remains separately chunked to limit eigensolver
workspace.

Each completed execution batch is split into checksum-verified five-trajectory
NPZ result shards.  During a batch, one rolling full-state/RNG/observer
`checkpoint.npz` plus `checkpoint.json` is published every 10 cycles using the
same temporary-copy, readback, and atomic-replace protocol.  Restart therefore
resumes a long resident batch from its last verified ten-cycle boundary.  No
Drive API, lease, archive, dashboard,
migration, or server-manifest layer is used.

The v1 output and its 20-trajectory Ny=60 checkpoint remain immutable. The
superseded v2 attempt placed 40 Ny=60 trajectories on the GPU, reached
27.94 GiB, and then needed another 3.43 GiB for a conditional-update workspace;
the bundle's own 30-GiB allocator ceiling therefore stopped it before cycle 1,
and it saved no v2 result or checkpoint. V3 raised the explicit ceiling to
38 GiB but retained the 40-trajectory Ny=60 batch. After resuming its cycle-10
checkpoint, that batch held 34.81 GiB and requested another 3.43 GiB, exceeding
the ceiling; it saved no completed result shard. V4 keeps the 38-GiB safety
ceiling and uses the memory-scaled resident batches above. It uses a new output
directory and does not change the scientific protocol.

Across both notebooks, the outer `tqdm` bars cover all 140 durable result shards
(60 in lane A and 80 in lane B), with an inner cycle bar for the active execution
batch. Each lane prints a task-start line
before model construction and a measured remaining-time projection after each
completed execution batch.

## Saved data

At every cycle, including zero, the observer saves realized Born log-probability
increments, cumulative log probability, and event counts.  At stride-four
cycles plus `Ny,2Ny,3Ny,4Ny`, it saves active-space occupations, exact-cap masks,
entropy, charge variance, `log Z`, the leading 64 `log(sigma^2)` many-body
levels, 16 soft modes' transverse profiles/wall weights, and numerical
residuals.  It does not save covariance histories, eigenvectors, or final
covariance matrices.

The hard transfer space has `N_eff=22Ny` after the Born-conditioned exterior
preparation.  The soft transfer space is the full `N_eff=40Ny`.  Exact zero/one
caps remain infinite-cost modes; they are not clipped into finite gaps.

## Relation to existing data

Existing Ny=30,40,50 purification data remain immutable and are not silently
pooled with this campaign.  Ny=30 is intentionally repeated as an independent
overlap.  A later analysis may combine compatible summaries only after checking
the complete protocol, normalization origin, source identity, cycle window,
and sampling revision.  The present bundle performs acquisition only.

## Completed hard-wall import

The hard-wall campaign completed on 2026-09-17 and was downloaded without
modifying Drive into
`gpu_data/maxmix_manybody_lyapunov_nx20_ny20-60_hard-soft_s100_4ny_gpu_v4_38gib_memory_scaled/`.
The local snapshot contains 140 checksum-verified NPZ/completion pairs: 20
five-trajectory shards and exactly 100 trajectories for each of
`Ny=20,24,30,36,44,56,60`. All 280 downloaded files, source/configuration
identity, cycle and spectrum masks, and sample coverage are bound by the
bundle-local `DOWNLOAD_MANIFEST.json`. The soft-wall arm was not run and is
not present in this import.

## Completed dynamical-critical analysis

`analyze_campaign.py` verifies and analyzes the imported hard-wall data. Its
immutable output revision is
`analysis_outputs/hard_4ny_dynamical_critical_v1/`. Run it from the repository
root with:

```bash
python 00_WORKSPACE/CURRENT/final_production_new_designs/13_maxmix_manybody_lyapunov_4ny/analyze_campaign.py
```

The analysis checks all 140 result/completion pairs, reconstructs the leading
64 many-body squared-singular-value levels, fits each trajectory in three
preregistered windows, bootstraps whole trajectories, and analyzes purification,
record closure, finite-size coefficients, sample convergence, and aggregate
wall-subspace localization. It creates PNG figure assets and one reader-facing
two-column PDF, `dynamical_critical_analysis.pdf`.

The historical download manifest's aggregate inventory digest does not
reproduce under the algorithm written in that manifest. The manifest is kept
immutable and the discrepancy is recorded explicitly in `analysis_summary.json`.
Every NPZ is nevertheless SHA-256 verified against its completion JSON, and
every completion is checked against the pinned source and configuration
identity before analysis.

The original gated output remains immutable at
`analysis_outputs/hard_4ny_dynamical_critical_v1/`. The current default is the
ungated revision
`analysis_outputs/hard_4ny_dynamical_critical_v2_ungated/`, which always reports
the fitted `alpha_c_eff`, `alpha_x_i`, `r_i=x_i/c_eff`, and `x_i/x_1` values.
Bootstrap intervals, window shifts, resolved fractions, and omitted-size and
subleading-term sensitivities are retained as diagnostics rather than used to
suppress estimates.
