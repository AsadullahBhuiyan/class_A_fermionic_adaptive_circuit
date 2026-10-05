# S100 hard/soft domain-wall correlator scaling

This independent A100 campaign reproduces the legacy S100 streaming-correlator
protocol while extending it across wall construction, interaction strength,
shell truncation, and circumference. It deliberately uses the repository's
simple Colab pattern: execute locally, checkpoint one active task, and publish
finished products to Drive with completion JSON written last.

## Locked scientific contract

- `Nx=20`; `Ny=24,28,32`; physical duration `2*Ny`
- `alpha_1=1,3`; `alpha_2=30`
- `nshell=1,2,None`, with `None` labeled `dense`
- 100 independent pure-state trajectories per configuration
- four fixed batches of 25 trajectories per configuration
- `init_mode=default`, half filling, `n_a=0.5`, `trial_orbitals=X`
- `sequence=raster_y`, perfect correction, no postselection
- complex128 occupied-frame dynamics through the canonical
  `classA_U1FGTN_gpu.run_markov_circuit`
- hard lane: `DW=True`, `dw_truncation=True`, `meas_slab_only=True`
- soft lane: `DW=True`, `dw_truncation=False`, `meas_slab_only=False`

The sampling revision is
`domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1`
and the independent root seed is `2026090408`. The two lanes together contain
36 configurations, 144 tasks, and 3,600 trajectories. The hard and soft
notebooks are disjoint and may run concurrently in separate A100 runtimes.

## Saved observables

Each completed 25-trajectory task writes one compressed NPZ containing every
cycle from `t=0` through `2Ny`:

- `cycles`, `normalized_cycles`, `ry_values`, and `x_values`
- `global_sample_indices`
- `x_resolved_square_correlator`, shaped
  `(25, 2*Ny+1, 20, Ny/2+1)`
- `xavg_square_correlator_vs_ry`, the exact `x` average of that tensor and the
  exact legacy estimator
- integer `global_charge`, equal to the occupied-frame rank

The estimator is evaluated directly from the occupied frame. It does not
materialize or save covariance matrices. No Chern marker, entropy, local charge
map, Born-probability map, frustration field, or postselection control is saved.

For the hard wall, cycle zero is observed after the canonical Born-conditioned
exterior preparation. For the soft wall it is the untruncated initial pure
state.

## Resume behavior

Each lane has 72 deterministic tasks. A result is complete only when its NPZ
and completion JSON both re-open through DriveFS and match the task/configuration
identity, byte count, and SHA-256. Completed tasks are skipped on restart.

The active task advances through canonical 16-cycle calls. After each call the
runner replaces one stable `checkpoint.npz` and `checkpoint.json`. The
checkpoint contains the occupied frame/ranks, NumPy and Torch CPU/CUDA RNG
states, observer prefix, completed cycle, task identity, configuration hash,
and source hashes. A hard-wall continuation passes the already-prepared frame
back to the engine and explicitly verifies that exterior preparation was not
repeated. The checkpoint is removed only after the final NPZ/completion pair
reverifies.

There is no Drive API, lease, dashboard, migration ledger, archive, remote
status process, or generation/pointer protocol. An interruption can lose at
most the current 16-cycle segment; restart the same notebook to continue.

## Running in Colab

Upload this whole folder to
`/content/drive/MyDrive/final_production_new_designs/`, then open one or both:

- `run_hard_wall_correlator.ipynb`
- `run_soft_wall_correlator.ipynb`

Use an A100 40-GB-class runtime. In the configuration cell, leave
`REPORT_ONLY=False` to run, or set it to `True` for a verified inventory only.
Set `MAX_NEW_TASKS` to an integer to cap newly completed tasks in that session;
`None` runs until completion or interruption. The notebook copies the bundle
to `/content`, streams both `tqdm` levels live, and writes outputs under:

`/content/drive/MyDrive/classA_final_production_outputs/domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1`

Expected total cost is approximately 120--160 A100-hours including the
every-cycle observer. Two concurrent lanes should require roughly three to
five continuous days. Final compressed results are expected to occupy about
0.5--1 GB; one active rolling checkpoint per lane may temporarily use several
hundred MB.
