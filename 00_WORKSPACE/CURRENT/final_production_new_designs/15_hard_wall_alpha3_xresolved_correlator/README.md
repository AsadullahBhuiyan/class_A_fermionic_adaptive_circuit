# Alpha-3 hard-wall x-resolved correlator, Nx20 Ny60 S100

Run `run_hard_wall_alpha3_correlator.ipynb` on an A100 with 40-GB-class memory.
Upload this bundle folder to `MyDrive/final_production_new_designs/` first; the
notebook stages only its four executable files into `/content`. It does not
depend on another bundle. No upload or production run is performed by building it.

## Matched scientific contract

- Nx=20, Ny=60, alpha_1=3, alpha_2=30, nshell=1.
- Hard wall only: DW=True, dw_truncation=True, meas_slab_only=True; walls [5,15].
- 100 independent pure half-filled initial states, canonical Born-conditioned
  exterior preparation once, raster_y, perfect correction, no postselection.
- Canonical `classA_U1FGTN_gpu.run_markov_circuit`, physical occupied frames,
  complex128, 120 physical cycles (2Ny), reorthonormalization every cycle.
- Independent root seed 2026091401, revision
  `hard_wall_xresolved_nx20_ny60_a1-3_nsh1_s100_2ny_raster_endpoint_v1`.

This matches the geometry/protocol of the completed bundle-14 alpha_1=1 Ny60
cohort; it changes alpha_1 and the independent sampling identity, not the
correlator estimator. Do not pool the two masses or pair trajectories by index.
The endpoint tensor is compatible with the smaller alpha-3 Ny24/28/32 campaign.
Recorded source hashes allow downstream comparisons to audit engine versions.

## Execution, outputs and resume

Three resident GPU batches: 40+40+20. The PyTorch allocator is capped at 30 GiB;
this does not certify total process GPU memory or promise 30-GiB utilization.
Twenty five-trajectory result shards have deterministic batch seeds and global
sample indices. Native Colab tqdm bars show shards and physical cycles, with
the last saved cycle displayed separately. REPORT_ONLY and
MAX_NEW_EXECUTION_BATCHES are the session controls. Keep scientific CONFIG unchanged.

Each result NPZ contains cycles=[120], normalized_cycles=[2], all 20 x columns,
ry=0..30, sample indices, x_resolved_square_correlator (5,1,20,31),
xavg_square_correlator_vs_ry (5,1,31), integer global_charge and half_filling_offset.
The estimator is sum over y and both orbital indices of |C_ij|^2/(2Ny),
where C=F F-dagger. Results retain values without a plotting cutoff.
No endpoint frame, covariance matrix, occupation spectrum or history is saved.

One rolling checkpoint NPZ/JSON per active batch holds frame/ranks, completed
cycle, NumPy and Torch CPU/CUDA RNG states, elapsed time and exact identities.
It is replaced every five cycles. The final-cycle checkpoint precedes endpoint
reduction, so endpoint publication can resume without repeating dynamics. It is
removed only after every associated result/completion pair passes readback.
Outputs are staged locally, copied to a DriveFS temporary path, reopened for
size/SHA-256 checks, renamed, then followed by their completion JSON.

Completed shards are skipped. With a valid checkpoint, interruption repeats at
most five cycles; a missing/corrupt checkpoint requires restarting that resident
batch. NPZ/JSON replacement is not a two-file transaction: interruption between
their publications can invalidate that checkpoint. DriveFS readback is not a
server-side durability guarantee. No API, lease, dashboard, archive or migration
protocol is added. Do not launch concurrent copies against this output directory.

## Space and timing

Final correlators are roughly 0.5 MB of numerical arrays before metadata and
compression. A 40-sample checkpoint is roughly 1.8–2 GiB; replacement temporarily
needs another copy. Keep at least 5 GiB free on Drive and 4 GiB on local scratch.
Mounted-filesystem free-space checks cannot certify account-wide Drive quota.

Budget roughly 8–14 A100-hours as a planning estimate based on the matching
alpha-1 Ny60 workload, not a measured alpha-3 runtime. Use the first completed
batch and cycle progress to refine it; no GPU qualification or full simulation
has been run locally. Multiple sessions can resume this same output directory.

The runner's resume/publication machinery is adapted from bundle 14, with its
large endpoint-state/spectrum products removed. The compact estimator is the
same frame-native contraction. `build_notebook.py` regenerates the frontend.
