# Hard-wall x-resolved correlator scaling

This independent A100 campaign fills the missing large-\(N_y\) x-resolved
correlator data needed for the domain-wall power-law analysis. It does not
modify or pool with the completed endpoint entropy/charge campaign.

## Locked scientific contract

- `Nx=20`; `Ny=60,50,40` in that execution order
- 100 trajectories per size
- hard/support-terminated wall at `xL=5`, `xR=15`
- `DW=True`, `dw_truncation=True`, `meas_slab_only=True`
- `alpha_1=1`, `alpha_2=30`, `nshell=1`
- pure half-filled initialization, `raster_y`, perfect correction
- complex128 physical occupied-frame dynamics for `2*Ny` cycles
- canonical `classA_U1FGTN_gpu.run_markov_circuit`
- root seed `2026091001`
- revision
  `hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_endpoint_frame_halfcov_occupations_v2_30gib_batched`

The final-time squared correlator, integer total charge, occupied frame,
occupied ranks, half-system covariance, and half-system single-particle
occupations are retained.
For each trajectory,

```text
G_x(r_y) = sum_{y,mu,nu} |C[(x,y,mu),(x,y+r_y,nu)]|^2 / (2*Ny).
```

The saved x average is the mean of this quantity over all 20 x columns and
exactly matches the legacy dense-covariance estimator. The correlator observer
evaluates the projector directly from the occupied frame and never materializes
the full-system covariance. The saved complex128 frame `F` and integer ranks
also reconstruct the endpoint covariance exactly as `C=F@F_dagger` and
`G=2*C-I`.

The fixed saved subsystem is

```text
A = [0,Nx) x [0,Ny//2).
```

With row ordering `(y,x,orbital)`, `F_A` is the first `2*Nx*(Ny//2)` rows of
the active occupied frame. The bundle saves the restricted correlation matrix
`C_A=F_A@F_A_dagger` and the ascending eigenvalues
`nu_A=eigvalsh(C_A)` for every endpoint sample. These eigenvalues are the
nontrivial half-system single-particle occupation spectrum needed for the
spectral density. Tiny roundoff excursions are recorded before values are
clipped into `[0,1]`.

Each five-sample NPZ stores `cycles=[2*Ny]`, `normalized_cycles=[2.0]`, the
`x_values`, `ry_values`, wall positions and global sample indices,
`x_resolved_square_correlator` with shape `(5,1,20,Ny//2+1)`, its exact x
average, integer `global_charge`, integer `half_filling_offset`, complex128
`occupied_frame`, integer `occupied_ranks`, complex128
`half_system_covariance` with shape
`(5,1,2*Nx*(Ny//2),2*Nx*(Ny//2))`, and the corresponding float64
`half_system_occupation_spectrum` with shape `(5,1,2*Nx*(Ny//2))`.
The length-one time axis is deliberately compatible with the corresponding
final slice of the existing Ny=24,28,32 every-cycle files.

## A100 batching and resume

The resident execution sizes are 80 trajectories at `Ny=40`, 50 at `Ny=50`,
and 40 at `Ny=60`. Thus the three sizes use two, two, and three resident
batches respectively, for seven GPU execution batches. Each batch is split
into immutable five-trajectory result shards, for 60 durable products in total.
This doubles the large-size residency of v1 while leaving the durable shard
size unchanged. PyTorch's allocator is hard-limited to 30 GiB and the runner
prints peak reserved memory after each execution batch; a batch that cannot
fit fails visibly rather than silently altering the ensemble.

Dynamics run in five-cycle canonical-engine segments. After every segment, one
stable uncompressed `checkpoint.npz` plus `checkpoint.json` is published. It
contains the occupied frame and ranks, completed cycle, NumPy RNG state, Torch
CPU/all-CUDA RNG states, and exact task/config/source identity. Restoration is
bitwise exact and uses `frame_init_prepared=True`, so the hard-wall exterior is
not prepared a second time.

The final-cycle checkpoint is retained while endpoint shards are computed.
This means a disconnect during endpoint reduction or Drive publication does
not repeat the dynamics. The checkpoint is deleted only after every associated
five-sample NPZ/completion-JSON pair passes DriveFS byte-count and SHA-256
readback.

The vectorized engine uses one deterministic RNG stream per execution batch;
the global sample indices identify its independent batch rows. Batch sizes and
ordering are part of the locked sampling revision and must not be changed when
resuming this ensemble.

The superseded v1 output remains immutable. Its checkpoints and result pairs
are not compatible with this v2 execution partition and are never deleted or
silently pooled.

## Running

Upload this directory unchanged to:

`MyDrive/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling`

Open `run_hard_wall_xresolved_correlator.ipynb` on an A100 40-GB runtime and run
top to bottom. The notebook mounts Drive once, stages executable files under
`/content`, and executes the staged runner inside the notebook kernel so
`tqdm.auto` renders both progress levels natively. Runtime controls are
`REPORT_ONLY` and `MAX_NEW_EXECUTION_BATCHES`.

Outputs are written to:

`MyDrive/classA_final_production_outputs/hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_endpoint_frame_halfcov_occupations_v2_30gib_batched`

The matching production measurements imply approximately 1.5 A100-hours at
`Ny=40`, 5.7 hours at `Ny=50`, and 11.5 hours at `Ny=60`: about 18.7 hours for
dynamics. Budget **21--24 A100-hours total** for endpoint covariance
construction, eigendecomposition, and writing and readback-verifying the large
payloads. In one uninterrupted runtime that is roughly
one day, but Colab will normally require several sessions. Reopen and rerun the
same notebook after a disconnect. A restart skips verified result shards and
continues a partial execution batch from its last five-cycle checkpoint.

The uncompressed endpoint frames occupy about **9.18 GiB** and the half-system
covariances occupy about **4.59 GiB**, for **13.77 GiB** of dominant final
payloads. A five-sample result is approximately 146 MiB at `Ny=40`, 229 MiB at
`Ny=50`, and 330 MiB at `Ny=60`. The occupation eigenvalues are small by
comparison. The runner calculates the storage still required from the verified
resume inventory and demands that amount plus 15% overhead and 3 GiB working
headroom (about **20.84 GiB free** for a completely fresh campaign). It also
requires 4 GiB of local runtime storage. Uncompressed NPZ is intentional
because numerical frames and covariance matrices have little useful
compression and compression would waste A100 time.

There is deliberately no Drive API, lease, dashboard, archive, migration,
remote-status, or qualification workflow.
