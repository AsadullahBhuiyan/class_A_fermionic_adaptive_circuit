# Hard-wall all-`Ay` entropy-contour campaign

This independent A100 campaign extends the endpoint observer used by the
completed hard-wall v2 campaign. It keeps the same scientific dynamics but
uses a new seed ensemble and retains a trajectory-resolved, all-origin-averaged
von Neumann contour for every subsystem width. Existing v2 outputs are never
read, resumed, or pooled with this revision.

## Locked contract

- `Nx=20`; `Ny=30,35,40,45,50,55,60`; 100 trajectories per size
- hard/support-truncated walls at `x=5,15`
- `nshell=1`, `alpha_1=1`, `alpha_2=30`
- pure initialization, `raster_y`, perfect correction, complex128
- `2*Ny` physical cycles through `classA_U1FGTN_gpu.run_markov_circuit`
- root seed `2026091416`
- revision `hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3`

Only rank-based global charge is observed during the circuit. At the endpoint,
the runner evaluates

`A(y0,Ay) = [0,Nx) x [y0,y0+Ay)`

for every periodic `y0=0,...,Ny-1` and `Ay=0,...,Ny//2`. It first aligns each
contour in `dy=(y-y0) mod Ny`, then averages the `Ny` origins inside each
trajectory. The result shard retains those per-trajectory averages. The
independent statistical unit is therefore one trajectory, never one translated
origin.

Each five-trajectory result contains:

- `endpoint__contour_von_neumann_y0avg` with shape
  `(5, Ny//2+1, Nx, Ny//2)`;
- `valid_dy_count[Ay]=Ay`, with exact-zero padding after the valid `dy` range;
- trajectory-resolved endpoint curves for `S1`, `S2`, `S3`, strip charge, and
  intrinsic charge variance;
- global charge and half-filling offset for every cycle, including cycle zero.

No per-origin contour, Rényi contour, charge-variance contour, covariance
history, or occupied-frame history is retained.

## Lanes and resume

- Lane A: `Ny=60,50`
- Lane B: `Ny=55,45,40,35,30`

Execution batches contain `80,60,40,30,25,20,20` trajectories for increasing
`Ny`; immutable result shards contain five trajectories. Both lanes together
produce 25 execution batches and 140 result/completion pairs.

Dynamics checkpoints are published every five cycles. Once dynamics finishes,
the final frame stays durable while the endpoint solver publishes progress
after every completed `Ay`. A restart resumes the last stage without repeating
completed widths. The runner uses local `/content` scratch and a simple DriveFS
temporary-copy, readback, checksum, and atomic-rename publication contract.

## Batched endpoint solver

At fixed `Ay`, trajectory/origin pairs are flattened into matrix batches.
`torch.linalg.eigh` diagonalizes the Hermitian restricted correlation matrices;
the same occupations produce all five scalar observables, while eigenvectors
are used only for the von Neumann contour. Contours are accumulated by
trajectory on the GPU and only the origin average is transferred to the CPU.

An A100 benchmark tries matrix batches `8,16,32,64,80,128`, requires 8 GiB of
headroom, checks dense/SVD equivalence and contour closure, and blocks
production if the projected complete `Ny=60`, 20-trajectory endpoint exceeds
one hour.

## Running

Upload this folder unchanged to
`MyDrive/final_production_new_designs/16_hard_wall_entropy_contour_all_ay`.
Open both notebooks in separate A100 40-GB runtimes and run them top to bottom.
The editable controls are `REPORT_ONLY`, `BENCHMARK_ONLY`,
`MAX_NEW_EXECUTION_BATCHES`, and, in lane A, `RUN_ANALYSIS`.

Outputs go to:

`MyDrive/classA_final_production_outputs/hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3`

Enable analysis only after report-only runs across both lanes total 140/140
verified shards.

## Analysis

The analysis first averages origins within each trajectory and only then
averages the 100 trajectories. Cellwise uncertainty is the ordinary trajectory
SEM. It integrates the contours over `x={4,5,6}`, `x={14,15,16}`, and all `x`,
then fits the ensemble-mean curves over `Ay=8,...,Ny//2`. Slope uncertainty
uses the full trajectory covariance between subsystem widths. The single-wall
target is `m=1/6` (`c_wall=6m=1`); the full-strip target is `m=1/3`.

There is no bootstrap, Drive API, lease, dashboard, migration, archive, or
checkpoint-pointer protocol.
