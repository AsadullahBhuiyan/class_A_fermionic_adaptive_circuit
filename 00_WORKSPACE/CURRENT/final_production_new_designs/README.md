# Final production designs: simple independent campaign bundles

On 2026-09-01, the previous seven-bundle design was moved intact to
[`00_WORKSPACE/LEGACY/final_production_new_designs_quarantine_2026-09-01/`](../../LEGACY/final_production_new_designs_quarantine_2026-09-01/).

New bundles in this directory follow the repository's simpler legacy-proven
Colab contract: local execution under `/content`, deterministic scientific work
units, visible `tqdm` progress, and completion-based DriveFS resume. They must
not import code from the quarantined tree, reuse its deployment, or write into
its v3/v4 output collections. Preserved Drive outputs and their checksums are
under
[`00_WORKSPACE/COLAB/final_production_drive_recovery_2026-09-01/`](../../COLAB/final_production_drive_recovery_2026-09-01/).

Active bundles:

- `30_square_purification_contour_t40`: 40-cycle version of the square
  full-measurement purification pilot, one trajectory each at `30x30` and
  `40x40`, hard walls, alpha1=1. Saves every-cycle entropy contours and spectra
  including cycle zero, with five-cycle state/RNG checkpoints. Separate seed
  and output folder; the 60-cycle bundle 29 remains unchanged.

- `29_square_purification_contour_pilot`: one full-measurement, maximally mixed
  hard-wall trajectory each at `30x30` and `40x40`, alpha1=1, alpha2=30, through
  60 cycles. Every-cycle purification entropy, cell entropy contour and spectra;
  independent five-cycle covariance/RNG/observer checkpoints. Small drop-in
  notebook; no covariance or eigenvector histories.

- `28_full_measurement_purification_gap_t40`: independent full-measurement
  hard-wall purification at `Nx=20`, `Ny=30,36,42,48,54,60`, 100 trajectories
  per size, 40 physical cycles. `meas_slab_only=False`, global maxmix, alpha1=1,
  alpha2=30. Saves full spectra, sample-wise gaps and all finite-occupation
  eigenvectors at cycles 6–40. Five-cycle state/RNG checkpoints, GPU-tuned
  spectral microbatches, 18 execution
  batches and 4,200 five-sample/cycle result pairs; no old-data migration.

- `01_uniform_bulk_validation`: S100, every-cycle real-space Chern and global
  charge for uniform topological perfect-correction dynamics with `DW=False`.
- `02_domain_wall_bipartite_mutual_information`: S100 endpoint bipartite mutual
  information for opposite quarter-width strips at `Nx=20`, `Ny=20,24,28`,
  across 21 critical-dense `alpha_1` values swept from 3 down to 1 and
  hard/soft domain walls.
- `03_uniform_bulk_validation_large_l`: **closed; do not resume.** Its complete
  `L=28,32` cells extend the accepted balanced bulk-validation dataset through
  `L=32`. Partial `L=36` data are retained as supporting evidence; the remaining
  `L=36` and all `L=40` work were retired on 2026-09-09.
- `04_maxmix_manybody_lyapunov_pilot`: S100 per circumference at `Nx=20` and
  `Ny=20,22,24,26,28,30,36,40`, using hard-wall maximally mixed trajectories
  to reconstruct active-space many-body Lyapunov levels and record free energy.
- `05_hard_wall_entropy_charge_batched_v2`: two disjoint A100 lanes for S100 pure-state,
  hard-wall perfect-correction dynamics at `Nx=20` and
  `Ny=30,35,40,45,50,55,60`, with genuinely batched endpoint-only
  S1/S2/S3 and charge scaling, one half-strip contour per size, rolling
  five-cycle dynamics checkpoints, and per-width endpoint resume. The v1
  observer is quarantined as superseded for performance.
- `07_maxmix_hard_soft_purification`: corrected v3 hard- and soft-wall A100
  notebooks for S100 maximally mixed purification at `Nx=20` and
  `Ny=20,30,40` through `4Ny`, saving every-cycle occupation spectra and
  cell-resolved entropy/charge-fluctuation contours plus final covariances,
  with rolling ten-cycle checkpoints. V3 accepts heterogeneous pure/mixed
  trajectories only under an explicit covariance representation and can
  checksum-import exact v2 checkpoints without altering the v2 output.
- `08_domain_wall_correlator_scaling`: independent hard- and soft-wall A100
  notebooks for S100 pure-state perfect-correction dynamics at `Nx=20`,
  `Ny=24,28,32`, `alpha_1=1,3`, and `nshell=1,2,dense`, saving the exact legacy
  wall correlator plus its x-resolved parent and total charge at every cycle
  through `2Ny`, with rolling 16-cycle checkpoints.
- `09_pure_tangent_replay_acquisition`: aggressively batched S100 pure-state
  acquisition at `Nx=20`, `Ny=24,28,32`, hard/soft walls, and `alpha_1=1,3`.
  It saves prepared and final occupied frames plus complete ordered Born records
  for later tangent-cocycle replay, without online tangent or Choi propagation.
- `10_soft_wall_entropy_charge_batched_v2`: two disjoint A100 lanes for the
  soft/untruncated counterpart of bundle 05 at `Nx=20` and
  `Ny=30,35,40,45,50,55,60`, with S100 pure-state full-layer dynamics,
  endpoint-only batched S1/S2/S3 and charge scaling, fixed half-strip contours,
  rolling five-cycle dynamics checkpoints, and per-width endpoint resume.
- `11_wall_pump_width_endpoints`: A100 endpoint acquisition at fixed `Ny=24`
  for the missing `Nx=24,28,32` dense and `Nx=28,32` `nshell=1` S100 cells,
  plus three S25 CPU/GPU bridge cells. It saves final occupied frames in
  five-trajectory shards, uses benchmark-selected execution batching, and
  publishes exact rolling five-cycle state/RNG checkpoints. This bundle is the
  endpoint-acquisition stage only; the local spectral calculation, complete
  runbook, and two-column methods note live in
  [`frozen_record_flux_charge_pilot`](../frozen_record_flux_charge_pilot/WALL_DIABATIC_WIDTH_SWEEP_RUNBOOK.md).
- `12_wall_diabatic_spectral_pump_gpu`: A100 complex128 execution of the
  M=256 primary wall-diabatized spectral-flow task over the existing endpoint
  states. It retains the CPU result schema and strict numerical gates, adds a
  mandatory eigensystem parity check, benchmark-selects one, two, four, or
  eight concurrent endpoint lanes under an 8-GiB headroom gate, adds task-level
  Drive resume, and excludes the separate 2,050-task sensitivity suite.
- `16_hard_wall_entropy_contour_all_ay`: two balanced A100 lanes for an
  independent S100 hard-wall ensemble at `Nx=20` and
  `Ny=30,35,40,45,50,55,60`. At `2Ny`, it saves each trajectory's von Neumann
  contour for every full-x strip width after averaging all periodic origins in
  relative-y coordinates. Analysis averages trajectories second and propagates
  ordinary trajectory SEM to the left- and right-wall log-chord slopes.
- `17_hard_wall_tangent_gap_cocycle`: two disjoint A100 lanes for 6,700
  hard-wall pure-state tangent trajectories at `Nx=20`. It saves five
  trajectory-resolved finite-time endpoint gaps on the 21-point `alpha_1`
  sweep at `Ny=24,28,32` and the `alpha_1=1,3` endpoints at `Ny=36,40`, while
  retaining the full scale-separated chronological cocycle only for the 200
  `Ny=40` samples. Execution v2 uses smaller restart batches and a new output
  collection, while importing the saved 25-row v1 result read-only. Quenched
  statistics extract gaps before trajectory averaging.
- `21_hard_wall_full_measurement_purification`: one A100 notebook for hard-wall
  alpha1=1,3 at Nx=20, Ny=30, 100 globally maximally mixed trajectories each,
  through 60 cycles with `meas_slab_only=False`. Both save every-cycle spectra
  and final covariances; alpha1=1 additionally saves entropy/charge-variance
  contours each cycle and the final minimum-absolute-rate eigenmode per sample.
  One 100-sample batch per alpha, ten-cycle checkpoints, forty five-sample
  result pairs. Independent output collection; no old-data migration.
- `24_flattened_imaginary_time_density`: deterministic CPU/Colab equilibrium
  benchmark at Nx20, Ny30, hard walls, alpha1=1, alpha2=30, nshell1. The
  half-filled signed OW-parent ground state and imaginary-time evolution use
  the same 1e-7 regulated Hamiltonian. Saves local density and column-charge
  correlations at 241 times, spectra, raw/normalized curves, and four figures.
  No A100, trajectory ensemble, circuit simulation, or fitted exponent.
- `26_fixed_width_hard_wall_gap_t40`: fixed-width `Nx=20` and fixed-time
  `T=40` hard-wall gap pilot for `Ny=20,24,28,32,36,40,44,48`, ten trajectories
  each. Uses the original slab-only protocol, endpoint occupation spectra and
  all finite-rate eigenvectors with spatial basis maps,
  five-cycle covariance/RNG checkpoints, and sixteen five-sample result shards.
  Raw modular gaps and gaps divided by 80 are analyzed with ordinary sample SEM.
- `25_square_hard_wall_gap_pilot`: fixed-time square-system A100 pilot at
  `Nx=Ny=20,24,28,32,36,40,44`, ten Born trajectories each and ten physical
  cycles. Uses the original hard-wall slab-only purification protocol and
  saves endpoint active occupation spectra, raw modular gaps, and gaps divided
  by `2T=20`. Five-cycle covariance/RNG checkpoints, fourteen five-sample
  result shards, ordinary trajectory SEM, and no automatic scaling fit.
- `22_hard_wall_full_measurement_clipped`: explicitly versioned alpha1=3-only
  continuation from campaign 21's verified cycle-30 checkpoint, copied read-only.
  The canonical engine clips occupation eigenvalues to [0,1] at handoff and
  after cycles 31..60; per-sample corrections and original-prefix provenance
  are saved. Twenty new five-sample shards; all v1 outputs remain untouched.
