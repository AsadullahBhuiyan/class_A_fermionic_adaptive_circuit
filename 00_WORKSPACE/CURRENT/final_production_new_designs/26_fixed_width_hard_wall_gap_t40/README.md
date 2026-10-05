# Fixed-width hard-wall gap pilot: 40 cycles

One self-contained Colab notebook for **Nx=20**, **Ny=20,24,28,32,36,40,44,48**,
**ten independent trajectories per size**, and **40 physical cycles for every
size**. Eight resident ten-sample batches produce sixteen immutable five-sample
result shards. Largest Ny runs first.

## Locked scientific contract

This keeps Campaign 25's original gap protocol while holding the width and
observation time fixed:

- Hard/support-truncated walls at x=5,15; `nshell=1`, trial orbitals `X`.
- `alpha_1=1`, `alpha_2=30`, perfect correction, no postselection, `raster_y`.
- Maximally mixed initialization, canonical Born-conditioned exterior preparation,
  then slab-only measurements (`meas_slab_only=True`, `triv_region_local_mode=False`).
- Complex128 covariance dynamics through
  `classA_U1FGTN_gpu.run_markov_circuit`; no covariance spectral clipping.
- Endpoint **active-slab occupation spectra and all finite-rate eigenvectors at
  cycle 40 only**. No intermediate spectra, entropy contours, record weights,
  or state histories.
- Independent root seed `2026092826`, revision
  `fixed_width_hard_wall_gap_nx20_ny20-48_s10_t40_modes_v1`.

Campaign 25 and all previous data remain unchanged. This is neither a continuation
of its ten-cycle states nor a pooled extension of its square-geometry ensemble.
Only Ny=20 has the same geometry as Campaign 25; its samples are still independent.
The seed is deterministic per execution batch; changing batch composition changes
the sampling identity.

The active slab contains `22*Ny` modes: 440,528,616,704,792,880,968,1056.
For each trajectory, retain the raw/capped occupation spectra, cap masks,
modular energies, signed rates, both gaps, and numerical/identity metadata:

\[
\epsilon_j=\log\frac{1-\nu_j}{\nu_j},\qquad
g_{\mathrm{mod}}=\min_j|\epsilon_j|,\qquad
\Delta=g_{\mathrm{mod}}/(2T)=g_{\mathrm{mod}}/80.
\]

Here Delta is the occupation-derived finite-time Lyapunov **half gap**, not the
occupation-spectrum gap around one half. Pure caps within 1e-9 of zero/one map
to infinite modular energies, not finite artificial floors. Occupation-bound
violations exceeding tolerance fail visibly.

Analysis takes each trajectory's minimum first, then the sample mean and ordinary
`SD/sqrt(10)` SEM. It exports per-sample/size CSVs, a provenance JSON, and separate
single-column PDF/PNG plots for raw and normalized gaps. No bootstrap or automatic
power-law fit. All sixteen result pairs must verify before combined analysis.
Forty cycles is a fixed observation window, not a demonstrated infinite-time limit.

### Finite-rate eigenmode storage

At the endpoint, `torch.linalg.eigh` diagonalizes the same restricted correlation
matrix `C_A=(I+G_A)/2`, one sample at a time. Every mode with
`1e-9 < nu < 1-1e-9` is retained, including a mode at nu=1/2 with zero rate.
Only pure capped modes (infinite rates) are excluded. These are the eigenvectors
of the endpoint correlation/modular matrix, not eigenvectors of the equilibrium
parent Hamiltonian or a separately propagated transfer matrix.

Each NPZ stores:

- `finite_mode_vectors`: complex128 array `(5, N_active, Kmax)`, zero padded.
- `finite_mode_count`: number of valid columns for each trajectory.
- `finite_mode_indices`: positions in the full stored occupation/modular/rate
  spectra, in ascending occupation order; unused entries are -1.
- `active_indices` and `active_coordinates_x_y_orbital`: the physical basis map.
- Eigen-equation and orthonormality residuals per sample.

For sample `i`, read `n=finite_mode_count[i]`,
`V=finite_mode_vectors[i,:,:n]`, and `j=finite_mode_indices[i,:n]`.
The matching occupations and rates are `occupation_spectrum[i,j]` and
`lyapunov_rates[i,j]`. Spatial amplitudes can be embedded into a full vector
using `full[active_indices]=V[:,k]`, then reshaped to `(Ny,Nx,2)`.
The canonical full-space index is `2*(y*Nx+x)+orbital`. Summing absolute squares
over the two orbitals gives a normalized cell-density profile. Exterior entries
are zero in this active-slab representation.

Vector phases are arbitrary; degenerate eigenspaces allow arbitrary orthonormal
bases. Compare subspace projectors or summed densities when degeneracies matter,
not individual phase-sensitive eigenvectors across samples. The vectors are saved
sample-wise and are never averaged during acquisition.

## Run and resume

Place this complete folder at
`MyDrive/final_production_new_designs/26_fixed_width_hard_wall_gap_t40/`, and open
`run_fixed_width_hard_wall_gap_t40.ipynb` in an **A100 40-GB-class** runtime.
The notebook mounts Drive once, exposes the full configuration in one cell,
copies executable files to local `/content`, and streams Jupyter-safe progress.

- `REPORT_ONLY=True`: check existing results/checkpoints without CUDA or simulation.
- `MAX_NEW_EXECUTION_BATCHES=1`: optionally run just the first pending size.
- `RUN_ANALYSIS=True`: analyze after all sixteen shards complete.
- Relaunch the same notebook after interruption; never run concurrent writers.
- The final cell releases the runtime.

Outputs go to
`MyDrive/classA_final_production_outputs/fixed_width_hard_wall_gap_nx20_ny20-48_s10_t40_modes_v1`.
Scientific changes require a separate revision; the displayed contract is locked.

Dynamics run in eight five-cycle segments. Each boundary publishes one rolling
full covariance/RNG checkpoint, with configuration/source identity. Restores
set `G_init_prepared=True`, skip repeated exterior preparation, and restore RNG
immediately before the canonical call. Cycle 40 is checkpointed before endpoint
diagonalization; an interruption during extraction repeats no dynamics.
Only after both five-sample result pairs verify is that checkpoint removed.

Publication uses local scratch, DriveFS temporary copy, size/SHA-256 readback,
atomic replacement, and completion JSON last. No Drive API, leases, dashboards,
generations, or migrations. A mismatched/partial checkpoint pair is rejected and
the batch reruns deterministically from zero. DriveFS readback does not constitute
independent server verification.

The largest ten-sample covariance checkpoint is about 0.59 GB uncompressed.
Temporary replacement can double that on Drive. Finite-mode storage depends on
the number of mixed occupations; even retaining every active mode would use less
than 0.8 GB of uncompressed vectors across all eighty trajectories.
Allow 8 GiB free on both local scratch and the output filesystem. GPU allocation
is capped at 35 decimal GB; the endpoint eigensolver handles one sample at a time.
Cycle/checkpoint timings and peak reserved GPU memory are printed. No production
A100 timing guarantee is made before this geometry has run.

## Validation

`tests/test_fixed_width_hard_wall_gap_t40.py` covers configuration and all eight
sizes, normalization by 80, rectangular CPU-backed engine continuation, RNG
preservation, partial/corrupt checkpoints, failed Drive readback, endpoint and
result-publication crash recovery, completion skipping, mean/SEM analysis,
notebook staging/progress/disconnect, and canonical-source byte identity.
Finite-mode tests cover complex eigen-equations, orthonormality, degenerate
subspaces, variable counts, pure-only samples, padding, and unchanged state/RNG.
The canonical engine itself is unchanged. No production job or Drive upload is
performed by repository tests.
