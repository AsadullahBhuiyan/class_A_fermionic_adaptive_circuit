# HANDOFF

## Scope

This handoff is for the recent `src/interface_chern` work on:

- equilibrium / monitored strip Green's functions,
- the row-resolved Chern marker,
- the bulk-reset proposal,
- and the current nonquantization debugging thread.

The old `HANDOFF.md` had drifted toward other note work. This one is meant to
be a compact status note for the present marker / Keldysh / slab-geometry
thread.

## Main Files Touched

- [src/interface_chern/InterfaceChern.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/InterfaceChern.jl)
- [src/interface_chern/grids.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/grids.jl)
- [src/interface_chern/marker.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/marker.jl)
- [src/interface_chern/green_solver.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/green_solver.jl)
- [src/interface_chern/bulk_reset.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/bulk_reset.jl)
- [note/monitored_CI2.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/monitored_CI2.tex)
- [note/slab_geometry_note.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/slab_geometry_note.tex)
- [note/output/monitored_CI_interface_note.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/output/monitored_CI_interface_note.tex)
- [note/bulk_projector_reset_proposal.md](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/bulk_projector_reset_proposal.md)

## Code Changes Already Made

### 1. Keldysh `-i` Convention Fix

The repo convention is:

```math
G^K = G^R \Sigma^K G^A,
\qquad
C = \frac12 I - \frac{i}{2}\int \frac{d\omega}{2\pi}\, G^K.
```

So `\Sigma^K` must be anti-Hermitian.

Fixed downstream:

- [note/monitored_CI2.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/monitored_CI2.tex)
- [note/slab_geometry_note.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/slab_geometry_note.tex)
- [note/output/monitored_CI_interface_note.tex](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/output/monitored_CI_interface_note.tex)
- [src/interface_chern/green_solver.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/green_solver.jl)

`sigma_monitor_keldysh_matrix(...)` now inserts `-1im * K` rather than a real
diagonal kernel.

### 2. Bulk Reset Proposal Note

Added:

- [note/bulk_projector_reset_proposal.md](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/bulk_projector_reset_proposal.md)

It explains the distributed projector-reset idea in:

- Wannier / Parseval-frame language,
- projector-kernel language in mixed `(k_x,y)` space,
- Lindblad form,
- and the corresponding quadratic Keldysh self-energies.

Important caveat already clarified there:

- this Gaussian reset bath is a reservoir-like gain/loss construction,
- it pins the one-body density toward the band projector,
- but it does broaden the spectrum via `\Sigma^R = -i\gamma I/2`,
- so it is not a noninvasive “flip only when wrong” measurement.

### 3. Marker Derivative Fix

The previous marker implementation used a 2-point centered periodic derivative
in `k_x`. That was replaced by an exact periodic spectral derivative on the
finite `k_x` grid.

Changes:

- [src/interface_chern/InterfaceChern.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/InterfaceChern.jl)
  now imports `FFTW`
- [src/interface_chern/grids.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/grids.jl)
  has new `spectral_periodic_derivative(...)`
- [src/interface_chern/marker.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/marker.jl)
  now supports `derivative_scheme=:spectral` and uses it by default

This only changed the equilibrium `M=100` TI average modestly:

- old centered: `c_ti_avg = -0.8065708790157218`
- new spectral: `c_ti_avg = -0.8107391642044548`

So the `k_x` derivative was one issue, but not the dominant one.

## Equilibrium Marker Debugging: What Was Tested

### 1. `M=20` Domain-Wall vs Uniform-Topological Comparison

Generated under:

- [data/interface_chern_M20_compare](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_M20_compare)

Files of interest:

- domain wall marker:
  [marker.jld2](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_M20_compare/eq_rti1p000_rtriv3p000_M20_Nkx41/marker.jld2)
- uniform topological marker:
  [marker.jld2](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_M20_compare/eq_rti1p000_rtriv1p000_M20_Nkx41/marker.jld2)
- comparison plot:
  [marker_two_panel.pdf](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_M20_compare/marker_two_panel.pdf)

Key result:

- the deep TI-side rows `y = 10:19` in the domain-wall run and in the uniform
  topological run agree to about `8.3e-7`
- both have a plateau near `-0.926625`

Interpretation:

- the equilibrium nonquantization is not caused by the interface / domain wall
- it is already present in the uniform topological strip with the current
  equilibrium construction

### 2. `eta` Scan Interpretation

Existing saved data:

- [data/interface_chern_eta_fixed_dw_M30](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_eta_fixed_dw_M30)

Observed marker trend:

- `eta = 0.10`: `c_ti_avg ≈ -0.78546`
- `eta = 0.04`: `c_ti_avg ≈ -0.88105`
- `eta = 0.02`: `c_ti_avg ≈ -0.90309`

This suggested smaller `eta` helps, but only up to the point where the
real-frequency integral is still numerically resolved.

Additional check on the full saved covariance matrix `C`:

- `eta = 0.10`: full eigenvalues stay near `[0,1]`
- `eta = 0.04`: still acceptable
- `eta = 0.02`: full `C` becomes badly unphysical, with eigenvalues outside
  `[0,1]`

Interpretation:

- a smaller `eta` is physically the right direction for the equilibrium
  projector limit,
- but at fixed `\omega` grid the real-axis quadrature becomes under-resolved and
  the numerics degrade

So “smaller `eta` pushes `C` away from a projector” is true for the current
discretized integral, but that is a numerical-resolution failure, not the
correct physical trend.

### 3. `omega_max` vs `domega` Convergence at Fixed `eta`

Controlled test saved in:

- [data/interface_chern_omega_convergence_M20/omega_convergence_summary.md](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_omega_convergence_M20/omega_convergence_summary.md)

Setup:

- uniform topological strip
- `M = 20`, `Nkx = 41`
- `eta = 0.04` fixed

Cases tested:

- base: `omega_max = 12`, `domega_dense = 0.02`
- wide: `omega_max = 20`, `domega_dense = 0.02`
- fine: `omega_max = 12`, `domega_dense = 0.01`
- wide_fine: `omega_max = 20`, `domega_dense = 0.01`

Result:

- increasing `omega_max` from `12` to `20` changes `c_ti_avg` only by about
  `2.9e-4`
- refining `domega_dense` from `0.02` to `0.01` changes it by less than `1e-6`
- all four runs keep the full `C` physical

Interpretation:

- once `eta = 0.04` is already resolved, neither the frequency window nor the
  step size is the dominant source of the residual nonquantization

## Current Diagnosis

The remaining equilibrium nonquantization is probably not:

- the domain wall,
- the `k_x` derivative discretization,
- or the `\omega` integration window / step size, once `eta=0.04` is resolved.

The strongest remaining suspect is the equilibrium construction itself:

- the code uses the same `eta_eff` in the boundary-condition surface Green's
  functions and again in the strip resolvent
- so the same imaginary regulator enters both the embedded leads and the
  explicit strip
- equilibrium `C` is still being built from a broadened real-axis integral,
  rather than directly from the projector / exact zero-temperature occupation

Concretely:

- [surface_greens.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/surface_greens.jl)
  uses `z = omega + i eta`
- [green_solver.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/green_solver.jl)
  also uses `z = omega + i eta` in the strip resolvent

So the same `eta` is used for:

- the BC / lead self-energies,
- and the strip Green's function

This should almost certainly be separated.

## Most Likely Next Fix

Implement separate regulators:

- `eta_lead` for the surface Green's functions / embedded BC
- `eta_strip` or `eta_corr` for building the equilibrium strip correlator
- optionally `eta_spec` for the plotted spectral functions

Better still, for equilibrium marker runs:

- stop computing `C` from the broadened real-axis integral,
- and compute the equilibrium projector directly

That is the clean fix if the goal is a quantized equilibrium local marker.

## Important Gotchas

### 1. Case Tag Collision

The current `case_tag(...)` in [src/interface_chern/utils.jl](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/src/interface_chern/utils.jl)
does not include:

- `omega_max`
- `omega_dense`
- `domega_dense`
- `domega_coarse`
- `eta`

So rerunning the same physical geometry with different integration settings can
overwrite the same `green.jld2` and `marker.jld2`.

That already happened during the `omega`-convergence check; the summary file was
added specifically to preserve those numbers.

### 2. Reduced Blocks Are Not Projectors

A strip-restricted or row-restricted block of the full projector is not itself a
projector. So `C^2-C` on a subblock is not a valid diagnostic by itself.

The correct sanity checks are:

- Hermiticity,
- eigenvalues of the full computed strip `C` lying in `[0,1]`,
- and convergence of the local marker plateau under numerical refinement

## Generated Artifacts Worth Looking At

- [note/bulk_projector_reset_proposal.md](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/note/bulk_projector_reset_proposal.md)
- [data/interface_chern_M20_compare/marker_two_panel.pdf](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_M20_compare/marker_two_panel.pdf)
- [data/interface_chern_omega_convergence_M20/omega_convergence_summary.md](/mnt/c/users/haoyu/onedrive%20-%20cornell%20university/projects/monitored%20chern%20insulator/edge%20monitoring/code/data/interface_chern_omega_convergence_M20/omega_convergence_summary.md)

## Workspace State

At the time this handoff was written, local status showed:

- `HANDOFF.md` modified
- `src/interface_chern/InterfaceChern.jl` changed
- `src/interface_chern/grids.jl` changed
- `src/interface_chern/marker.jl` changed

There may also be unrelated user changes elsewhere in the repo. Do not revert
anything broadly; inspect before touching.
