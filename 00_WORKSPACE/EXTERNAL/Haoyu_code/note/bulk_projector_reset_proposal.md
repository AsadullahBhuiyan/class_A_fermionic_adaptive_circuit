# Proposal: Distributed Bulk Projector Reset for the Slab Chern-Marker Problem

## Purpose

The current slab/interface calculation in `src/interface_chern` treats the two
exterior bulks as equilibrium leads attached only at the outer boundary rows of
the explicit strip.  This is enough to compute the exact reduced correlation
matrix on the explicit strip, but it does **not** literally keep the full bulk
correlation matrix equal to the isolated zero-temperature bulk projector after
the interface is coupled in.

The present note describes a different proposal aimed at the following goal:

- keep the top bulk (`y > 0`) and bottom bulk (`y < 0`) locally pinned toward
  their zero-temperature band projectors throughout the bulk,
- while still allowing a nontrivial bulk-interface cross-sector correlation
  matrix,
- and while staying within a quadratic Keldysh / Gaussian description.

The motivating physical picture is:

- in each bulk, monitor in a Wannier frame adapted to the underlying band
  structure,
- and reset any weight that is not in the desired zero-temperature band sector.

At the Gaussian level, the clean implementation of that idea is a
**distributed bulk reset dissipator** whose dark state is the desired bulk band
projector.

## Executive Summary

For each bulk side `s in {top, bottom}`, define the zero-temperature occupied
band projector `P_s` and empty-band projector `Q_s = I_s - P_s` on the bulk
rows to be stabilized.  Then add a Markovian quadratic dissipator built from
gain jumps into `P_s` and loss jumps out of `Q_s`.

In the Keldysh convention used by the current slab solver,

```math
G^K = G^R \Sigma^K G^A,
\qquad
C = \frac{1}{2} I - \frac{i}{2}\int \frac{d\omega}{2\pi}\, G^K,
```

the Keldysh self-energy itself must be anti-Hermitian.  The reset channel
therefore corresponds to adding on the stabilized bulk sector

```math
\Sigma_{s,\mathrm{reset}}^R = -\frac{i \gamma_s}{2} I_s,
\qquad
\Sigma_{s,\mathrm{reset}}^A = +\frac{i \gamma_s}{2} I_s,
```

```math
\Sigma_{s,\mathrm{reset}}^K
=
-i \gamma_s (Q_s - P_s)
=
-i \gamma_s (I_s - 2 P_s).
```

For an isolated uniform bulk, this drives the equal-time correlation matrix to

```math
C_s = P_s,
```

which is exactly the zero-temperature band projector.

The important point is that the proposal depends only on the projector kernels
`P_s`, not on a particular orthonormal Wannier basis.  This is essential on the
Chern side, where an exponentially localized orthonormal Wannier basis of the
occupied band does not exist.

## Why the Current Boundary-Lead Embedding Is Not the Same Thing

The present interface code uses semi-infinite equilibrium leads attached only at
the outer boundary rows of the explicit strip.  This imposes the correct
zero-temperature FDT relation for the reservoirs, but once the strip is coupled
to those reservoirs, the near-boundary bulk correlator is dressed by the
interface through the usual Dyson backaction.

So the current lead construction gives:

- the exact reduced correlator on the explicit strip,
- exact zero-temperature reservoir statistics at the cut,
- but **not** an exact pointwise identity between the full bulk correlator and
  the isolated bulk projector after coupling.

If the goal is instead to hold the bulks themselves close to zero temperature
throughout their volume, then one needs a distributed stabilization mechanism
acting throughout the bulk, not only at the edge of the explicit strip.

## Target Zero-Temperature Bulk Projectors

For each uniform bulk side `s`, with mass parameter `r_s`, the Bloch Hamiltonian
is

```math
h_s(k_x,k_y)
=
t_0 \Big[
\sigma^1 \sin k_x
+ \sigma^2 \sin k_y
+ \sigma^3 (r_s - \cos k_x - \cos k_y)
\Big].
```

Write it as

```math
h_s(k) = \mathbf d_s(k) \cdot \boldsymbol \sigma,
\qquad
E_s(k) = |\mathbf d_s(k)|.
```

Then the zero-temperature occupied-band and empty-band projectors are

```math
P_s(k_x,k_y) = \frac{1}{2} \left[ I - \hat{\mathbf d}_s(k) \cdot \boldsymbol \sigma \right],
\qquad
Q_s(k_x,k_y) = I - P_s(k_x,k_y).
```

Since the slab calculation keeps translation invariance in `x`, the natural
representation is mixed `(k_x, y)`.  In that representation the bulk projector
kernel is

```math
P_s(k_x; y-y')
=
\int_{-\pi}^{\pi} \frac{d k_y}{2\pi}\,
e^{i k_y (y-y')} P_s(k_x,k_y),
```

and similarly for `Q_s`.

For any chosen stabilized row set `S_s`, define the restricted kernels

```math
\big(P_s^{(S)}(k_x)\big)_{yy'} = P_s(k_x; y-y'),
\qquad
y,y' \in S_s.
```

Because `P_s + Q_s = I` in the full bulk, restriction to `S_s` still gives

```math
P_s^{(S)}(k_x) + Q_s^{(S)}(k_x) = I_{S_s}.
```

This is the basic input needed for the reset construction.

## Wannier-Frame Interpretation

The physical intuition is to monitor in a Wannier frame associated with the
underlying band structure.

Let `{ |w_{s,a}^{occ}> }` be a family of states spanning the occupied-band
subspace of bulk `s`, and `{ |w_{s,b}^{emp}> }` a family spanning the empty-band
subspace.  These families are allowed to be overcomplete, but they must resolve
the corresponding projectors:

```math
\sum_a |w_{s,a}^{occ}\rangle \langle w_{s,a}^{occ}| = P_s,
\qquad
\sum_b |w_{s,b}^{emp}\rangle \langle w_{s,b}^{emp}| = Q_s.
```

Such a family is a Parseval frame for the corresponding subspace.  More
generally, if the frame operator is `A P_s` or `A Q_s` with a constant `A > 0`,
the frame is tight and one can absorb `A` into the reset rate.

This is the precise sense in which an overcomplete Wannier basis is acceptable.
The dissipator depends only on the frame operator, so as long as the frame
resolves the desired projector, the Gaussian theory only sees `P_s` and `Q_s`.

This avoids the Chern-band obstruction:

- on the topological side there is no exponentially localized orthonormal
  Wannier basis for the occupied band alone,
- but one may still use an overcomplete Parseval frame, or skip the explicit
  frame entirely and work directly with the projector kernel `P_s(k_x; y-y')`.

## Quadratic Reset Dissipator

The proposed Markovian bulk reset uses jump operators

```math
L_{s,a}^{in} = \sqrt{\gamma_s}\, c^\dagger(w_{s,a}^{occ}),
\qquad
L_{s,b}^{out} = \sqrt{\gamma_s}\, c(w_{s,b}^{emp}),
```

where

```math
c(w) = \sum_j \langle w | j \rangle c_j
```

and `j` labels the single-particle states in the stabilized bulk region.

At the single-particle level, these jumps define gain and loss kernels

```math
\Lambda_s^+ = \gamma_s P_s^{(S)},
\qquad
\Lambda_s^- = \gamma_s Q_s^{(S)}.
```

For a quadratic Lindblad problem, the equal-time correlation matrix obeys

```math
\dot C
=
-i [h, C]
- \frac{1}{2} \{ \Lambda^+ + \Lambda^-, C \}
+ \Lambda^+ .
```

With the above choice,

```math
\Lambda_s^+ + \Lambda_s^- = \gamma_s I_{S_s},
```

so on an isolated stabilized bulk,

```math
\dot C_s
=
-i [h_s, C_s]
- \gamma_s (C_s - P_s^{(S)}).
```

Since `[h_s, P_s] = 0`, the isolated steady state is exactly

```math
C_s = P_s^{(S)}.
```

This is the simplest mathematical statement of the proposal.

## Corresponding Keldysh Self-Energy

In the convention used in `monitored_CI2.tex` and `src/interface_chern`, the
same reset channel is represented by a frequency-independent bulk self-energy

```math
\Sigma_{s,\mathrm{reset}}^R(\omega, k_x)
=
-\frac{i \gamma_s}{2} I_{S_s},
```

```math
\Sigma_{s,\mathrm{reset}}^A(\omega, k_x)
=
\left(\Sigma_{s,\mathrm{reset}}^R\right)^\dagger
=
\frac{i \gamma_s}{2} I_{S_s},
```

```math
\Sigma_{s,\mathrm{reset}}^K(\omega, k_x)
=
-i \gamma_s \left( Q_s^{(S)}(k_x) - P_s^{(S)}(k_x) \right)
=
-i \gamma_s \left( I_{S_s} - 2 P_s^{(S)}(k_x) \right).
```

It is useful to separate the actual anti-Hermitian Keldysh self-energy from the
underlying Hermitian distribution kernel.  Define

```math
K_{s,\mathrm{reset}}(k_x)
=
\gamma_s \left( I_{S_s} - 2 P_s^{(S)}(k_x) \right),
```

so that

```math
\Sigma_{s,\mathrm{reset}}^K(\omega, k_x)
=
-i K_{s,\mathrm{reset}}(k_x).
```

This is directly analogous to writing a monitored-orbital source in the form

```math
K_M = \Gamma (1 - 2n),
\qquad
\Sigma_M^K = -i K_M.
```

In several older notes the Hermitian kernel `K_M` was written informally as
`\Sigma_M^K`; for literal implementation into the present solver one must keep
the extra factor of `-i`.

The reset kernel is therefore analogous to the local monitored-orbital formula

```math
K_M = \Gamma (1 - 2n),
```

except that here `n` is replaced by the desired band projector.

A few structural remarks are important:

- `\Sigma_{\mathrm{reset}}` is local in time because the reset is Markovian.
- It is diagonal in `k_x` because translation invariance in `x` is preserved.
- It is generally **not** strictly local in `y`, because `P_s^{(S)}(k_x)` is a
  projector kernel on the stabilized row set.
- For a gapped bulk, the projector kernel decays exponentially with row
  separation, so this nonlocality is short-ranged in practice.

## Geometry for the Interface Problem

Choose a finite explicit window

```math
W = \{ -M_{tot}, \dots, M_{tot}-1 \},
```

and within that window define:

- an interface/core region where the nontrivial monitored physics lives,
- a top stabilized bulk region `S_top subset W` with rows well inside `y > 0`,
- a bottom stabilized bulk region `S_bot subset W` with rows well inside
  `y < 0`.

One possible choice is

```math
S_{top} = \{ y_T, y_T+1, \dots, M_{tot}-1 \},
\qquad
S_{bot} = \{ -M_{tot}, \dots, y_B \},
```

with `y_B < 0 < y_T`, leaving an unreset interface buffer between the two
stabilized bulks.

The total self-energy then becomes

```math
\Sigma^R_{tot}
=
\Sigma^R_{lead}
+ \Sigma^R_{reset,top}
+ \Sigma^R_{reset,bot}
+ \Sigma_M^R,
```

```math
\Sigma^K_{tot}
=
\Sigma^K_{lead}
+ \Sigma^K_{reset,top}
+ \Sigma^K_{reset,bot}
+ \Sigma_M^K.
```

The steady-state Green's functions are

```math
G^R(\omega, k_x)
=
\Big[
(\omega + i \eta) I
- H_W(k_x)
- \Sigma^R_{tot}(\omega, k_x)
\Big]^{-1},
```

```math
G^K(\omega, k_x)
=
G^R(\omega, k_x)\,
\Sigma^K_{tot}(\omega, k_x)\,
G^A(\omega, k_x).
```

Finally,

```math
C_W(k_x)
=
\frac{1}{2} I
- \frac{i}{2} \int \frac{d\omega}{2\pi}\, G^K(\omega, k_x).
```

This gives a correlation matrix on the entire explicit window `W`, including:

- interface-interface blocks,
- bulk-interface cross blocks,
- and bulk-bulk blocks inside the stabilized explicit window.

## What This Proposal Does and Does Not Guarantee

This point is subtle and important.

### What it does guarantee

For an isolated uniform bulk with the reset channel applied everywhere in the
stabilized region, the steady state is exactly the zero-temperature projector
`P_s`.

In the coupled interface problem, the reset channel continuously drives the bulk
toward `P_s`, so one expects:

- deep in the stabilized bulk, the correlation matrix should be very close to
  the zero-temperature projector,
- the bulk-interface cross-sector correlation matrix is generated dynamically by
  the full Keldysh solution,
- and the interface region can remain nontrivial.

### What it does not guarantee at finite reset rate

At finite `\gamma_s`, the coupled interface problem does **not** keep every bulk
matrix element exactly equal to the isolated projector all the way up to the
interface.  The Hamiltonian coupling to the interface competes with the reset,
so one should expect a healing region near the interface.

Therefore the proposal is best thought of as:

- a distributed zero-temperature anchor for the bulks,
- not an exact algebraic constraint that freezes the full bulk correlator
  pointwise at finite `\gamma`.

If exact pointwise freezing of the full bulk correlator and simultaneous
nontrivial cross-sector coherence are both required, one would need a more
engineered nonreciprocal or feedback-like construction.

## Why the Projector-Kernel Formulation Is Better Than Explicit Wannier States

For coding and analysis, it is cleaner to work directly with the projector
kernels `P_s(k_x; y-y')` than with explicit Wannier states.

The reasons are:

- the Gaussian dissipator depends only on the frame operator, hence only on
  `P_s` and `Q_s`,
- the projector kernel is directly available from the bulk Bloch Hamiltonian,
- the topological obstruction on the Chern side is automatically handled,
- and the mixed `(k_x, y)` representation already used by the slab code matches
  this construction naturally.

So the practical implementation should be:

1. Compute `P_s(k_x, k_y)` from the bulk Bloch Hamiltonian.
2. Fourier transform in `k_y` to obtain `P_s(k_x; y-y')`.
3. Restrict that kernel to the stabilized top and bottom row sets.
4. Build `\Sigma_{reset}^R` and `\Sigma_{reset}^K` from those restricted
   kernels.
5. Solve the full Keldysh problem with those additional self-energies.

## Suggested Implementation Path in `src/interface_chern`

The existing code path in `src/interface_chern` already organizes the problem in
the right mixed representation, so the proposal fits the current structure well.

### New ingredients

Add a new helper file, for example

```text
src/interface_chern/bulk_reset.jl
```

with routines along the following lines:

- `bulk_projector_symbol(kx, ky, r; t0=1.0)`
  returns `P(kx,ky)` and `Q(kx,ky)`.
- `bulk_projector_kernel(kx, dy, r; t0=1.0, Nky=...)`
  computes the Fourier-transformed kernel in `y-y'`.
- `restricted_bulk_projector(kx, rows, r; ...)`
  builds the projector matrix on a chosen explicit row set.
- `build_bulk_reset_self_energies(kxs, row_ids; ...)`
  returns the top and bottom reset matrices for all `k_x`.

### New solver inputs

Extend the main solver to accept parameters such as:

- `bulk_reset_enabled::Bool`
- `Gamma_bulk_top::Real`
- `Gamma_bulk_bottom::Real`
- `y_reset_top_lo::Integer`
- `y_reset_bottom_hi::Integer`
- `Nky_projector::Integer`

The top reset region would be all explicit rows `y >= y_reset_top_lo`, and the
bottom reset region all explicit rows `y <= y_reset_bottom_hi`.

### Changes to the Green's-function pass

At present the code forms

```math
G^K = G^R (\Sigma_{lead}^K + \Sigma_M^K) G^A
```

when monitoring is enabled.  The new bulk-reset proposal simply adds another
source term:

```math
G^K = G^R (\Sigma_{lead}^K + \Sigma_{reset,top}^K + \Sigma_{reset,bot}^K + \Sigma_M^K) G^A.
```

Similarly, the retarded inverse Green's function acquires

```math
- \Sigma_{reset,top}^R - \Sigma_{reset,bot}^R.
```

Since the reset self-energy is frequency-independent, it can be precomputed once
for each `k_x`.

### Data to save

Save at least:

- the full `C(k_x)` on the explicit window,
- the stabilized row sets,
- the reset rates,
- the `k_x` grid,
- and diagnostics comparing deep stabilized rows with the target bulk
  projectors.

## Numerical Diagnostics

The following checks should be carried out before interpreting any marker data.

### Bulk-only checks

For a uniform topological or trivial strip with the reset channel applied and no
interface:

- verify that the computed `C(k_x)` matches the corresponding bulk projector,
- verify Hermiticity of `C`,
- verify `0 <= C <= I` numerically,
- verify convergence with `Nky_projector`.

### Interface checks

For the coupled interface geometry:

- compare deep stabilized rows against the target bulk projector kernel,
- vary `\gamma_{top}` and `\gamma_{bottom}` to measure how quickly the bulk
  heals back to zero temperature,
- vary the size of the unstabilized interface buffer,
- and track the bulk-interface cross blocks as a function of reset strength.

### Marker checks

If the goal is a Chern-marker profile that uses the extended correlation matrix,
the marker should be built from the full explicit `C(k_x)` that includes the
stabilized bulk rows.  Truncating `C` before evaluating the commutator formula
changes the observable.

## Practical Interpretation

The proposal should be viewed as the clean Gaussian version of the statement

> In the bulk, measure in a band-adapted Wannier frame and reset the state
> toward the desired zero-temperature band sector.

At the trajectory level one may talk about projective measurements in a Wannier
frame.  At the averaged quadratic Keldysh level, however, the correct object is
the projector-reset dissipator described above.

That is the mathematically controlled way to combine:

- bulk stabilization toward zero temperature,
- a nontrivial interface steady state,
- and access to the bulk-interface cross-sector correlation matrix.

## Bottom Line

The proposed distributed bulk reset is:

- compatible with the current mixed `(k_x, y)` slab solver,
- naturally expressed using bulk projector kernels,
- valid on both trivial and Chern sides,
- and the most direct Gaussian realization of the intended Wannier-basis reset
  idea.

The main limitation is not conceptual but dynamical:

- at finite reset rate it anchors the bulks to the zero-temperature projectors,
  but does not freeze them exactly pointwise right up to the interface.

For the present problem, this still looks like the most natural next step.
