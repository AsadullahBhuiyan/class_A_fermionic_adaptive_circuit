# Microscopic numerical Hall protocol

This is the execution-facing summary of the Hall proposal. It does not use a
Keldysh calculation, a current-current correlator, a frozen measurement word,
or postselection.

## What is exact before Monte Carlo

For a normalized controller projector `P`, `Q = 1 - P`, and target occupation
`s in {0,1}`, perfect correction gives the exact Born-mean update

```text
C_mean(next) = Q C_mean Q + s P.
```

This analytically sums every Born outcome at that event. The canonical GPU
engine already exposes this as `mean_replacement=True`. Random site order is
still sampled, so each batch member can represent an independent realization
of the exogenous space-time disorder.

The corresponding mean ancilla transfer is

```text
y_mean = s - Tr(P C_mean).
```

Therefore the Born-mean flux pump should be screened deterministically before
spending trajectories on its distribution.

## Cheap reduced-channel pump

For `Nx = 20`, use complementary half-system regions

- `R_L`: x = 0,...,9, containing the left interface;
- `R_R`: x = 10,...,19, containing the right interface.

The actual interfaces are the bonds `(4,5)` and `(15,16)`, while the two region
cuts lie inside the topological and trivial sectors. Stream only

```text
N_a(c) = 1/2 Tr[R_a (1 + G(c))]
A_a(c) = sum_events (target - outcome) Tr(R_a P_event)
q_a(c) = N_a(c) - N_a(0) - A_a(c)
q_x(c) = (q_R(c) - q_L(c))/2.
```

The event field already called `transfer` is `target - outcome`. In seam gauge,
`Tr(R_a P_event)` is flux independent and can be precomputed. This source-
subtracted regional balance is an exact and cheap generalized monitored-channel
pump. Record by record it also contains conditional measurement-innovation
noise, so it is not yet literal electrical current.

## Literal electrical current

For the phrase “electrical Hall conductance,” compile every `nshell = 1`
controller orbital into a fixed local nearest-neighbor Givens tree:

1. rotate the controller orbital onto its center anchor;
2. perform the local number measurement and filled/empty ancilla fSWAP reset;
3. apply the inverse Givens tree.

For every charge-conserving elementary gate `u_g`, count

```text
q_x,g/e = Tr[R (u_g C_g u_g^dagger - C_g)]
```

when the declared gate path crosses the x cut. Summing these terms is the
trajectory electrical current. Validate two different local Givens trees and
two cut placements; a quantized large-scale coefficient must agree.

## Flux protocol

Burn in at zero twist for `2 Ny` cycles. Then ramp the seam holonomy through
one circle with

```text
phi(c) = sign * 2 pi * [u - sin(2 pi u)/(2 pi)],  u = c/T_phi.
```

Use both signs. Every cycle receives a new random site schedule and every
measurement is sampled from its instantaneous Born probability. Do not freeze
or replay outcomes.

A proof-of-concept needs no engine rewrite: the existing cycle observer can
record cycle `c` and then call `set_controller_twist(phi[c+1], gauge="seam")`
for the next cycle. Production should add a versioned
`controller_twist_schedule` argument to the canonical engine for cleaner
metadata and validation.

The primary reduced estimator is the flux-odd part

```text
C_odd(T) = orientation_sign * [mean q_x(+2pi) - mean q_x(-2pi)]/2.
```

Fit `C_odd(T) = C_inf + a/T`, and test an added `b/T^2` term. Never reweight
ordinary Born-sampled trajectories by their stored record probability.

## Staging and hyperparameters

All stages use `nshell = 1`, local backend, `alpha_top = 1`,
`alpha_triv = 30`, pure half-filled initialization, perfect correction,
random schedules, complex128, and no covariance histories.

| Stage | Geometry | Samples | Ramp times | Cases |
|---|---:|---:|---:|---|
| Smoke | 12 x 16 | 1 | `Ny` | interface, exact continuity checks |
| Discovery | 20 x 24 | 5 | `4Ny`, `8Ny` | interface forward; reverse at `8Ny`; trivial at `8Ny` |
| Pilot | 20 x 24 | 10 | `4Ny`, `8Ny`, `16Ny` | interface forward; mandatory reverse and trivial controls |
| Production | 20 x 24 | 25, shards of 5 | `4Ny`, `8Ny`, `16Ny` | forward all rates; interface reverse and trivial +/- at `8Ny` |
| Size check | 20 x 32 | 25 | `8Ny` first | interface and matched trivial |

The validation timing gives a rough planning range of 15–25 aggregate A100
hours for the 20 x 24 production matrix, before new-observer/cache overhead.
Three Colab lanes should complete it in roughly a day once the implementation
passes smoke tests. Compact streamed output should stay below 1 GB. Never save
full covariance histories: one such 25-trajectory slow-ramp case would be on
the order of 150 GB.

## Claim ladder

1. Deterministic Born-mean plateau: exact in intrinsic Born outcomes for each
   sampled schedule.
2. Born-sampled reduced-channel pump: mean, confidence interval, and complete
   record distribution without postselection.
3. Compiled gate-current pump: literal electrical Hall transport of a specified
   local system-ancilla circuit.
4. Record typicality: decreasing variance and bad-record fraction with size and
   ramp time. Twenty-five records can support an initial distributional claim,
   not a theorem about every possible record.

H2 remains the independent real-time wall-chirality test. H3 remains a single
frozen-record static entanglement-flow diagnostic. Neither is this pump.
