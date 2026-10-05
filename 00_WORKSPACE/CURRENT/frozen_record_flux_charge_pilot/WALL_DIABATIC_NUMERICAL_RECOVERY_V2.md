# Wall-diabatized spectral-pump numerical recovery v2

## Scope

The original S100 calculation completed 1,595 of its 1,600 primary endpoint
tasks.  The five missing tasks all belong to the hard-wall ensemble.  Their
saved endpoint frames are finite and well conditioned: their Gram and
projector residuals are at the level of a few times `1e-15`.  The failures
occurred during the subsequent dense spectral continuation, not during the
monitored preparation.

The immutable parent calculation remains
`N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1`.  Its runner,
configuration, results, completions, failure records, and logs are not edited
or replaced.

## Locked recovery set

The v2 recovery contains exactly these original sample IDs:

- `wall_diabatic_dense_N28x24_hard_sample_014`
- `wall_diabatic_dense_N32x24_hard_sample_012`
- `wall_diabatic_dense_N32x24_hard_sample_037`
- `wall_diabatic_nsh1_N24x24_hard_sample_092`
- `wall_diabatic_nsh1_N32x24_hard_sample_068`

No replacement trajectories are generated.  Each recovery task checksum-pins
the same endpoint NPZ/completion pair used by v1.

## Numerical correction

The physics is unchanged: the same flattened endpoint parent, 256-interval
signed flux grid, `1e-7` regulator, wall-diabatized branch rule, charge
observable, and acceptance thresholds are used.  Only the linear-algebra
implementation is hardened.

1. The NumPy Hermitian eigensolve remains the first full-spectrum attempt.
2. A failed or non-finite solve is retried with independent SciPy LAPACK
   drivers `evr`, `evd`, and `evx`.
3. Every fallback is checked for finite values, eigenpair residual, and
   eigenvector orthonormality before it is accepted.
4. If the assembled continued frame drifts beyond `1e-12`, symmetric polar
   orthonormalization is applied.  This changes the frame basis but preserves
   its occupied subspace and therefore preserves the physical projector.
5. If every driver fails, the failure record now includes the precise flux,
   subset, driver, and residual information.

The recovery publishes to the independent directory
`results/N20_24_28_32x24_wall_diabatic_spectral_pump_numerical_recovery_v2`.
Its completion receipts bind the parent runner/configuration, recovery
runner/configuration, endpoint dependency, and recovered result by SHA-256.

## Analysis rule

The primary analysis retains the 1,595 verified v1 pairs and fills only the
five absent `(cell, wall, sample_id)` keys from the verified v2 recovery.
Recovery rows must remain labeled by their recovery revision in any exported
sample table.  They must not be described as replacement samples or as a new
Born ensemble.
