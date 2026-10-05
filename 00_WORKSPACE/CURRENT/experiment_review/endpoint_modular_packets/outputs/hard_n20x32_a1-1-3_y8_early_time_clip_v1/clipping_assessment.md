# Early-time clipping assessment

All 100 trajectories per alpha1 and all 32 periodic cut origins were analyzed.
The original eight source shards passed their completion checksums. No circuit
simulation was rerun. Old long-time results are preserved in their separate
directory. Each cutoff uses precisely the same occupied frames, covariance
eigenvectors, packet sources, and averaging procedure.

## What is stable, and what is not

For alpha1=1, the left-wall ensemble-mean displacement is positive and the
right-wall displacement negative at every saved nonzero time from 0.01 to 1,
for all three thresholds. The magnitude and oscillation frequency are not
cutoff-independent. At t_mod=0.2:

| covariance clipping epsilon | left mean displacement | right mean displacement |
|---|---:|---:|
| 1e-8 | +0.375290 | -0.392166 |
| 1e-10 | +0.557219 | -0.570929 |
| 1e-12 | +0.737043 | -0.737094 |

These changes greatly exceed trajectory SEM (roughly 0.002--0.006 here).
The paired differences and their trajectory SEM are in displacement.csv.
Thus the sign is robust across this cutoff family; interpreting the absolute
speed as a regularization-independent velocity is not warranted.

The alpha1=3 control has maximum absolute ensemble-mean displacement below
0.00263 over the full time grid for every cutoff, versus 0.71--0.93 for alpha1=1.
It is weakly mobile, not exactly stationary. All comparisons concern the stated
wall-window estimator, not an unwrapped bulk propagation velocity.

## Why the rapid ripples are cutoff-dependent

For centered covariance eigenvalues clipped at +/- (1-epsilon), the modular
energies are capped at +/- Ecap, Ecap=2 atanh(1-epsilon). The extreme-branch
interference period is 2 pi/(2 Ecap). Observed left-wall peak spacings after
t_mod=0.3 closely follow this prediction:

| epsilon | predicted extreme-branch period | measured median peak spacing |
|---|---:|---:|
| 1e-8 | 0.16436 | 0.16 |
| 1e-10 | 0.13245 | 0.13 |
| 1e-12 | 0.11092 | 0.11 |

The measured spacing is quantized by the 0.01 time grid. It was obtained by
scipy.signal.find_peaks on the unsmoothed left-wall mean and taking consecutive
peak differences for peaks later than 0.3. This paired intervention provides
strong evidence for cutoff-controlled ripples, not for a minimum-gap timescale.
It does not assert that every feature of the packet dynamics comes from the cap.

At epsilon=1e-10, the median packet spectral weight on clipped modes is about
85% on either wall for alpha1=1. The corresponding packet energy standard
deviations are 22.21 and 22.25, while the median minimum absolute modular
energy is only 0.08756. These are medians across individual sample/cut generators,
not eigenvalues of an averaged Hamiltonian. The minimum absolute energy is
unchanged across the cutoff family; the observed fast timescale is not.
For alpha1=3, nearly all packet spectral weight lies on clipped modes, yet
the packet is much less mobile: spectral widths alone cannot establish spatial
transport or chirality; eigenvector structure and observable matrix elements
matter as well.

## Validation and interpretation

- Four propagation tests pass, including comparison with direct matrix
  exponentials at all three cutoffs on synthetic covariances.
- The recomputed epsilon=1e-10 alpha1=1 per-trajectory displacement agrees
  exactly with the previously saved 0..1 segment (maximum difference zero).
- Aggregate means and trajectory SEM were independently recomputed from the
  sample arrays; packet charge is conserved within 1e-10. Maximum observed
  errors: 6.44e-15 for alpha1=1 and 8.18e-11 for alpha1=3.
- All 200 cache payload checksums were verified during aggregation.

Panels (a,c) now show t_mod=0,0.1,0.2; (b) shows 0..1 at the original epsilon=1e-10.
They display early handed motion and a weakly mobile control using observable
averaging. This check supports robustness of the displacement signs, not a
cutoff-independent modular speed or a proof of chirality from three snapshots
alone. It concerns reduced-state modular evolution, not a full-circuit polar
generator or physical circuit-time transport.
