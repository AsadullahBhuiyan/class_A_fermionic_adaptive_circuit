# Ny=40 endpoint singular-vector extraction

This is local post-processing of the 200 saved slot-17 endpoint cocycles:
100 trajectories each at alpha1=1 and alpha1=3, Nx=20, Ny=40, T=80.
No circuit is rerun and no raw file or original gap result is changed.

Extraction completed for **200/200 samples**. The NPZ outputs total
7,052,026,132 bytes (about 6.57 GiB). The worst relative matrix-reconstruction
error is 1.94e-14 and the worst orthogonality residual is 4.67e-15.
Numerically resolved rank ranges are 14--20 at alpha1=1 and 696--715 at
alpha1=3 under the saved-matrix threshold below; these are not a replacement
for the source blockwise tangent-null counts.

## What can and cannot be recovered

The saved matrix is the 880 x 1600 active-row representation of the
chronological one-leg product. The full-matrix thin SVD is

\[
K[\mathrm{active},:]=e^\ell\widehat K,
\qquad \widehat K=U\operatorname{diag}(s)V^\dagger.
\]

All 880 thin-SVD triplets are saved, in descending singular-value order,
together with the scale-separated logarithms log(s)+ell. There is no need
to exponentiate the scale, which could cause overflow or underflow.

**These are not yet occupied-empty slow tangent pair modes.** The original
prepared frame F0 and record were transient at Ny=40 and are absent from the
persisted products. Initial random frames and exterior Born draws used CUDA
random numbers. A same-seed CPU initialization is not a faithful replacement.
The source occupied/empty block dimensions and five pair indices/rates are
retained as provenance, but they do not specify the missing basis, so we do
not infer labels by matching rates. Identifying the original physical pair
modes requires a verified reconstruction of that initial projector (or a
separately saved frame), beyond this matrix-only extraction.

## Direction convention

The source defines the covariance action as H_T=K^dagger H_0 K. Consequently:

- U lives on the **initial active** coordinates (880 rows).
- V lives on the **endpoint full** coordinates (1600 rows).
- An initial matrix direction U_i U_j^dagger maps to
  exp(2*ell) s_i s_j V_i V_j^dagger, before any physical tangent-space restriction.

This is intentionally not labeled using the usual right-input/left-output
shorthand for the action of K itself. The scientific action uses K^dagger
on the left. The original occupied-empty pair indices must not be used to
index these full-matrix singular vectors.

## Arrays in each sample NPZ

- `left_vectors_initial_active`: complex128 U, shape (880,880).
- `right_vectors_endpoint_dagger`: complex128 V^dagger, shape (880,1600).
- `singular_values_normalized`: all 880 normalized singular values.
- `chronological_cocycle_log_scale`: original logarithmic scale ell.
- `raw_log_singular_values`: log(s)+ell, including unresolved values.
- `resolved_mask`, `numerical_rank`, `numerical_null_count`:
  numerical-rank diagnostic using eps*max(matrix.shape)*s_max on the saved matrix.
- `resolved_log_singular_values`, `resolved_one_leg_rates_per_cycle`:
  unresolved entries are NaN, not invented finite exponents or physical zero gaps.
- `one_leg_nearest_zero_mode_indices`: up to 16 resolved **whole-matrix one-leg**
  rates closest to zero. They are not the source occupied-empty pair modes.
- `initial_x_profiles`, `endpoint_x_profiles`: (880,20) normalized squared-amplitude
  profiles, summed over y and both orbitals. The source coordinate convention is
  index=orbital+2*x+2*Nx*y; active row indices are saved explicitly.
- Original task/sample identity, source hashes, pair-rate diagnostics, and SVD
  reconstruction/orthogonality residuals.

Singular vectors below the numerical threshold are saved only to complete the
thin decomposition. Their orientations are arbitrary and must not be interpreted
as physical slow modes. Degenerate singular subspaces also have basis freedom.
The thin SVD does not include the extra 720 endpoint right-nullspace vectors.
The analysis threshold diagnoses this saved matrix; it is **not** substituted
for the source blockwise cutoff or used to revise the previous gap figure.

## Loading a sample

```python
import numpy as np

z = np.load('endpoint_singular_modes_ny40_v1/alpha1_1/sample_000.npz')
U = z['left_vectors_initial_active']
V = z['right_vectors_endpoint_dagger'].conj().T
s = z['singular_values_normalized']
Khat = (U * s) @ V.conj().T
valid = z['resolved_mask']
endpoint_profile = z['endpoint_x_profiles'][valid]  # mode x x-coordinate
```

## Reproduction

```bash
python -m unittest -v test_extract_endpoint_singular_modes.py
python extract_endpoint_singular_modes.py --workers 4 --threads-per-worker 2
```

Four CPU processes each use two BLAS threads. Each independent sample writes
one NPZ and a checksum-bound completion JSON; rerunning verifies and skips valid
results. `--max-new-samples 1` is a cheap first-sample check. All 200 full thin
decompositions require approximately 6.6 GiB. `manifest.json` records source
identities, output hashes, numerical ranks, and verification residuals.
