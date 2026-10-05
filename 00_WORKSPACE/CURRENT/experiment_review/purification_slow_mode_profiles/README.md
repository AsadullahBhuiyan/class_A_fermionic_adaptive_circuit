# Purification slow-mode profiles: slots 04 and 07

This analysis extracts resolved single-particle endpoint eigenmodes from the
600 completed slot-07 purification trajectories and compares their spatial
profiles with the 800 completed slot-04 trajectories. It does not rerun any
dynamics, change any raw data, or mix these data with the pure-state tangent
cocycle campaign.

Completed extraction: 600/600 slot-07 trajectories, with **989 finite
hard-wall modes and 186 finite soft-wall modes**. All extracted finite-mode
counts match the original GPU spectra. The maximum eigenvector residual is
1.96e-15 and the maximum CPU/GPU endpoint-spectrum difference is 3.56e-15.
All retained modes are individually separated under the declared 1e-10
occupation-spacing criterion; no finite subspace crosses a numerical cluster
into an excluded cap.

The primary 1e-9 cap is consequential: varying it between 1e-8 and 1e-10
changes the resolved count in 212/300 hard and 138/300 soft trajectories.
This sensitivity is retained in the sample diagnostics, not hidden by clipping.

## Scientific contracts

Both campaigns have Nx=20, nshell=1, alpha1=1, alpha2=30, maximally mixed
initialization, raster-y ordering, perfect correction, no postselection,
and complex128. Hard-wall spectra use only the active slab x=5,...,15,
after Born-conditioned exterior preparation; the exterior is excluded.

- Slot 04: hard walls, Ny=20,22,24,26,28,30,36,40, S=100 each, T=2Ny.
  It saves occupation spectra and the 16 slowest mode profiles at multiple
  checkpoints. We compare its final profiles and also export the number
  of resolved modes at every saved spectrum checkpoint.
- Slot 07: hard and soft walls, Ny=20,30,40, S=100 each, T=4Ny.
  The completed hard-v2 and corrected soft-v3 output trees are used separately.
  Full endpoint centered covariances G_final are the extraction inputs.

The two campaigns are independent ensembles at **different evolution times**.
Their comparison cannot isolate a time effect from individual-trajectory
variation, and their samples must not be pooled as one ensemble.

## Mode selection and numerical limitations

Diagonalize the occupation matrix C=(G_final+I)/2 on the appropriate spatial
domain. A mode is resolved if

\[
10^{-9}<\nu_j<1-10^{-9},\qquad
\lambda_j^{(T)}=\frac{\log(1-\nu_j)-\log\nu_j}{2T}.
\]

This is the existing occupation-cap convention, not the normalized-product
SVD cutoff used by the separate pure-state tangent analysis. Modes are ordered
by increasing absolute lambda. All resolved modes are saved, not merely one
eigenvector. No finite exponent is manufactured by clipping saturated
occupation eigenvalues. Cap sensitivity at 1e-8 and 1e-10 is saved per sample
as a diagnostic; the primary analysis remains fixed at 1e-9.

Each slot-07 NPZ includes the full occupation spectrum, all resolved complex
eigenvectors and signed rates, spatial x-profiles, source/sample identities,
eigensolver residuals, and comparison with the original saved GPU spectrum.
The global phase of each vector is fixed using its largest component.

Neighboring occupation eigenvalues separated by at most 1e-10 form a
numerical cluster. Individual vectors in such a cluster have basis freedom.
Cluster sizes and individual-mode separation flags are saved explicitly.
A finite subspace sharing a cluster with excluded saturated modes is flagged
and excluded from profile averages. An empty finite subspace is represented
by zero vectors and a NaN aggregate profile, never an invented slow mode.

The finite-mode count is **not an exact mathematical rank** or a proof that
all other exponents are infinite. It states what these double-precision
endpoint covariances resolve under the declared criterion.

## Spatial estimator and figure caption

For each eigenvector, p_j(x) sums squared amplitudes over y and both orbitals,
using the canonical orbital+2*x+2*Nx*y ordering. Each p_j sums to one.
The per-trajectory quantity is the equal-weight mean over its entire resolved
finite subspace. This is proportional to a projector diagonal and is invariant
under unitary basis rotations within a fully included degenerate subspace.
Then take an equal-weight mean over trajectories with nonempty, separated
finite subspaces; error bars are trajectory SEM, not mode SEM.

**Caption.** Resolved purification-mode spatial weights for (a) slot-04 hard
walls at T=2Ny, (b) slot-07 hard walls at T=4Ny, and (c) slot-07 soft walls at
T=4Ny, shown at Ny=20,30,40. Dashed gray lines mark x=5 and x=15. Legends state
the contributing trajectory counts S_res; soft-wall profile means are
conditional on a trajectory having a numerically resolved subspace, not
unconditional S=100 averages. Panel (d) shows the number of resolved modes,
averaged over **all 100 trajectories**, including zero counts, with SEM.
All cases start maximally mixed, use the scientific settings above, and are
analyzed trajectory first. Lines join data; there is no fit or extrapolation.

This does not supersede earlier capped log-polar analyses. Those studies
regularized the otherwise saturated full generator, applied additional
averaging/twirling, and tested cap stability of spectral claims. Here only
uncapped, finite trajectory-resolved endpoint directions are used. A small
or empty resolved subset is endpoint non-identifiability, not evidence that
physical edge modes are absent.

## Files and reproduction

- `endpoint_modes/{hard,soft}/NyNNN/sample_NNN.npz`: resolved slot-07 vectors.
- `tables/summary.csv`: per-case counts, profile coverage, and wall weights.
  Inner-wall weights use x=5,6,14,15; the broader common wall neighborhoods
  use x=4,5,6,14,15,16, capturing the soft wall's exterior-side peak too.
- `tables/slot07_sample_diagnostics.csv`: individual counts, cap sensitivity,
  numerical separation, residuals, and output hashes.
- `tables/slot04_time_resolved_counts.csv`: finite-mode counts at saved cycles.
- `tables/mean_profiles.csv`: plotted means and SEMs.
- `analysis_manifest.json`: checksum-verified source inventory and output hashes.
- `figures/purification_resolved_mode_profiles.{pdf,png}`: comparison figure.

```bash
cd 00_WORKSPACE/CURRENT/experiment_review/purification_slow_mode_profiles
python -m unittest -v test_profiles.py
python analyze_profiles.py
```

The script uses four local CPU processes with two BLAS threads each for
eigendecomposition. Plotting uses the repository's Computer Modern sans-serif
LaTeX setup and its existing plotting compatibility shim. Raw files are read
only; extracted NPZs are published atomically in this analysis folder.

## Related pure-state data

Slot 09 is a different, pure-state campaign with perfect correction and
domain walls enabled: Nx=20, Ny=24,28,32, hard/soft walls, alpha1=1,3,
alpha2=30, nshell=1, T=2Ny, and 100 samples per configuration. It saves
`final_frame` plus `final_ranks` for all 1,200 trajectories. Select only
the occupied columns of each padded frame; then P=F F^dagger and G=2P-I.
These permit reconstruction of physical endpoint covariances without replay.
The new uniform DW-off bulk-Chern campaign saved compact observables only;
the slot-17 Ny40 tangent products are not physical endpoint covariances.
