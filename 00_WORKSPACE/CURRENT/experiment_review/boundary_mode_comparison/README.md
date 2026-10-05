# Physical-wall modes versus entanglement-cut modes

**Follow-up:** the nearest-four comparison below does not exhaust the modular
spectrum. The completed [full-spectrum scan](full_spectrum_scan/README.md)
examines all 165,816 numerically resolved modes across 1,200 pure endpoints.
It finds additional wall-concentrated modes, primarily at high absolute
modular energy and within a few rows of the entanglement cut. The earlier
conclusion about the four lowest modes must not be generalized to all modes.

This completed, trajectory-first analysis investigates boundary localization in
600 slot-07 mixed-state purification endpoints and 1,200 slot-09 pure-state
occupied-frame endpoints. No dynamics were rerun and no original data changed.

## Main finding

**Slot 07 directly resolves modes localized at the physical domain walls.**
For Ny=40 the trajectory-averaged weight within x=4,5,6,14,15,16 is
96.70% +/- 0.28 percentage points (hard) and 95.53% +/- 0.88 percentage points
(soft). Errors are trajectory SEM. Hard includes 100/100 trajectories; soft is
conditional on the 52/100 trajectories having a numerically resolved mode.
The corresponding Ny=20 and 30 results are also approximately 96%.

**Slot 09's half-system modular modes are primarily localized at the
entanglement cut, not along the physical walls.** The default subsystem is all
x and y=0,...,Ny/2-1, retaining both orbitals and both domain walls. This is the
existing repository half-y convention, explicitly chosen for this investigation.
At Ny=32 the averages over the four lowest-absolute-modular-energy modes are:

| Walls | alpha1 | Physical-wall weight | Entanglement-cut weight |
|---|---:|---:|---:|
| hard | 1 | 46.94% | 94.51% |
| soft | 1 | 47.16% | 94.27% |
| hard | 3 | 25.83% | 99.77% |
| soft | 3 | 26.58% | 99.43% |

Each row uses all 100 independent trajectories. The entanglement-cut weight
uses the two rows adjacent to each cut: y=0,1,Ny/2-2,Ny/2-1. Physical-wall
and entanglement-cut windows overlap and their weights must NOT be added.
Results at Ny=24 and 28 are similar. Complete means and SEMs are in
`summary.csv` and `summary.json`.

The maps show that the alpha1=1 pure-state modes are enhanced where the cut
meets the walls, but remain extended across the slab along the cut. In the
alpha1=3 comparison they primarily occupy the interior of the cut. This is
not a failure to reproduce slot 07: the observables, initial states, and
evolution times differ. It is an estimator/protocol distinction, not evidence
that physical edge physics is absent in a pure state.

## Why the two eigenproblems differ

Slot 07 starts maximally mixed and evolves for T=4Ny with perfect correction,
Nx=20, alpha1=1, alpha2=30, nshell=1, raster-y ordering and no postselection.
Its endpoint occupation matrix is generally mixed. The existing verified
extraction diagonalizes it on the hard active slab (x=5,...,15) or the full
soft-wall system. The resolved eigenvectors are shared by the endpoint modular
Hamiltonian, with energies epsilon=log(1-nu)-log(nu). The previous analysis
expressed the same spectrum as epsilon/(2T). We use all modes satisfying
1e-9 < nu < 1-1e-9. The counts depend on this cap and zero-resolved-mode samples
are not evidence of zero physical modes. Source cap-sensitivity diagnostics
remain in `../purification_slow_mode_profiles/`.

Slot 09 starts pure and evolves for T=2Ny, with the same other parameters but
alpha1=1 or 3, Ny=24,28,32, hard/soft walls. Use only `final_ranks` columns of
each saved padded `final_frame`. The full-system occupation projector
P=F F^dagger has eigenvalues 0 and 1. Its occupied eigenvectors, including
individual columns of F, can rotate arbitrarily and are not uniquely defined
physical boundary eigenmodes. A full-system modular Hamiltonian also has
divergent energies for an exactly pure state.

Instead restrict rows of F to A, form G_A=F_A F_A^dagger, and diagonalize G_A.
Its eigenvectors are the single-particle modular eigenmodes with
epsilon=log[(1-nu)/nu]. This procedure is gauge-invariant under F -> F U and
does not construct the full-system covariance. It gives a meaningful
entanglement spectrum but introduces an entanglement boundary. All correlation
and vector conventions follow the saved occupied-frame convention.

For every pure trajectory save the full occupation spectrum and resolved
modular energies, plus the 16 closest-to-zero complex eigenvectors and their
spatial densities. Extend a selection if its boundary has an absolute-energy
tie (tolerance 1e-8). Plot the equal-weight mean over the four closest-to-zero
modes, then average over independent trajectories; never average covariance
matrices or eigenvectors before diagonalization. Modes are chosen spectrally,
not by how wall-localized they look. Full-spectrum wall/cut weights are also
saved, so other spectral windows can be examined without another eigensolve.

These are called **low-modular-energy modes**, not automatically in-gap modes.
No bulk gap has been established for a trajectory spectrum. At Ny=32 the median
minimum absolute modular energy is approximately 0.086 (hard) / 0.082 (soft)
for alpha1=1, versus 4.091 / 4.093 for alpha1=3; the latter also has occasional
low-energy outlier trajectories, so the comparison is not an assertion of a
strict ensemble-wide gap. Boundary localization alone is not a chirality test.

## Figures and captions

- `figures/boundary_profiles.pdf` and `.png`: comparison of x profiles for
  all three sizes in each dataset. Left column: slot 07 all finite modes,
  hard above / soft below, maximally mixed initialization, T=4Ny,
  Ny=20,30,40. Middle and right: slot 09 four closest-to-zero modular modes,
  pure random initialization, T=2Ny, Ny=24,28,32, alpha1=1 and 3 respectively.
  All cases have Nx=20 and the settings above. Means are first over selected
  modes within a trajectory, then over trajectories, with trajectory SEM.
  S labels contributing trajectories, not modes; slot-07 soft results are
  conditional subsets (55,54,52 of 100). Dashed lines mark physical walls
  x=5,15. Lines join lattice points; no fitting or extrapolation is used.
- `figures/representative_boundary_modes.pdf` and `.png`: orbital-summed
  squared amplitude of the single closest-to-zero resolved mode, at the largest
  available size (Ny=40 slot07 / Ny=32 slot09). For each case choose the
  trajectory nearest the median selected-subspace wall weight, breaking ties
  by sample index; this is not a search for a maximally localized example.
  Mode vectors are individually separated under the declared criterion.
  Slot07 shows the full y domain; slot09 only the retained half y domain.
  Cyan vertical lines mark physical domain walls; horizontal edges of the
  slot09 panels are entanglement cuts. Each mode is normalized individually,
  and each panel has its own color scale. Maps illustrate individual modes;
  quantitative ensemble claims use the preceding profile figure and tables.

## Provenance, files, and validation

- `modes/slot07/`: 600 compact derived files; links and SHA-256s bind them to
  the previous checksum-verified eigenvector extraction, whose manifest in turn
  binds the original slot07 raw covariance outputs.
- `modes/slot09/`: 1,200 files holding spectra, 16 selected complex vectors,
  xy densities, full-spectrum boundary weights, separation flags, frame Gram
  and eigen residual diagnostics, and the entire source completion receipt.
- `manifest.json`: 1,800 sample identities, input/output SHA-256s, numerical
  conventions, and per-trajectory localization diagnostics.
- `summary.json`, `summary.csv`: 18 independent cases, exactly 100 samples each.

All 32 raw slot09 NPZ/receipt pairs passed size, hash and identity checks.
All 600 existing slot07 extraction hashes passed. The maximum pure-frame Gram
error is 7.10e-14 and maximum extracted eigenvector residual 2.73e-15.
All 19,200 retained pure-state modes are individually separated using a 1e-10
neighboring occupation spacing criterion. Raw saturated occupation eigenvalues
are preserved; modular energies are NaN when unresolved, never manufactured
by clipping. The derived mode files occupy about 241 MiB.

`test_analysis.py` checks a known Schmidt spectrum, occupied-frame gauge
invariance, spatial indexing and normalization, tied-mode selection, and
rejection of a nonorthonormal frame. Local execution uses four worker processes
with two BLAS threads each; it does not start or resume any circuit campaign.

```bash
cd 00_WORKSPACE/CURRENT/experiment_review/boundary_mode_comparison
OPENBLAS_NUM_THREADS=2 python -m unittest -v test_analysis.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u analyze.py
```
