# Half-filled flattened-parent Chern reference

Computed locally on 2026-09-27 with `../../analyze_random_center_chern.py`.
One deterministic ground state for each of 20x20, 20x30 and 20x40; no circuit
dynamics and no trajectory ensemble. Canonical CPU OW construction, nshell=1,
alpha1=1, alpha2=30, hard support truncation, periodic boundaries, X trial
orbitals, and interfaces x=5,15. The parent is the established
sum of upper-band OW outer products minus lower-band OW outer products.
Occupy its lowest Nx*Ny single-particle modes.

The Chern partition matches bundle 23: x0=10, R=4, ten distinct integer y
centers drawn with its root seed 2026092701 and sample=0, cycle=0 center key.
Disk coordinates wrap at the y seam. The occupied-frame estimator uses
Gamma=(V V†)^T and is checked against explicit projector-block contractions.

| Geometry | Ten-center mean Chern number | Absolute deviation from 1 |
|---|---:|---:|
| 20x20 | 0.9999578452345611 | 0.0000421547654389 |
| 20x30 | 0.9999578494989485 | 0.0000421505010515 |
| 20x40 | 0.9999578510593468 | 0.0000421489406532 |

The maximum center-to-center spread is 2.78e-15. The ground-state projector
is y-translation invariant to 1.67e-15, so the ten centers are redundant
spatial checks, not independent samples. No SEM is assigned. The explicit
and frame calculations agree within 4.45e-16. The half-filled spectral gaps
are approximately 1.50e-9; no twist regulator or occupation smoothing was used.

`centers.csv` contains all 30 center values. Each `nx20_ny*.npz` stores the
occupied frame, its rank, the full single-particle spectrum, center coordinates,
and both Chern calculations. `summary.json` records parameters, source hashes,
diagnostics and product checksums.

This supplies an equilibrium baseline near C_G=1 for the same local estimator.
It does not establish that the stochastic circuit's steady state equals this
ground state: in particular, no Born-conditioned exterior preparation is
applied to this equilibrium reference. The uploaded dynamics bundle is unchanged.
