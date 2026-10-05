Square hard-wall channel spectral gap versus alpha (2026-10-01)

Approved grid: alpha_1=1.0,1.1,...,3.0; Nx=Ny=20,40,60 (63 cases).
alpha_2=30, nshell=1, inclusive walls floor(L/4),floor(3L/4), all slabs
active, hard-wall support truncation, X trial orbitals, periodic boundaries,
zero twist, raster-y Ap/Am/Bp/Bm, complex128, perfect correction and number
measurement dephasing. No trajectories, initial condition, or cycle horizon.

Reuse the validated square_large_spectral_v1 block construction/eigensolver
and canonical CPU OW construction. Diagonalize the full interior AND periodic
exterior blocks; retain every eigenvalue and dominant left/right eigenvectors.
Delta_C=-2 log rho(A) is the covariance-sector decay rate per cycle, not a
measurement of the full many-body gap. No modes are discarded to create a gap.
Near-unit modes are flagged, and the raw rate is saved without clipping.

Alpha=2 is INCLUDED EXACTLY, not nudged off the critical value. The canonical
engine replaces an exactly zero Bloch-vector norm by 1e-15 when constructing
the band arrays (zero numerator stays zero). No new regularization is added.
Tiny tests compare fractional alpha and alpha=2 to the full dense product.

Scientific sources and previous campaigns remain unchanged. Each case saves
one spectrum.npz and checksum/configuration/source-bound completion.json;
valid completed pairs are skipped on restart. Only interrupted cases repeat.
Random probes check the product action but are not trajectory sampling.
Per-case logs show the configuration, Q-product progress and eigensolver
start/end messages. Three workers use one core each, with all library thread
counts one. The task assignment gives each worker seven cases of each size.

Launch: python run_scan.py launch --cpus 0,7,40
Status: python run_scan.py report --root RESULTS_DIRECTORY
Resume: python run_scan.py launch --root RESULTS_DIRECTORY --cpus 0,7,40
Do not change source files while running; source identity is checked at commit.

The detached tmux queue produces an analysis/gaps.csv and a PNG/PDF plot of
gap versus alpha with one curve per size after all 63 pairs verify. It does
not refit or subtract a mean. Failed cases remain incomplete and are logged.
Current case-level resume deliberately does not checkpoint an eigensolver.
