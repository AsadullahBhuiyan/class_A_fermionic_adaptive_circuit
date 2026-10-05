Extended square hard-wall spectral sweep (2026-09-30)

Six new cases: Nx=Ny=60,80,100 and alpha_1=1,3. Same scientific
contract as square_2Ny_v1 spectral references: alpha_2=30, nshell=1,
inclusive walls [floor(L/4),floor(3L/4)], support truncation, all slabs
active, X trial orbitals, zero twist, periodic boundaries, perfect
correction, measurement dephasing, raster-y Ap/Am/Bp/Bm, complex128.
No explicit dynamics or sampled trajectories. No cycle horizon applies.

Canonical CPU OW construction is unchanged. Hard-wall OW modes have
strictly zero support in the opposite wall sector. Therefore the ordered
product A has two invariant blocks: interior and the entire periodic
exterior (the exterior connects across the x boundary). Diagonalize BOTH
blocks completely, concatenate all eigenvalues, and take the largest
modulus over their union. This is an exact reordering, not active-slab
dynamics, not a frozen exterior, not a Fourier approximation, and not an
iterative estimate of only a few eigenvalues. Delta_C=-2 log rho(A).

Validation: strict zero cross-sector support, normalization, independent
full-system projector action on random probes, dominant left/right
residuals, and full-system dominant action. Tiny cases compare the entire
block spectrum and matrix against the original full dense product. Saved
unit-modulus modes are flagged, not discarded. Completion pairs bind
configuration, sources, filename, bytes, checksum; completed cases resume.

L20 regression caveat: comparing every saved eigenvalue with absolute
forward tolerance 1e-9 failed for near-zero eigenvalues (about 2.6e-4
displacement for alpha=1), while the dominant gap agreed to 2.7e-15.
validate_existing.py preserves this discrepancy and checks full matrix
equivalence and every block eigenpair's backward residual. These nonnormal
near-zero modes must not be advertised as individually forward-accurate;
the requested spectral radius/gap is separately checked against the saved
full-matrix calculation. See validation_against_L20.json for both alphas.

Launch:
  python run_large.py launch --cpus 0,7
Report:
  python run_large.py report --root RESULTS_DIRECTORY
Resume (creates a tmux queue; do not run while same queue is active):
  python run_large.py launch --root RESULTS_DIRECTORY --cpus 0,7

Two single-core subprocesses for L60/L80, serialized L100 cases to limit
peak canonical OW-construction memory. Existing unrelated jobs untouched.
Queue and per-case logs retain progress and eigensolver start/end events.
Failed or memory-rejected cases are not called complete. Old data untouched.
After six verified results, analyze_large.py combines old L20--50 spectra
with new data, producing raw and mean-centered PNG/PDF figures and CSV.
The centered plot then uses the mean over all eleven sizes, not the old
eight-size mean. Mean subtraction is not an estimate of the infinite-size
gap; the crossing is forced by centering and cannot establish gap closure.

These are covariance-sector rates, not a measurement of the full many-body
channel spectral gap or proof of a positive thermodynamic gap.
