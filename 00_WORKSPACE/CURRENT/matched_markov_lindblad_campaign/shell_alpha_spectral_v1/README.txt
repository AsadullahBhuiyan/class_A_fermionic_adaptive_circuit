Hard-wall shell / alpha channel-gap sweep
=======================================

315 independent spectral calculations: alpha_1=1.0,1.1,...,3.0;
Nx=20, Ny=20,40,60,80,100; n_shell=1,2,infinity. Shell 3 is excluded.
Infinity is Python None: no square shell cutoff, but the same hard-wall
region mask is retained. This is not a soft-wall or fully unmasked control.

Matches Figure 11(c): alpha_2=30, inclusive walls x=5,15, both interior and
periodic exterior active, periodic x/y, zero twist, X trial orbitals,
complex128, perfect correction with measurement dephasing, fixed raster-y
order (x outer, y inner), Ap/Am/Bp/Bm at each cell. At alpha_1=2 use the
canonical exact-zero Bloch-norm convention; do not shift alpha.

Use canonical CPU OW construction with the requested shell explicitly.
Reuse the established two-block projector-product and complete eigensolver.
No trajectory sampling or covariance time evolution: compute the one-cycle
A=Q_M...Q_1, g_C=1-rho(A)^2, and kappa_C=-2 log rho(A).
Save both block spectra, dominant left/right eigenvectors, residuals,
configuration, source hashes and timings. Unresolved unit-modulus modes are
kept and flagged. Old scripts and results are not modified or substituted.

Execution: python run_scan.py launch
Default: 21 independent single-core processes, lower priority, BLAS threads=1,
with six GiB/worker admission headroom and a 12 GiB reserve. Each worker has
one alpha and visits all sizes for shell 1, shell 2, then infinity. Dense
projector products cost substantially more than finite-shell products.
The queue lives in detached tmux and logs every case with a construction
progress bar and eigensolver messages. No automatic notification is installed.

Each case is resumable by checksum/config/source-verified NPZ + completion
JSON. Repeat launch with --root PATH after a stopped queue to skip completed
cases. Use report --root PATH to inspect progress. Final CSV and PDF/PNG
shell-comparison plots are generated only once all 315 pairs verify.

Interpretation: this measures cutoff sensitivity at fixed Nx=20, not a
two-dimensional thermodynamic limit or a theorem about arbitrary truncations.
