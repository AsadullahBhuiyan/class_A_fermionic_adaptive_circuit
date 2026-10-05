# Minimum absolute Hamiltonian eigenvalue: square-size scan

Nx=Ny=N=12,16,20,24,32,40; W^2=9; 100 realizations per size.
IID Gaussian zero-mean diagonal disorder at x=N/4 and 3N/4 only,
independent along y and between orbitals. Canonical CPU OW parent,
alpha1=1, alpha2=30, nshell1, trial X, hard-wall truncation.

Compute min(abs(E)) within each Hamiltonian, then the arithmetic disorder mean.
Original energy zero, no recentering or reflattening. Error bars: one SEM.
The fixed-half-filling gap and midgap chemical potential are separate diagnostics.
Wall separation and edge circumference both grow with N.

500 new spectra, plus the existing 100 spectra at 40x40 with original provenance.
Saved per-sample spectra, potentials, checksums, aggregate caches, executed notebook,
summary and sample CSVs, diagnostics, comparison to fixed Ny40, PDF and 300-dpi PNG.

Run run_tmux.sh. Resumable state receipts. CPUs40..55, 16 workers, one BLAS thread.
Only eigenvalues are calculated; no entropy or dynamics. Each new size has
active-wall and exterior checks against the canonical parent, trace checks, and
one full-space spectral cross-check against exact block reduction.
