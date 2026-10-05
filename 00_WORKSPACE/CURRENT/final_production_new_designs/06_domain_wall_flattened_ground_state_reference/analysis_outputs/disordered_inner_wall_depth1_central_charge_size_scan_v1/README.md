# Equilibrium disorder: central-charge size scan

Tmux session: disorder-c-size-scan.
Attach with: tmux attach -t disorder-c-size-scan
Progress: run.log. Final shell status: exit_code.txt (0 means success).
The tmux pane remains visible after the pipeline exits.
Resume with bash run_tmux.sh from this directory. Verified completed
realizations are reused; incomplete or mismatched receipts are recomputed.

Nx=20; Ny=36,40,44,48 newly acquired; Ny=32 reused from the preceding
central-charge analysis. Nine variances W^2=1,2,3,4,6,9,12,16,25.
100 independent realizations per size/strength plus clean references.
Canonical CPU classA_U1FGTN OW parent, hard walls x=5,15,
alpha1=1, alpha2=30, nshell=1. Add independent mean-zero Gaussian diagonal
potentials only at x=5,6,14,15, independently for each y and orbital.
No reflattening. Globally occupy the lowest 20*Ny energies.

All Ay=1..Ny/2; average full entropy over all Ny periodic origins within
each realization before averaging realizations. No spectral window.
Fit S=s0+(c_fit/3)log[(Ny/pi)sin(pi Ay/Ny)] with a free intercept.
Primary: Ay>=5. Sensitivities: Ay>=2, Ay>=8,
Ay>=ceil(5Ny/32), and Ay>=ceil(Ny/4), always ending at Ny/2.
The fractional ranges allow comparison at fixed relative subsystem widths.
Coefficient errors are realization SEMs; bootstrap whole realizations.
No asymptotic extrapolation law is assumed.

CPU range 8..55; 48 processes, one BLAS thread each.
run_analysis.py first validates canonical construction and full-matrix
entropy cross-checks, then writes independent NPZ/JSON checkpoints.
Per-state files retain entropy by origin, disorder potential, Hamiltonian
energies, and half-system centered spectra. Matrices are validated before
symmetrization; finite eigenvalues, physical bounds, traces, purity,
complement symmetry, and global filling are checked.
build_notebook.py then executes the analysis and exports:
- central_charge_size_trends.pdf/png
- central_charge_disorder_by_size.pdf/png
- entropy_fits_across_sizes.pdf/png
- central_charge_size_fits.csv, all_fit_windows.csv, size_trends.csv
- disordered_equilibrium_central_charge_size_scan.ipynb
- input provenance, bootstrap results, diagnostics, and completion manifest

PNG outputs use 300 dpi; PDFs retain vector geometry.
The fitted coefficient describes the total strip at finite Nx and Ny.
Sampling errors exclude fit-window and finite-size/model uncertainty.
