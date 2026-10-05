# Mixed-mode half-system size comparison

Run half_system_mixed_mode_size_overlay.ipynb on CPU. The first code cell exposes an inclusive CPU range. Uses all 100 hard-wall alpha_1=1 trajectories at each Ny=24,28,32, Nx=20, endpoint cycles 2Ny.

Each subsystem covers all x, y=0..Ny/2-1, both orbitals. Raw centered spectra are cached separately per size. Input hashes and metadata are verified before calculation or cache reuse. Existing Ny32 derived spectra may be reused after identity verification. Earlier analysis and source data are not modified.

The overlay retains abs(lambda)<1-1e-8, with separate unit-area normalization for each size. All curves use 100 common bins and a log vertical axis. Edit MIXED_TOL and BIN_COUNT in the plot cell to rebin without repeating diagonalization.

Products: executed notebook, per-size raw NPZ and provenance JSON, combined histogram CSV, overlay diagnostics JSON, caption, vector PDF, 300-dpi PNG. The local LaTeX compatibility shim and pdftoppm conversion match the earlier figure workflow.
