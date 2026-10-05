# Equilibrium disorder: central-charge fits
Same 900 saved realizations as the nine-panel gap-ratio comparison.
Nx20 Ny32, iid orbital-resolved Gaussian potential on x=5,6,14,15.
W^2=1,2,3,4,6,9,12,16,25, 100 samples each; one clean reference.
Reconstruct the occupied projector from the saved canonical CPU Hamiltonian
and exact saved potentials. Fill globally lowest 640 energies.
Full von Neumann entropy, no spectral window. Average 32 periodic origins
within each realization, then realizations. Save all widths Ay1..16.
Fit S=s0+(c_fit/3)log[(Ny/pi)sin(pi Ay/Ny)] over Ay5..16;
also Ay2..16 and Ay8..16. Unweighted free-intercept OLS.
c_fit is a finite-size total-strip coefficient, not per wall.
Coefficient errors: one SEM across realization profiles, preserving all
cross-width correlations. Bootstrap whole realizations, not cut origins.
Sampling errors exclude finite-size, model, and fit-range effects.
Run run_analysis.py then build_notebook.py. Acquisition resumes using
verified per-realization receipts. Executed notebook has editable plots.
Saved: raw origin entropies, provenance, fits, bootstrap, diagnostics,
vector PDF and 300-dpi PNG figures, completion checksums.
