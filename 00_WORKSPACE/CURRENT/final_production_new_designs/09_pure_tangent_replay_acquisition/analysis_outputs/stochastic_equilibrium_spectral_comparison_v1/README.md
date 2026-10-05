# Stochastic versus equilibrium spectral comparison

Read REPORT.md and the executed stochastic_equilibrium_comparison.ipynb.

Accepted production analysis: 600 trajectories across Nx=20, Ny=24,28,32,40,50,60; matched equilibrium references; cut-origin and contour controls. Original datasets are unchanged.

Reproduce with analyze_comparison.py, extend_sizes.py, quantify_findings.py, build_notebook.py, then execute the generated notebook with nbclient. CPU allocation is editable in the notebook's first executable cell. Plot bins and endpoint tolerance can be changed without recomputing spectra. Original production checksums are verified by the analysis scripts.

The three postselection runner scripts are exploratory numerical diagnostics, not a stable late-time production recipe. Only accepted_postselection_cycle32.npz is used as the new finite-cycle postselection control. Read exploratory_controls_status.json before running those diagnostics; late native-frame evolution amplifies cross-sector roundoff. No stationary postselected state is claimed.

Uncertainties use independent trajectories; translated cuts are averaged inside each trajectory. Figures use CMU Sans Serif, inward ticks, vector PDF and 300-dpi PNG. Local latex_support/type1ec.sty is the existing repository compatibility shim.
