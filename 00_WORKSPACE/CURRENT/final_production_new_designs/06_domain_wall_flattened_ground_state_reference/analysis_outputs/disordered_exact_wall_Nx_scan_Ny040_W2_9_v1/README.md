# Wall-only disorder: transverse size scan

Ny40, W^2=9, Nx12,16,20,24,32,40. Walls at Nx/4 and 3Nx/4;
zero-mean iid Gaussian potentials on both orbitals of the wall columns only.
100 independent ground states per Nx; fixed global half filling. Canonical CPU
OW parent (alpha1=1, alpha2=30, nshell1, trial X, dw_truncation=True).
Full entropy in nats, all widths1..20 and all40 periodic origins, all x/orbitals.
Average entropy over origins within each state, then disorder. Two free-intercept
OLS fits, Ay5..20 and Ay8..20; c_eff=3*slope. Error bars are realization SEM.
Bootstrap whole profiles to retain correlations across widths.

The Nx20 point reuses the completed wall-only Ny40 W2=9 ensemble with original
checksums and seeds. New sizes use seed [2026092710,Nx,40,sample_id].
Changing Nx also changes wall separation and transverse bulk width.
This study alone cannot separate all width effects from inter-wall overlap.

Run run_tmux.sh; uses CPUs40..55, 16 workers, one BLAS thread each.
Resumable per-state receipts and per-width rolling checkpoints. Each new size
gets direct full-space validations including wrapping cuts. The Nx20 reference
fit is cross-checked against the prior analysis. Figures refresh after each size.
The manifest labels incomplete size snapshots explicitly.
Final artifacts: executed notebook, sample-resolved entropy profiles/potentials,
fit and profile CSVs, bootstrap coefficients, diagnostics, vector PDFs and
300-dpi PNGs. No circuit simulation or post-disorder reflattening.
