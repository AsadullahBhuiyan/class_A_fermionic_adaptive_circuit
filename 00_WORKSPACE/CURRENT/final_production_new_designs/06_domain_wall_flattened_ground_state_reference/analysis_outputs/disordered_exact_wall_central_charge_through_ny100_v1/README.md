# Disorder only on the domain walls

Repeat the prior two-panel c_eff figure for Nx20, Ny32,36,40,44,48,80,
100 independent samples per W^2=1,2,3,4,6,9,12,16,25 plus clean references.
The only support is x=5,15; both orbitals have independent Gaussian potentials.
Fixed global half filling; canonical CPU OW parent; no reflattening.
Full entropy: all widths and all origins; average origins before realizations.
Fit windows 5..Ny/2 and 8..Ny/2. Error bars: realization SEM.

Seed root and index scheme match the inner-wall large-size campaign, allowing
paired disorder at the shared sites for Ny80 and Ny100. Other prior sizes used
a different scheme. This is a separate campaign with separate receipts.

Run run_tmux.sh. Outputs refresh after each completed size. The Ny32 gap-ratio
figure is generated immediately after its acquisition. Partial size snapshots
are explicitly marked in completion_manifest.json. Resuming validates both
per-state receipts and rolling width checkpoints. CPUs40..55, 16 workers,
one BLAS thread each; avoids the CPUs of the ongoing inner-wall large-size run.

Requested scope is recorded in requested_scope.json; Ny100 is no longer scheduled.
The directory name and immutable acquisition identity retain the original size
cap so existing checkpoints remain valid.
