# Large-size extension of equilibrium-disorder entropy fits

Session: disorder-c-large-sizes
Attach: tmux attach -t disorder-c-large-sizes
Progress: run.log. Final status: exit_code.txt (0 = success).
Resume: bash run_tmux.sh. The tmux pane remains visible after exit.

New Ny=80,100,120,160; fixed Nx20; 100 independent realizations per
variance W^2=1,2,3,4,6,9,12,16,25 plus a clean reference per size.
Same canonical CPU hard-wall parent and iid orbital-resolved Gaussian
diagonal disorder only at x=5,6,14,15. Fixed global half filling.
Full entropy, all widths Ay1..Ny/2, all Ny periodic cut origins.
Reuse verified Ny32,36,40,44,48 outputs in the final comparison.

32 workers, CPU range 8..39, one BLAS thread each.
Per-realization NPZ and checksum receipts. Additionally save one rolling
entropy-profile checkpoint per active realization after each completed
width, bound to the complete configuration/source identity. The Hamiltonian
and projector are reconstructed deterministically from saved seed components
on resume; completed entropy rows are reused. No evolving stochastic state.
The exterior block is exactly translation invariant: verify this at matrix
level before computing origin zero and repeating its entropy for all origins.
The disordered active block always evaluates every origin explicitly.

The script executes clean and W^2=25 sample-zero full-matrix checks at
each new size, including a wrapping cut. Validate physical spectra,
Hermiticity, projector purity, trace and complement entropy identities.
No new circuit dynamics or GPU computation.

The pipeline automatically executes the notebook, writes CSV fit tables,
and exports vector PDF / 300-dpi PNG figures. The requested c_eff versus W^2
figure includes all nine sizes and three fitting windows:
5..Ny/2; ceil(Ny/4)..Ny/2; 8..Ny/2.
Other saved sensitivity windows: 2..Ny/2 and ceil(5Ny/32)..Ny/2.
Error bars are realization SEM; bootstrap whole profiles, not individual cuts.
No infinite-size extrapolation law is imposed.
