# G4: max-mix purification and trajectory-operator CFT

This standalone, versioned bundle runs one max-mix Born ensemble and emits two
scientifically independent products from the same checkpoint eigendecomposition.
G4a contains purification, entropy, charge, and spatial-contour observables. G4b
contains the direct Gaussian spectrum of the trajectory operator and the derived
charge-resolved tower, `k_op`, and `c_eff_op` analysis.

No tangent cocycle, Choi state, covariance history, or replayable measurement
record is saved. Only cumulative conditional log probability per cycle is retained.
The wall and uniform-trivial arms have identical active support and channel count.

Production is 25 trajectories per case in five immutable shards, at `Nx=20`,
`Ny=20,30,40,50,60`, and `T=2*Ny`. The previous `04_maxmix_master` and its outputs
remain provenance artifacts and are not pooled with this revision.

Open `run_production_bundle.ipynb` in an A100 Colab runtime. The notebook performs an enforced G4 A100 runtime check before its opt-in pilot or production queue; `RUN_QUEUE=False` is the safe default. Existing immutable shards are identity- and checksum-verified before being reused.
