# G5: pure-state tangent stability

This standalone, versioned bundle tests the stability of independent half-filled
pure Gaussian Born trajectories with the analytic tangent cocycle. It does not
reuse G4 words, seeds, or states. It does not evolve explicit `+epsilon` and
`-epsilon` states and stores no replayable trajectory record.

The first `Ny` cycles align the occupied/empty tangent blocks and are excluded
from rate accumulation. The final `Ny` cycles determine all physical
occupied-empty covariance pair rates before the 16 slowest physical modes are
selected. Production is 25 trajectories per case in five immutable shards at
`Nx=20`, `Ny=20,30,40,50,60`, and `T=2*Ny`, for wall and matched-trivial arms.

This experiment makes no numerical equality requirement between the pure-state
tangent spectrum and G4's state-dependent trajectory-operator spectrum.

Open `run_production_bundle.ipynb` in an A100 Colab runtime. The notebook performs an enforced G5 A100 runtime check before its opt-in pilot or production queue; `RUN_QUEUE=False` is the safe default. Existing immutable shards are identity- and checksum-verified before being reused.
