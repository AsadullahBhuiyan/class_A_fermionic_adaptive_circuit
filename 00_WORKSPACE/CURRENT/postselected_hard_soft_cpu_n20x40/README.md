# Postselected hard/soft CPU comparison at 20 x 40

This local campaign runs two separate canonical CPU scripts:

- `run_hard_wall.py`: support-truncated hard wall, with `meas_slab_only=True`;
- `run_soft_wall.py`: untruncated soft wall, with `meas_slab_only=False`.

Both use `classA_U1FGTN.run_markov_circuit`, maximally mixed initialization,
`postselect=True`, `nshell=1`, `alpha_1=1`, `alpha_2=30`, raster-y ordering,
complex128 covariance dynamics, and `T=4 Ny=160`. Full postselection is a
deterministic forced-outcome protocol, so the engine correctly runs one
trajectory rather than a stochastic sample ensemble.

Each script records total entropy and the occupation-derived finite-time gap
`min_j abs(-atanh(a_j)/t)` at every cycle. It retains the analyzed endpoint
covariance, occupation spectrum, and eigenvectors. The hard run analyzes the
`22 Ny` active slab modes after the exterior has been forced into its fixed
occupied product sector; the soft run analyzes all `40 Ny` modes.

A rolling checkpoint is written every ten cycles. Launch both NUMA-separated
jobs with:

```bash
bash 00_WORKSPACE/CURRENT/postselected_hard_soft_cpu_n20x40/launch_tmux.sh
```

Outputs and logs are written under `outputs/`.
