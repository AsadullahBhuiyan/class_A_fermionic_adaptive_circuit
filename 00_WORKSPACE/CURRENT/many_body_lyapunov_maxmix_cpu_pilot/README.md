# Nx=16 hard/soft many-body Lyapunov CPU campaign

This is a local CPU campaign, not a Colab bundle. It calls the repository's
canonical `classA_U1FGTN.run_markov_circuit` entry point.

## Scientific contract

- `Nx=16`; `Ny=20,22,24,26,28,30`
- hard/support-truncated and soft/untruncated walls at `x=[4,12]`
- 100 independent Born trajectories per construction and circumference
  (1,200 total)
- `T=4*Ny`, maximally mixed initialization, `nshell=1`
- `alpha_1=1`, `alpha_2=30`, `raster_y`, perfect correction, no postselection
- complex128 covariance dynamics

The hard transfer space is the prepared slab (`N_eff=18*Ny`); the one-time
Born probability of exterior preparation is outside the transfer problem. The
soft transfer space is the full top layer (`N_eff=32*Ny`).

The observer saves cumulative realized Born log probability at every cycle. At
stride-four cycles and all window boundaries `Ny,2Ny,3Ny,4Ny`, it saves the
full occupation spectrum, exact cap masks, entropy, charge variance, leading
64 `log(sigma_i^2)` levels, the 16 softest modes' transverse weights, and
numerical residuals. No covariance history or eigenvectors are retained.

## Run and monitor

```bash
00_WORKSPACE/CURRENT/many_body_lyapunov_maxmix_cpu_pilot/launch_tmux.sh
```

The outer `tqdm` bar tracks trajectories. One task is one independently seeded
trajectory. A rolling checkpoint every four cycles contains the complex128
covariance, all canonical RNG streams, schedule state, record weight, and
observer prefix, so an interrupted trajectory resumes exactly.

Useful overrides:

```bash
WORKERS=8 MAX_WORKERS=8 CPU_LIST=84-91 NICE_LEVEL=10 \
  00_WORKSPACE/CURRENT/many_body_lyapunov_maxmix_cpu_pilot/launch_tmux.sh
```

Inventory only:

```bash
python 00_WORKSPACE/CURRENT/many_body_lyapunov_maxmix_cpu_pilot/run_campaign.py \
  --report-only
```

Completed trajectories are verified by their NPZ/completion-JSON byte count,
SHA-256, task identity, configuration hash, and source hashes. A restart skips
only verified pairs. There is no Drive API, archive, lease, dashboard, or
migration machinery.

## Analysis intent

Fit each trajectory separately in `[Ny,2Ny]`, `[2Ny,3Ny]`, and `[3Ny,4Ny]`,
then bootstrap whole trajectories. The hard and soft constructions remain
separate. Report finite-size coefficients and `x_i/c_eff` only when temporal
and circumference-window stability gates pass; no anisotropy calibration or
absolute temporal `c_eff` claim is inferred from this campaign alone. The
existing `Nx=20` data are a transverse-width sensitivity comparison, not a
pooled ensemble.

## Machine-load note

The launcher accepts a CPU affinity and niceness level. The initial repository
launch deliberately uses a small, low-priority SMT allocation because another
56-worker production campaign currently occupies all physical cores. Increase
`WORKERS` or relaunch with a larger free `CPU_LIST` after that campaign exits;
the completed trajectories and rolling checkpoints resume automatically.
