# 16×16 outcome-trajectory independence validation

This campaign holds the random-pure initial state and the complete random site
schedule fixed while varying only the Born-outcome RNG across ten trajectories.
It tests two distinct statements:

1. whether all trajectories end in the same occupied subspace;
2. whether distinct final states nevertheless all converge to the uniform
   finite-size Chern-insulator target.

The evolution kernel uses `classA_U1FGTN.run_markov_circuit(...)` with the
physical occupied-frame backend and `site_schedule_replay`. Physical projectors
are reconstructed only during post-run analysis. One covariance replay of
trajectory zero audits the new schedule-only path.

## Launch

```bash
00_WORKSPACE/CURRENT/trajectory_independence_validation/launch_tmux.sh --dry-run
00_WORKSPACE/CURRENT/trajectory_independence_validation/launch_tmux.sh --preflight-only
00_WORKSPACE/CURRENT/trajectory_independence_validation/launch_tmux.sh
```

Resume a stopped campaign at verified sample-shard granularity:

```bash
00_WORKSPACE/CURRENT/trajectory_independence_validation/launch_tmux.sh --resume CAMPAIGN_ID
```

The launcher selects ten idle physical cores from one NUMA node. Override this
with `CPU_LIST=0,1,...,9`; set `ALLOW_BUSY=1` only when intentional.

## Statistical unit

One complete Born-outcome trajectory, conditional on the common initial state
and schedule, is the independent unit. The 45 pairwise state distances are
dependent comparisons and are never counted as 45 samples. Confidence bands
resample the ten complete trajectories 10,000 times.

Physical classification does not control process success. A campaign can
validly conclude exact state independence, topology-only independence, or
trajectory dependence. Process failure is reserved for incomplete/corrupt
products, shared-input disagreement, nonfinite values, excessive Gram drift,
or covariance-audit failure.
