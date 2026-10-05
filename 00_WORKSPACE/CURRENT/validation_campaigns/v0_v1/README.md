# V0/V1 CPU Validation

This campaign implements the CPU portion of V0 and the staged V1 order-artifact
check. Every trajectory calls `classA_U1FGTN.run_markov_circuit`. GPU code is
only inspected and is always reported as `AUDIT_ONLY / NOT_VALIDATED`.

Launch the full gated run in a detached tmux session:

```bash
validation_campaigns/v0_v1/launch_tmux.sh
```

Launch the reduced V1-only campaign using a previously passing V0 summary:

```bash
validation_campaigns/v0_v1/launch_tmux.sh \
  --mode v1 --v1-nx 16 --v1-ny 16 --v1-samples 10 \
  --no-v1-auto-escalation \
  --v0-summary validation_campaigns/results/<prior-run>/v0/validation_summary.json
```

The launcher runs the CPU test suite and a smoke campaign, then confines
production to CPUs 56–111 with 28 one-thread workers. V1-only launches also
time one representative trajectory before production and save a projected
initial-stage runtime. Per-trajectory checkpoints, verified shard hashes, and
deterministic merged arrays make relaunching with `--resume` safe. Use
`--wait-for-session NAME` when an explicit tmux prerequisite is required.

For a short infrastructure check:

```bash
python validation_campaigns/v0_v1/run.py all --smoke \
  --output-dir /tmp/v0_v1_smoke --workers 2 --bootstrap-samples 100 --resume
```

The scientific status is written to `v0/validation_summary.json` and
`v1/validation_summary.json`. A successful V0 is labeled
`CPU_PASS / GPU_NOT_RUN`; V1 chirality remains conditional until H2 supplies
the Born score-response term.
