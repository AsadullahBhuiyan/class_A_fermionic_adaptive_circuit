# Automated Colab runbook

## Safety contract

- Active output: `classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved`.
- Ordinary cases: two five-trajectory shards, indexed `0,1`.
- Scaling geometry: `Nx=20`, `Ny=20,30,40,50,60`.
- Existing output trees are read-only compatibility sources; incompatible archives are preserved and explained.
- Frozen P1 is finished only by its dedicated notebook.

## Launch procedure

1. Choose an A100 runtime and upload the whole package to MyDrive.
2. Open the desired bundle's `run_production_bundle.ipynb`.
3. Run report-only mode.  Confirm the queue, identities, and first pending item.
4. Set `RESUME_REPORT_ONLY=False` and rerun.  Completed shards receive no preflight or child launch.
5. Leave the cell running until the final receipt verification and automatic disconnect.

The runner provides live `tqdm` bars, receipt-scan progress, per-shard output, 60-second
heartbeats, elapsed time, and rolling ETA.  A failure record contains the exact stage,
case, shard, command, paths, exit code, and the last 200 child lines.

## Dependencies

The fixed baseline is informative and non-gating.  Bundle 02 no longer waits for a width
decision.  H3 needs only its two `20x40` parent shard-zero archives.  P2 and bundle 05 still
honor their scientific decision files.  M3 wall cases appear only after the M3 bulk gate
emits a three-to-five-value noise bracket.  B1 is independent.
