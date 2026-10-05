# Three-session Colab lanes

Run `00_validation/run_production_bundle.ipynb` alone first. After it passes, open these
three notebooks in the three available A100 sessions and use **Runtime → Run all**:

1. `lane_1_baselines.ipynb`: bundle `01`, followed by bundle `06`.
2. `lane_2_parents_flux.ipynb`: bundle `02`, followed by dependent H3 bundle `03`.
3. `lane_3_maxmix_scans.ipynb`: bundle `04`, followed by bundle `05`.

All lanes default to `RUN_PROFILE='production'`. Production remains dependency-gated and
will refuse to start when its accepted-width or decision files are absent. The two pilot
profiles remain available only when selected explicitly.

From a clean production Drive, start Lane 1 first. After all bundle-01 archives exist, the
lane automatically runs the joint W1/B0 analysis and writes `accepted_width.json` before
continuing to required bundle 06. Start Lane 2 only after that file appears. Lane 3 must
wait for the evidence-backed S1/T1 launch decisions produced after reviewing bundle 02.
After Lane 3 completes the full M3 bulk matrix, it automatically writes
`m3_bulk_gate.json`; rerun Lane 3 once so verified bulk archives are skipped and the
resulting wall-noise bracket is executed.

H3 is the deliberate production exception to the generic two-shard rule: it runs one
preregistered record from parent shard `0` for each of the two agreed 33-point
interface/control cases. It is a representative branch diagnostic rather than an
ensemble-frequency estimate and writes under the final-production output tree.

Before any preflight or storage check, the lane runner displays `tqdm` bars while it
checksum-verifies current and compatible legacy archives.  It then prints a per-bundle
resume table and initializes the persistent lane queue bar at the verified-complete count,
so the first child launch is the earliest genuinely missing queue item.  Set
`RESUME_REPORT_ONLY=True` for this verification report without launching numerical work.
Long child commands stream their output live and emit a 60-second liveness heartbeat.

The lane runner checks storage before each new shard, uses the numbered bundles without
changing their physics, and writes schema-v2 progress records plus persistent logs to
`classA_pilot_outputs/production_10sample_v2/_lane_sessions/` or
`classA_final_production_outputs/production_10sample_v2/_lane_sessions/`. It safely skips
complete current archives and strictly compatible legacy shards that the numbered notebooks
already produced. Legacy reuse is recorded separately with archive paths and hashes; it is
not mislabeled as a current-revision result. Never run a lane and a numbered notebook on the
same bundle/case/shard concurrently.
