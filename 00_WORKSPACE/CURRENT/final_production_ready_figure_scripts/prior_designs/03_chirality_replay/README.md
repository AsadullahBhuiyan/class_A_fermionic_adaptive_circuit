# 03_chirality_replay

H3 frozen-record twist circles and exact replay recovery.

Calibration selects one exact record from one five-record `20x40` parent shard and runs a 33-point closed frozen-flux circle. Production runs record index `0` from parent shard `0` for each of the explicit-interface and matched-trivial cases: two representative circles total, not an ensemble-frequency estimate. Each point checkpoints elapsed time and A100 peak memory, and the manifest projects every denser-grid cost.

For an independent local check, `cpu_one_record_pilot/` uses the canonical CPU engine for one `20x40` record on a conservative nine-point closed circle. It is not part of the A100 queues.

The notebook defaults to production in checksum-verification/report-only mode. Inspect the resolved queue first, then set `RESUME_REPORT_ONLY=False` to compute only missing shards. Pilot profiles remain available in the runner but are not the default operational path.

Open `run_production_bundle.ipynb` in an A100 Colab runtime. One invocation runs one complete shard and atomically archives it to `MyDrive/classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved`. H3 has one record and one shard for each of its two 33-point protocols. Do not change sample count, duration, sequence, dtype, or physical protocol inside the notebook; select cases through the generated resumable queue. By default it queues every currently listed case and derives the valid shard count for each case; verified existing archives are skipped safely after an interruption. Before every new shard, the storage guard reserves 1 GB of headroom under the 12 GB active-output budget and refuses to launch when outputs must be offloaded.

All active cases resolve `Nx=20` directly; no accepted-width file is consulted.

`production_config.json` is immutable run intent. `src/source_manifest.json` records the canonical engine and helper hashes. Run `_maintenance/sync_bundle_sources.py` from the repository root whenever canonical source changes; never hand-edit the copied engine.
