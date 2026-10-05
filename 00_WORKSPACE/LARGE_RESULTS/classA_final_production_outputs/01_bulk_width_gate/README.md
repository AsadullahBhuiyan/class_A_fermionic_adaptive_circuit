# 01_bulk_width_gate

P1 bulk baseline and W1 transverse-width gate.

The notebook defaults to `RUN_PROFILE='pilot_calibration'`: the exact full-duration physics for a curated case list, but only shard zero (five trajectories), written to the separate `classA_pilot_outputs` tree. `pilot_science` expands the curated list and may select additional case-specific shards declared in `pilot_plan.json`. Neither pilot is a gate and neither is automatically pooled with production. Set `RUN_PROFILE='production'` explicitly only after reviewing the synchronized A100 timings and peak-memory fields saved by calibration.

Open `run_production_bundle.ipynb` in an A100 Colab runtime. One invocation runs one complete shard and atomically archives it to `MyDrive/classA_final_production_outputs`. Stochastic production cases have five fixed shards of five trajectories. Do not change sample count, duration, sequence, dtype, or physical protocol inside the notebook; select cases through the generated resumable queue. By default it queues every currently listed case and derives the valid shard count for each case; verified existing archives are skipped safely after an interruption. Before every new shard, the storage guard reserves 1 GB of headroom under the 12 GB active-output budget and refuses to launch when outputs must be offloaded.

`production_config.json` is immutable run intent. `src/source_manifest.json` records the canonical engine and helper hashes. Run `_maintenance/sync_bundle_sources.py` from the repository root whenever canonical source changes; never hand-edit the copied engine.
