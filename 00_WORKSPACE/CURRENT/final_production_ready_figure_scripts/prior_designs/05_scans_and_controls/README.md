# 05_scans_and_controls

Two-construction pure S2 entanglement-transition scan, max-mix S2 arm, and M2–M3 controls.

Every explicit-interface S2 `(Ny, alpha_in)` point has a pure and max-mix arm, and every point also has a pure support-terminated mirror. The pure constructions record identical entropy, topology, correlator, and tangent products; the max-mix arm records purification and the physical tangent spectrum. Dense Choi covariance tracking is disabled throughout, and distinct Born ensembles are never pooled. After all pure cases complete, the CPU analysis cell verifies/merges at least 10 trajectories, performs the locked log-chord and area-law fits, bootstraps whole trajectories, and produces the publication figure.

The notebook defaults to production in checksum-verification/report-only mode. Inspect the resolved queue first, then set `RESUME_REPORT_ONLY=False` to compute only missing shards. Pilot profiles remain available in the runner but are not the default operational path.

Open `run_production_bundle.ipynb` in an A100 Colab runtime. One invocation runs one complete shard and atomically archives it to `MyDrive/classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved`. Stochastic production cases have two fixed shards of five trajectories. Do not change sample count, duration, sequence, dtype, or physical protocol inside the notebook; select cases through the generated resumable queue. By default it queues every currently listed case and derives the valid shard count for each case; verified existing archives are skipped safely after an interruption. Before every new shard, the storage guard reserves 1 GB of headroom under the 12 GB active-output budget and refuses to launch when outputs must be offloaded.

All active cases resolve `Nx=20` directly; no accepted-width file is consulted.

`production_config.json` is immutable run intent. `src/source_manifest.json` records the canonical engine and helper hashes. Run `_maintenance/sync_bundle_sources.py` from the repository root whenever canonical source changes; never hand-edit the copied engine.
Production preflight also requires an accepted launch-decision JSON; copy `gate_decisions_template.json` to the output path shown in the notebook and set a decision true only after its cited analysis passes.
