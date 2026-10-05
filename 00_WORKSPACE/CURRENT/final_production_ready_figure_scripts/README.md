# Preserved prior-design production campaign

This parent now contains only the earlier production campaign and its supporting runtime.
The authoritative bundles live under `prior_designs/`; their names, output collections,
resume contracts, and scientific archives are unchanged.

All redesigned standalone campaigns moved to the independent sibling
`../final_production_new_designs/`. Nothing in that package imports this parent or
requires its prior-design files.

The active preserved contract is
`production_10sample_v4_occupied_frame_cycle_resolved`. Its output root remains
`MyDrive/classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved`.
Compatible older shards are reused only after checksum, engine, audit, case, seed, shard,
and sample-index verification.

Use `bundle_layout.py` and `bundle_index.json` as the prior-design registry. Edit
canonical shared sources under `_shared_src/`, then run
`_maintenance/sync_bundle_sources.py` and
`_maintenance/build_colab_notebooks.py`. The frozen
`prior_designs/01_p1_existing_completion` notebook remains byte-preserved.
