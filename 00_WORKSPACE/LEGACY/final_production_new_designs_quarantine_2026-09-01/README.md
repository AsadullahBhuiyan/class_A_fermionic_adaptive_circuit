# Final production new designs

This is the independent parent package for every redesigned Colab campaign. It has no
code or notebook dependency on the preserved legacy production campaign.

Implemented bundles:

- `01_p1_chern_dynamics`
- `02_wall_cft_windows`
- `03_h1_modular_response`
- `04_maxmix_operator_cft`
- `05_pure_tangent_stability`
- `07_log_gram_alpha_scan`
- `08_h1_endpoint_packet`

The parent-level `colab_bundle_runner.py`, `bundle_layout.py`, `pilot_plan.json`, and
`_shared_src/production_runtime.py` support the queue-driven P1, wall-CFT, H1 response,
and H1-v2 endpoint-packet notebooks. G4, G5, and log-Gram use their bundle-local runners. Every notebook resolves
its bundle from `MyDrive/final_production_new_designs`.

A Drive upload may include only the parent support files and the bundle folders that will
actually run. Sibling bundles are not required. In the repository, all seven bundles remain
registered in `bundle_layout.py` and `bundle_index.json`.

Canonical source snapshots are maintained by
`_maintenance/sync_bundle_sources.py`. Regenerate P1, wall-CFT, both H1 versions, G4, and G5 notebooks
with `_maintenance/build_colab_notebooks.py`; the bespoke log-Gram notebook is maintained
inside its bundle. Production GPU code must remain byte-identical to
`src/fgtn/classA_U1FGTN_gpu.py`.
