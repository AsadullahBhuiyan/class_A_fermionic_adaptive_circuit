# Physical workspace reorganization

Plan date: 2026-08-17.

This is the authoritative move map for replacing the former symlink-only workspace view
with a real high-level directory tree. A move changes only the canonical path; it does not
delete project content. Current and recent projects remain separate directories. Each
Colab package moves intact with its notebooks, source copies, and generated data.

## Repository root retained

The conventional executable core remains at repository root to avoid needless import and
cache-path breakage:

- `src/`, `scripts/`, `tests/`, `notebooks/`
- `cache/`, `figs/`
- `NOTES/` and `PROJECT_ADMIN/` (including `PROJECT_ADMIN/archive_manifests/`)
- `00_WORKSPACE/`
- required control/configuration paths, including `AGENTS.md`, `.gitignore`, and
  `.gitattributes`

## Physical destinations

### `00_WORKSPACE/CURRENT/`

- `final_production_ready_figure_scripts/`
- `experiment_review/`
- `prxq_draft/`
- `validation_campaigns/`
- `topological_frustration_diagnostics/`
- `tangent_cocycle_flux_snapshot/`
- `tangent_edge_channel/`
- `cpu_cft_extraction/`

### `00_WORKSPACE/COLAB/`

- `colab_charge_fluctuations/`
- `colab_large_entanglement_scaling_N20/`
- `colab_lyapunov/`
- `colab_no_feedback_alpha_sweep_transfer/`
- `colab_partial_post-select/`
- `colab_regularized_choi_transfer_matrix/`
- `colab_small_system_testing/`

### `00_WORKSPACE/LARGE_RESULTS/`

- `choi_covariance_cpu/`
- `dw_convergence/`
- `experiments/`
- `lyapunov_analysis_v2/`

### `00_WORKSPACE/LEGACY/`

- `exact_DW_benchmark_notes/`
- `form_factor_analysis/`
- `markov_transfer_operators/`
- `monitored_fermion_reference_sheet/`
- `perturbative_expansion/`
- `prl_draft/`
- `projector_form_factor_product/`
- `repo_synthesis/`
- `sample_average_testing/`
- `summary_of_results/`
- `tangent_edge_channel_note/`
- `topological_dynamics_introduction/`

### `00_WORKSPACE/EXTERNAL/`

- `Haining_code/`
- `Haoyu_code/`
- `OSG/`

### Other physical moves

- `00_START_HERE/` becomes `00_WORKSPACE/START_HERE/`.
- The old `CURRENT/` and the previous contents of `00_WORKSPACE/` are generated
  navigation artifacts, not canonical project data. They are replaced by the physical
  tree under the user's explicit direction to move projects into the designed layout.

## Compatibility and verification

Each category receives only the internal compatibility links needed by existing runners,
such as `src -> ../../src`. No old project-name link remains at repository root. After
each move batch:

1. update source-manifest roots and hard-coded launch paths;
2. rebuild the notes index and the physical-workspace recency index;
3. verify all links resolve;
4. run focused package tests and the maintained root suite;
5. compare every GPU engine copy to `src/fgtn/classA_U1FGTN_gpu.py`;
6. mirror the final paths in the clean successor with Git-aware moves.

No numerical source data or interrupted destination copy is deleted by this plan.
