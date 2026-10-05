# Physical workspace layout

The repository now has a genuinely simple physical tree. The project directories under
`00_WORKSPACE/` are canonical directories, not a layer of aliases.

```text
class_A_fermionic_adaptive_circuit/
├── 00_WORKSPACE/
│   ├── CURRENT/          production, manuscript, experiment review, recent evidence
│   ├── COLAB/            complete Colab packages, including their generated data
│   ├── LARGE_RESULTS/    standalone large-result projects
│   ├── LEGACY/           older methods, drafts, and analysis projects
│   ├── EXTERNAL/         OSG and external/reference implementations
│   └── START_HERE/       repository entry guide
├── NOTES/                unified documentation library
├── PROJECT_ADMIN/        policy, cleanup, storage, migration, provenance, manifests
├── src/                  canonical CPU/GPU dynamics source
├── scripts/              shared runnable and maintenance scripts
├── tests/                maintained test suite
├── notebooks/            shared exploratory/analysis notebooks
├── cache/                shared historical covariance cache
└── figs/                 shared historical/generated figures
```

`CURRENT/` keeps every recent project separate: production bundles, experiment review,
the sample-free mean-channel/Lindblad CPU campaign, the current PRXQ draft, validation
campaigns, topology/frustration diagnostics, tangent campaigns, and CPU CFT extraction
each retain their own directory.

`COLAB/` likewise keeps every package separate and intact. Its `gpu_data/`, `cpu_data/`,
and `analysis_outputs/` subdirectories remain with the notebook and source package that
produced them. They are active experiment data, not archive candidates.

The conventional shared core remains at repository root to preserve standard imports and
entrypoints. Each workspace category contains small internal links to that core (`src`,
`scripts`, `tests`, `notebooks`, `cache`, `figs`, and `.tmp`) so older package-relative
runners continue to resolve without reintroducing project clutter at repository root.

[`00_WORKSPACE/RECENCY_INDEX.md`](../00_WORKSPACE/RECENCY_INDEX.md) sorts the 35 physical
project directories by newest contained file. The exact physical move map is recorded in
[`PHYSICAL_REORGANIZATION_PLAN.md`](PHYSICAL_REORGANIZATION_PLAN.md).
