# Physical workspace

This is the canonical high-level project tree, not a symlink index. Start in `CURRENT/`
for paper production and recent evidence. Every named project directory below is the real
directory on disk.

| Directory | Contents |
|---|---|
| `CURRENT/` | Production bundles, experiment review, current manuscript, validation, and recent campaigns |
| `COLAB/` | Complete Colab packages with notebooks, source copies, and generated CPU/GPU data kept together |
| `LARGE_RESULTS/` | Standalone legacy result projects that may be considered separately for external archiving |
| `LEGACY/` | Older methods, drafts, analyses, and reference projects retained for provenance |
| `EXTERNAL/` | OSG launch material and external/reference implementations |
| `START_HERE/` | Repository entry guide |

The conventional shared core remains at repository root: `src/`, `scripts/`, `tests/`,
`notebooks/`, `cache/`, and `figs/`. Small internal compatibility links inside each
category let older runners locate that shared core; no old project-name links remain at
repository root.

Use [`RECENCY_INDEX.md`](RECENCY_INDEX.md) to see all physical project directories sorted
by their newest contained file. Documentation is collected separately in `../NOTES/`,
and maintenance/provenance records live in `../PROJECT_ADMIN/`.
