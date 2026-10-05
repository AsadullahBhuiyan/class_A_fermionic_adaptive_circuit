# Root reorganization manifest

Move date: 2026-08-17. These were same-filesystem moves; no scientific or provenance
content was deleted. Current/recent project directories remain separate inside the
physical `00_WORKSPACE/CURRENT/` category.

| Former root content | Canonical destination |
|---|---|
| `README.md`, `START_HERE.md` | initially `00_START_HERE/`, now `00_WORKSPACE/START_HERE/` |
| Repository policy, cleanup, storage, audit, and Git-migration Markdown files | `PROJECT_ADMIN/` |
| Standalone theory/method PDFs | `NOTES/20_THEORY_AND_METHODS/standalone/` |
| Loose external papers and extracted paper text | `NOTES/40_EXTERNAL_PAPERS/sources/` |
| Loose logs, scratch notes, protocol text, and paper-style draft | `NOTES/90_LEGACY/root_imports/` |
| `CI_Lindblad_DW.py` | `scripts/legacy/CI_Lindblad_DW.py` |
| Loose PNG/GIF outputs | `figs/legacy_root_outputs/` |

## Physical project-tree phase

The former `00_WORKSPACE/` and `CURRENT/` navigation layers contained only generated
links and small navigation files. At the user's direction they were replaced by physical
project categories. The exact 34-directory mapping is in
[`PHYSICAL_REORGANIZATION_PLAN.md`](PHYSICAL_REORGANIZATION_PLAN.md).

| Former root project class | Canonical destination |
|---|---|
| Eight active/recent projects | `00_WORKSPACE/CURRENT/` |
| Seven complete Colab packages | `00_WORKSPACE/COLAB/` |
| Four standalone large-result projects | `00_WORKSPACE/LARGE_RESULTS/` |
| Twelve older method/writing projects | `00_WORKSPACE/LEGACY/` |
| OSG plus two reference implementations | `00_WORKSPACE/EXTERNAL/` |

The only regular files intentionally retained at repository root are `.gitignore`,
`.gitattributes`, `AGENTS.md`, and `pytest.ini`. Git requires the first two at the
worktree root, the agent policy loader requires the third there, and pytest configuration
must remain at the discovery root. The remaining visible root directories
are the conventional shared core (`src`, `scripts`, `tests`, `notebooks`, `cache`,
`figs`), the physical workspace, notes, administration, and checksum inventories.
