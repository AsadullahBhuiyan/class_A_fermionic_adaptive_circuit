# Workspace cleanup completion audit

Audit date: 2026-08-17.  This document distinguishes completed evidence from actions
that still require user approval or a writable external filesystem.

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Current production work is easy to find | Passed | Thirty-five canonical project directories are physically grouped under five obvious `00_WORKSPACE/` categories, including the separate sample-free mean-channel/Lindblad CPU campaign. `CURRENT/` retains each active/recent project separately; `COLAB/` retains each complete package with its data. Only the conventional shared core, `NOTES/`, `PROJECT_ADMIN/`, hidden configuration, and required controls remain at repository root. `NOTES/` indexes the documentation library independently. |
| Erroneous large waste is removed | Passed | `erroneous_gpu_stuff/` was removed after authorization; 1,267 approved Git temporary objects (733.953 GiB) were removed; Git now reports zero garbage. |
| Remaining regenerable/ambiguous debris is handled safely | Passed | Candidates B and C were explicitly approved, revalidated, and removed. Zero approved paths remain; `.tmp/ref_papers/` and the unrelated smoke notebook were preserved. |
| Large result data is cataloged without misclassifying Colab work | Passed | Five standalone archive candidates cover 1,535 files and 837,778,719,908 bytes. Four additional manifests are retained only as integrity inventories for active Colab-package data: 2,728 files and 12,360,081,037 bytes. All nine inventories pass path/size/nanosecond-mtime verification after the physical move. Colab data stays under `00_WORKSPACE/COLAB/` beside its notebooks and runners. |
| Optional large-result offload is safe | Canceled by user; sources intact | `/data/abhuiyan` is writable on a separate filesystem. The original nine-dataset attempt was interrupted before any receipt completed after the user rejected archival classification of Colab outputs. The revised runner excludes all Colab data. The incomplete 377.3 GB destination copy is preserved pending explicit direction; no source deletion is authorized. |
| Git storage is repaired without losing user work | Passed for repair; promotion pending | Legacy `HEAD` resolves, connectivity passes, seven valid packs remain, and garbage is zero. The physical-layout successor is committed at `fb739bb`; its Git object store is about 154 MiB with zero garbage, and legacy pushes are disabled. |
| Clean successor matches current source work | Passed | The 34-project physical layout was mirrored with Git-aware moves, current text and generated evidence were synchronized, and the successor is clean at commit `fb739bb`. Its source-only notes view correctly omits uncopied bulk data and temporary references. |
| Clean successor runs maintained tests | Passed | Root pytest discovery is constrained to the maintained `tests/` directory. Both the canonical tree and successor pass 100 tests with 8 skips. |
| Canonical GPU copies are synchronized | Passed | All 16 `classA_U1FGTN_gpu.py` copies match canonical SHA-256 `19b200dce7b5fd004a627770492d32c5738ef19ff8f94066eebad5ad66d9f23d`. |
| Deletion approval rule is honored | Passed | No deletion after the rule was introduced occurred without explicit, scoped approval. Approved Candidates A, B, and C are recorded in `DELETION_CANDIDATES.md`. |

## Remaining terminal decisions

1. Decide whether to retain or explicitly delete the canceled, incomplete destination
   copy under `/data/abhuiyan`; this decision does not affect the source data.
2. Decide whether the five standalone large-result trees should ever be archived. Colab
   experiment data is excluded from that decision.
3. Choose a new remote or explicitly retain the committed v2 successor locally, then
   separately approve any directory-name switch or removal of the legacy repository.

The physical cleanup, navigation, and Git-repair work are complete. The decisions above
are deliberately separate destructive, archival, or publication actions and remain
unexecuted until explicitly authorized.
