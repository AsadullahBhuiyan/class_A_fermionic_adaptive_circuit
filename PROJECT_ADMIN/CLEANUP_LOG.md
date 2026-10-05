# Cleanup log

## 2026-08-17

- Permanently removed `erroneous_gpu_stuff/` after explicit user authorization; recovered
  approximately 162 GB.  It is not recoverable from the working tree.
- Identified approximately 733 GiB of Git-classified temporary pack garbage; it was kept
  pending approval and later removed in the explicitly approved operation below.
- Stopped one runaway background `git add` process that was creating another temporary
  pack.  No file was deleted by stopping the process.
- Expanded `.gitignore` to exclude raw campaign data, caches, large numerical containers,
  and regenerable build/test debris from future staging.
- Created the nonbreaking `CURRENT/` navigation layer and storage documentation.
- Confirmed that no writable external archive destination is currently mounted.
- Completed file-level SHA-256 source manifests for all nine cold datasets: 4,263 files
  and 850,138,800,945 bytes.  Sources remain in place and no external-copy receipt exists.
- Hardened the archive tool so it refuses to overwrite manifests, receipts, partial
  copies, and destination files.
- Created a verified source-only clean-repository candidate beside the original.  Its
  snapshot manifest records 996 files/symlinks and 193,555,531 regular-file bytes; no
  source file or original Git object was altered.
- Initialized and staged the candidate as a new `main` repository.  After synchronizing
  the organization/archive and live-production deltas it contains about 1,000 staged paths,
  approximately 141 MiB of
  loose objects, zero temporary Git garbage, no commit, and disabled legacy pushes.
- Verified the refreshed candidate with the maintained tests (93 passed, 8 skipped) and confirmed
  all 16 GPU engine copies match the canonical source SHA-256.
- Added a guarded all-cold-tier archive runner and runbook.  It requires a writable
  filesystem different from `/home`, verifies destination data twice, and never deletes
  source files.
- After explicit user approval, permanently removed exactly 1,267 Git-classified
  `tmp_pack_*`/`tmp_obj_*` files totaling 788,075,533,657 bytes (733.953 GiB).  Git now
  reports zero garbage; connectivity passed, seven valid packs remain, and `/home` has
  approximately 1.5 TB available.  The deleted temporary objects are not recoverable.
- Completed a requirement-by-requirement audit.  All cold manifests remain current;
  shared live/candidate paths have zero content mismatches; the unresolved gates are the
  separately approval-gated small debris, external archive access, and candidate commit.
- Created the local `00_WORKSPACE/` front door with 119 verified canonical links arranged
  by importance across active, recent-evidence, production-support, legacy, data,
  reference, maintenance, and pending-cleanup tiers.  Generated `RECENCY_INDEX.md` sorts
  the same targets by newest contained file with size/file-count context.  No canonical
  path was moved or deleted.
- Created the separate `NOTES/` documentation library: 202 canonical documents in a
  complete by-source mirror plus curated current, manuscript, theory/method, project,
  external-paper, and legacy sections.  Generated build/figure debris is excluded and no
  canonical document was moved.
- After explicit approval, removed Candidates B and C: approximately 1.49 MB of
  regenerable Python/pytest/Matplotlib caches, six zero-byte placeholder files across the
  legacy and clean-candidate trees, and the confirmed-empty `.agents/`, `logs/`, and
  `gpu_data/` directory trees. The unrelated `.tmp/ref_papers/` PDF and smoke notebook
  remain intact.
- Physically reorganized the legacy repository root. Standalone papers/notes now live in
  `NOTES/`, maintenance records in `PROJECT_ADMIN/`, the root Lindblad helper in
  `scripts/legacy/`, and loose generated media in `figs/legacy_root_outputs/`. Active and
  recent project directories remain separate at top level. Only `.gitignore`,
  `.gitattributes`, and the required `AGENTS.md` remain as root regular files.
- Rebuilt the generated views after the physical moves: `00_WORKSPACE/` now has 110
  resolving links, and `NOTES/` indexes 206 physical/linked canonical documents with zero
  broken links.
- Hardened cold-tier offload for a long-running transfer: completed immutable receipts
  can be validated and reused on restart, manifest paths are traversal-checked, and final
  verification rejects missing, changed, symlinked, or unexpected destination entries.
  Six focused archive tests pass and all nine existing manifests satisfy the stricter
  schema checks. Capacity preflight rejects undersized filesystems before creating an
  archive, archive subdirectories cannot traverse symlinks outside the destination, and
  post-manifest source inventory/size/mtime drift aborts before copying or receipt reuse.
- Created the reorganized v2 clean successor at
  `/home/abhuiyan/class_A_fermionic_adaptive_circuit_clean_candidate_20260817_v2`.
  Initial verification matched all 1,062 manifest entries in source and destination;
  100 maintained tests pass and
  8 skip; all 16 GPU engines match canonical; no raw-data path is staged. The first local
  clean commit is `3a6b4ab164bb202415363ebcb5ce7cc037576f73`, with legacy pushes disabled.
  Verification and migration status have baseline commit
  `9481e2dabc906276ebcdae687c3bf5bf68f035d5`.
- Began an external copy after `/data/abhuiyan` became writable, then stopped it immediately
  on the user's instruction when the user clarified that Colab-generated results are
  active, good experiment data. The interruption changed no sources and completed no
  verification receipt. It left approximately 377.3 GB at the destination (42 finalized
  copies and one partial file), which is preserved pending explicit deletion direction.
- Corrected the classification: `colab_small_system_testing/` and
  `colab_charge_fluctuations/` remain complete active packages with their notebooks,
  runners, CPU/GPU outputs, and analyses together. Their four existing manifests are
  integrity inventories only. The revised optional archive runner covers five standalone
  large-result trees (1,535 files; 837,778,719,908 bytes) and excludes all Colab data.
- Replaced the symlink-only workspace index with a physical high-level tree at the user's
  direction. Thirty-four canonical project directories moved into `CURRENT/`, `COLAB/`,
  `LARGE_RESULTS/`, `LEGACY/`, and `EXTERNAL/` under `00_WORKSPACE/`. Every Colab package
  moved intact with its generated data. The conventional shared core remains at root,
  with internal category compatibility links for older package-relative runners.
- Rebuilt `NOTES/` against the new physical paths (208 indexed documents, 238 links, zero
  broken) and replaced the workspace builder with a non-destructive physical-tree
  validator/recency indexer (34 projects, five categories, zero broken links). All nine
  data inventories remained fresh after their recorded source roots were updated, and the
  maintained suite passed with 100 tests and 8 skips after path-bootstrap updates.
- Added the quick file-tree map to `PROJECT_ADMIN/REPO_POLICY.md`, nested checksum
  inventories under `PROJECT_ADMIN/archive_manifests/`, and committed the same physical
  layout in the clean successor at `fb739bb`. The successor validates 34 projects, indexes
  190 source-available documents, passes 100 tests with 8 skips, has 16/16 matching GPU
  engines, compiles both controlling PDFs, and contains about 154 MiB of Git objects with
  zero garbage. The canonical notes view now indexes 209 documents and 46 curated links.
