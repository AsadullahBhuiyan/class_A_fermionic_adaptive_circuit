# Clean Git migration plan

Status date: 2026-08-17.  This plan is deliberately reversible.  It does not replace the
current repository, rewrite a remote branch, or authorize deletion.

## Why a successor is preferable to in-place history surgery

The working tree contains current source and documentation.  Before cleanup, `.git` was
approximately 1.2 TB; an explicitly approved removal of 733.953 GiB of orphaned temporary
objects reduced it to 399.815 GiB with zero Git garbage.  Seven valid historical packs
still occupy 383.410 GiB.  Rewriting that valid history in place would combine three
risks: losing uncommitted user work, silently changing old provenance, and requiring
substantial temporary space.  A small source-only successor can instead be verified
beside the original before any path is switched.

## Verified candidate

`scripts/create_clean_repo_snapshot.py` created:

```text
/home/abhuiyan/class_A_fermionic_adaptive_circuit_clean_candidate_20260817_v2
```

The candidate was selected from files currently present in the working tree that are
either tracked or untracked-and-not-ignored.  Ignored raw numerical data and the original
`.git` directory were not copied.  Nested `Haining_code` source was exported without its
nested `.git` metadata.  Existing paths in a resumed partial copy were accepted only
after byte-for-byte SHA-256 verification.

Its `PROJECT_ADMIN/CLEAN_SNAPSHOT_MANIFEST.json` records 1,062 files/symlinks and
201,680,152 regular-file bytes. Fourteen former tracked paths are absent: approved debris,
prior user-side deletions, and root paths whose contents now exist at their reorganized
destinations. The script did not restore or reinterpret those old paths.

The internal verification matched all 1,062 entries in both source and destination. The
manifest itself has SHA-256
`7d9bf387953b1639a63af625d09f19e3fdaf6ef6dfa012b6b87eb457abca937c`.
A new `main` repository is initialized in the candidate with 1,032 source/provenance paths
committed at `3a6b4ab164bb202415363ebcb5ce7cc037576f73`. Its object store is
approximately 150 MiB. The
legacy GitHub URL is
configured as fetch-only `legacy-origin`; its push URL is deliberately disabled.
The verification/status baseline commit is
`9481e2dabc906276ebcdae687c3bf5bf68f035d5`. The physical-layout synchronization is
committed at `fb739bb`.

The maintained suite passes in the candidate: 100 passed and 8 skipped. Root pytest
discovery is explicitly constrained to `tests/`, so compatibility links cannot collect
legacy runnable scripts as tests.
All 16 `classA_U1FGTN_gpu.py` files in the legacy working tree, including all production
bundle copies, match the canonical source SHA-256
`19b200dce7b5fd004a627770492d32c5738ef19ff8f94066eebad5ad66d9f23d`.

The earlier pre-reorganization candidate remains beside v2 as a rollback snapshot. It is
not the promotion target.

The user-requested physical project move into `00_WORKSPACE/` is now mirrored in the
candidate with Git-aware moves. It validates 34 projects across five categories, has no
broken links, compiles the 56-page campaign note and 29-page evidence atlas, and is clean
at `fb739bb`. The candidate is eligible for promotion, but promotion itself remains an
explicit user decision.

## Migration gates

1. Choose a new remote or explicitly approve replacement of the existing remote history.
2. Keep the original working tree and all experiment data untouched. Any optional archive
   decision is separate from clean-repository promotion.
3. Switch directory names or remove the old `.git` only after separate explicit approval.

## What this migration does not solve yet

- Active Colab data remains deliberately outside the archive runner. The optional five-tree
  large-result archive was canceled and is not a migration prerequisite.
- The valid 383.410 GiB Git history remains the only local copy of that historical object
  store and is not a deletion candidate yet.
- The candidate is a source snapshot, not a history-preserving filtered clone.  The old
  commit ID and manifests provide provenance, while the old repository remains retained.
