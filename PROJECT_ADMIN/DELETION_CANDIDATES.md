# Deletion record

Inventory date: 2026-08-17.  Nothing listed here is authorized merely by appearing in
this file.  Recheck paths, sizes, and active processes immediately before any deletion.

## Completed A: orphaned Git temporary objects

Exact scope:

```text
.git/objects/pack/tmp_pack_*
.git/objects/??/tmp_obj_*
```

The user explicitly approved this exact scope on 2026-08-17.  After confirming there
were no Git writers or locks and that the inventory still matched, 1,267 files and
788,075,533,657 bytes (733.953 GiB) were permanently removed.  Zero matching temporary
objects remain.  Git connectivity passed; all seven valid packs and working-tree files
remain.  This completed deletion is not recoverable.

## Completed B: obvious regenerable local caches

Approved and removed scope:

```text
__pycache__/
.pytest_cache/
.tmp/mplconfig/
.tmp/matplotlib/
```

In the legacy workspace these contained 26 files and 1,208,082 apparent bytes. The clean
candidate also contains one ignored `__pycache__` file (278,453 apparent bytes) copied
for preservation but not staged. The user approved Candidate B on 2026-08-17; the four
legacy cache directories and the candidate's `__pycache__/` were removed, recovering
approximately 1.49 MB. Zero approved cache paths remain.

Do not delete all of `.tmp/`: it also contains `.tmp/ref_papers/lu2206.13527.pdf` and
`.tmp/analyze_charge_sharpening_smoke.ipynb`, which are not assumed disposable.

## Completed C: suspicious empty root artifacts

The zero-byte files `.codex`, `0`, and `Born_bott:` and the empty directory trees
`logs/`, `gpu_data/` (six nested directories, no files), and `.agents/` were confirmed
empty/valueless and explicitly approved on 2026-08-17. They were removed from the legacy
workspace; the three mirrored zero-byte files were also removed from the clean candidate
and unstaged. Zero approved Candidate C paths remain.

## Explicitly not deletion candidates

- The seven valid Git pack files (383.410 GiB).
- Any tracked deletion or modified notebook already present in `git status`.
- Any numerical dataset without a separately approved deletion, regardless of whether an
  external copy or integrity inventory exists. Active Colab data is not an archive or
  deletion candidate.
- `experiment_review/b0_exact_domain_wall/`, `experiment_review/b1_controller_frame/`,
  `validation_campaigns/`, or `topological_frustration_diagnostics/results/`.
- The current or clean-candidate repository until migration gates are complete.
