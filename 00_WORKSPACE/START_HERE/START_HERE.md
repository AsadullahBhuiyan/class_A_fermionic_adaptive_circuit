# Start here

For the physical high-level tree, enter [`00_WORKSPACE/`](../README.md). Use its
`CURRENT/` directory for normal production work and `RECENCY_INDEX.md` when you want to
see what changed most recently.

For papers, theory notes, technical documentation, and working records, use the dedicated
[`NOTES/`](../../NOTES/README.md) library.

The project directories in [`CURRENT/`](../CURRENT/) are physical canonical directories,
not links. Colab packages are likewise physically grouped under [`COLAB/`](../COLAB/),
with their generated data retained inside each package.

## Current production workflow

1. Read `00_WORKSPACE/CURRENT/experiment_review/numerical_campaign_legacy_working.pdf`.
2. Use `00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/README.md` for the Colab bundle order.
3. Use `00_WORKSPACE/CURRENT/mean_channel_lindblad_cpu_campaign/README.md` for the
   sample-free exact mean-channel/Lindblad CPU campaign; it is not a Colab package.
4. Edit canonical dynamics only in `src/fgtn/`.
5. Run maintained tests from `tests/`.
6. Treat `00_WORKSPACE/CURRENT/validation_campaigns/` and
   `00_WORKSPACE/CURRENT/topological_frustration_diagnostics/results/` as
   recent evidence, not as disposable caches.

## Storage rules

- Raw covariance histories and campaign outputs do not belong in Git.
- Current accepted evidence remains local until it has a verified external copy.
- Active Colab data and standalone archive candidates are distinguished in [`DATA_CATALOG.md`](../../PROJECT_ADMIN/DATA_CATALOG.md).
- External offload instructions are in [`ARCHIVE_RUNBOOK.md`](../../PROJECT_ADMIN/ARCHIVE_RUNBOOK.md).
- The reversible Git successor workflow is in [`GIT_MIGRATION_PLAN.md`](../../PROJECT_ADMIN/GIT_MIGRATION_PLAN.md).
- The completed deletion record is in [`DELETION_CANDIDATES.md`](../../PROJECT_ADMIN/DELETION_CANDIDATES.md).
- Requirement-by-requirement status is in [`COMPLETION_AUDIT.md`](../../PROJECT_ADMIN/COMPLETION_AUDIT.md).
- Do not delete a dataset until its destination checksum has been verified and an archive
  receipt has been left at the original logical location.
- Ask the user before every file or directory deletion.

## Current archive status

`/data/abhuiyan` is writable, but the first copy attempt was canceled after its dataset
classification was corrected. Colab-generated data remains active and co-located with its
notebooks. The incomplete destination copy is preserved pending a separate user decision.
