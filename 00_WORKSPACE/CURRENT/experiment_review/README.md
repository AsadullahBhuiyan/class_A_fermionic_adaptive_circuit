# Experiment review workspace

This directory is the canonical home for review documents, reproducible review scripts,
and newly generated review data. Existing legacy datasets remain in their original
locations and are referenced by path and source hash rather than copied here.

## Contents

- `numerical_campaign_legacy_working.tex` — working copy of the numerical production
  contract with audited legacy-result annotations. The production contract at
  `00_WORKSPACE/CURRENT/prxq_draft/numerical_campaign.tex` is intentionally untouched.
- `b0_exact_domain_wall/` — deterministic CPU campaign for Campaign B0, including its
  tmux launcher, locked configuration, tests, and timestamped results.
- `b1_controller_frame/` — preserved CPU B1 pilot and its immutable partial artifacts.
  The simplified production implementation is the optional A100 bundle
  `final_production_ready_figure_scripts/prior_designs/06_b1_controller_frame/`.

The working production contract explicitly retires tripartite mutual information (TMI):
archived TMI reductions are provenance only, and no new TMI run, replay, gate, or figure
panel belongs to the executable campaign.

## Current audited campaign

Campaign B0 is complete in
`00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall/results/20260816_191957/`.
Its consecutive
full-calibration width rule selects `Nx=20`, and every item in its acceptance ledger passes.
The earlier run `20260816_184251` is preserved but superseded for interpretation because
its center-injected modular COM estimator cancelled the relevant endpoint dynamics and its
mass-only width rule selected locally underconverged `Nx=12` data.

Generated campaign directories are self-describing. Each contains a manifest, atomic
geometry products, tables, figures, logs, and a standalone RevTeX report.

## B1 transition to GPU

CPU campaign `20260816_225717` was stopped intentionally on 2026-08-17 after completing
all eight static cases and eight trajectories. Its files, manifest, and logs are preserved
unchanged as partial provenance; it must not be resumed or pooled with the new protocol.
The replacement `06_b1_controller_frame` bundle uses `Nx=20`, `Ny=48`, 25 trajectories
for the explicit interface and 100 for its all-trivial full-support control, with exactly
`2*Ny` cycles and no burn-in. It tests finite-window approach to the artificial signed
controller-frame ground-state manifold. It uses neither charge-sector weights nor a
training/test split and remains the lowest-priority nonblocking experiment.

The preserved aborted attempts document three pre-production launcher/validation defects:
`20260816_204900` captured a wait message as part of its CPU list; `20260816_204938`
failed an incorrect projector-normalization preflight reduction; and `20260816_205624`
waited indefinitely because CPU discovery parsed whole `/proc/stat` lines rather than
their leading `cpuN` tokens. None produced static or trajectory production data. Version 2
corrects CPU discovery, passes all seven tests and preflight checks, and supersedes them.
