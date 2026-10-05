# U(1) Kac–Moody and Rényi validation note

This directory contains the standalone one-column RevTeX note
`kac_moody_renyi_validation.tex`. It derives the Gaussian Rényi-entropy and
full-counting-statistics formulas, fixes the one-wall/two-wall Kac–Moody
normalization, separates conditional and record-ensemble observables, and defines a
reanalysis-first numerical validation ladder.

The note is deliberately separate from the manuscript. It does not modify simulation
APIs or source campaigns, and it does not treat static entropy or charge coefficients as
evidence of chirality.

## Contents

- `kac_moody_renyi_validation.tex`: canonical standalone note.
- `references.bib`: minimal primary-source bibliography.
- `build_validation_figures.py`: read-only analysis entry point; it hashes every input
  and writes figures, table fragments, machine-readable tables, and the analysis
  manifest.
- `analysis_manifest.json`: input/output hashes, source conventions, fit definitions,
  provenance, and extracted scalar results.
- `figures/`: vector PDF and 300-dpi PNG outputs.
- `tables/`: machine-readable tables and optional LaTeX fragments.
- `build/kac_moody_renyi_validation.pdf`: compiled canonical PDF.

The committed TeX source and generated Tier-0 products compile together. If generated
figures are deliberately removed, labeled fallbacks preserve compilability; locked
regression tables also have documented fallbacks. The following PDF files are included
automatically:

- `figures/figure_01_geometry_pipeline.pdf`
- `figures/figure_02_modular_kernels.pdf`
- `figures/figure_03_exact_b0_validation.pdf`
- `figures/figure_04_stochastic_legacy_validation.pdf`

The optional files `tables/exact_b0_regression.tex`,
`tables/result21_regression.tex`, and `tables/prior_production_regression.tex` are
LaTeX fragments containing complete `ruledtabular` and `tabular` blocks but no outer `table`
environment; the note supplies centering, captions, and labels.
`tables/scientific_claim_status.json` is the machine-readable scientific licensing
ledger; it is intentionally separate from implementation/pipeline integrity gates.

## Immutable numerical sources

The analysis resolves exact files through the source manifests and records their SHA-256
digests before reading them. The three source groups are:

1. Accepted exact B0 campaign:
   `00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall/results/20260816_191957/`
2. Audited Result-21 raw covariance runs and the matched archived `S_1` table:
   `00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/pure_state_covariance_snapshots/`
   and
   `00_WORKSPACE/COLAB/colab_small_system_testing/analysis_outputs/pure_state_entanglement_vs_system_size_cpu/full_x_late_window_log_chord_fit_rows.csv`
3. Prior-production common ensembles:
   `00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/production_10sample_v3_fixed_nx20_ny20_30_40_50_60/01_bulk_width_gate/`

No file beneath these roots may be changed by the builder. The canonical notation is an
occupation correlation matrix `G`; archived arrays called `C`, centered covariances, and
transposed conventions are converted only through their recorded metadata and are logged
in `analysis_manifest.json`.

## Run the read-only analysis

From this directory:

```bash
python build_validation_figures.py
```

The builder fails closed unless its output root is this canonical directory, and it
explicitly rejects output/source overlap with all immutable campaign-root families. The
committed implementation has completed Tier 0 only; it does not launch dynamics. Engine
compatibility replay (Tier 1) and targeted trajectory top-ups (Tier 2) require separate,
explicitly versioned work after the Tier-0 report is interpreted.

## Build the note

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build kac_moody_renyi_validation.tex
```

The expected PDF is `build/kac_moody_renyi_validation.pdf`.

## Interpretation guardrails

- Result 21 already contains matched von Neumann entropy and conditional quantum charge
  variance on the same four trajectory ensembles.
- The committed Tier-0 build now computes stochastic `S_2` and `S_3` directly from the
  archived raw covariance snapshots on those same trajectories. These are new reductions,
  not pre-existing restricted-spectrum products.
- The locked B0 numbers retain the historical explicit `1e-12` eigenvalue-floor estimator
  for exact regression compatibility. The endpoint-exact estimator is evaluated and
  recorded separately, together with the small difference between the two conventions.
- The prior-production archives support the paired `S_1`–charge test. They are not called
  a higher-Rényi dataset unless the necessary per-trajectory spectra are verified.
- The completed paired test does not pass its strict null uniformly: Result 21 at
  `Ny=30, nshell=2` excludes zero for `n=1,2,3`, and `Ny=30, nshell=1` excludes zero for
  `n=1`; some prior-production `S_1`–charge residual intervals also exclude zero. The
  scientific ledger therefore marks `minimal_u1_1` as `blocked_by_paired_null` while
  preserving separately valid near-one entropy and current-level estimates.
- Matched-trivial coefficient intervals pass the declared `1e-5` numerical-null bound,
  but their numerical-floor curves sometimes prefer the log model by AIC. Therefore the
  literal “trivial controls do not prefer log” criterion is recorded as
  `not_met_numerical_floor`, and overall scientific acceptance is qualified/not met even
  though all pipeline-integrity gates pass.
- Chirality requires a separate signed packet, response, or flux test.

The note also records, without editing the manuscript, that the numerical campaign’s
statement that no compatible legacy production comparison was identified is stale.
