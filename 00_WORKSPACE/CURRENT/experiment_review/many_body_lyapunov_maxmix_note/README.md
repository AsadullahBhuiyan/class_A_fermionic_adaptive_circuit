# Maximally mixed many-body Lyapunov note

> **Status (2026-09-09):** the PDF in this directory is the immutable
> historical `T=2Ny` diagnostic.  The canonical completed `T=4Ny` hard-v2 /
> soft-v3 analysis, including both figures in one reader-facing RevTeX PDF and
> an explicit non-pooled comparison to this pilot, is
> `../../final_production_new_designs/07_maxmix_hard_soft_purification/analysis_outputs/hard_v2_soft_v3_4ny_analysis_v1/purification_lyapunov_hard_soft_note.pdf`.
> The historical files are retained for provenance rather than overwritten.

This directory contains a two-column RevTeX working note that explains how a
Born-sampled maximally mixed Gaussian trajectory determines the many-body
singular spectrum of its resolved many-body evolution operator.

The note distinguishes three layers that must not be conflated:

1. the normalized spectrum reconstructed from one-particle occupations;
2. the absolute scalar normalization restored from the realized record log
   probability; and
3. the finite-size CFT interpretation of the resulting Lyapunov rates.

It follows Eqs. (3) and (4) of Zabalo *et al.* and explains that no spacetime
anisotropy is needed to collect the spectrum or record free energy. An
independent absolute Lyapunov-only value of the effective central charge does
require that normalization. A ratio of accepted finite-size coefficients would
cancel the anisotropy, but the completed v2 coefficient gates fail, so no such
ratio or operator dimension is reported here.

Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  many_body_lyapunov_maxmix_note.tex
```

The local validation implementation lives at
`../../many_body_lyapunov_maxmix_cpu_pilot/`. The production pilot is the
standalone A100 bundle
`../../final_production_new_designs/04_maxmix_manybody_lyapunov_pilot/`:
`Nx=20`, `Ny=20,22,24,26,28,30,36,40`, `T=2Ny`, and 100 independent records
per size. All 160 tasks and 800 trajectories have been analyzed. The note now
reports successful spectrum reconstruction and 95--97% boundary localization,
but every size fails at least one middle/late whole-trajectory convergence
test. It therefore recommends, without launching, a fresh `T=4Ny`, `S=100`
study at `Ny=20,30,40`.

Inputs consulted:

- Zabalo *et al.*, arXiv:2107.03393v3;
- the project notes on Gaussian transfer matrices, Nambu regularization, Choi
  compression, and tangent dynamics supplied with the request;
- `../legacy_evidence_figure_atlas/legacy_evidence_figure_atlas.pdf`.
