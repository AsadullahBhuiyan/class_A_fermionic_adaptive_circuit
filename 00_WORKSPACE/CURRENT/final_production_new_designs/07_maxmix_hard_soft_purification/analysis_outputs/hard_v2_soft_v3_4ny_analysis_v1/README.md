# Hard-v2 / soft-v3 purification Lyapunov analysis

This directory is the completed analysis of 600 independent maximally mixed
Born trajectories at `Nx=20`, `Ny=20,30,40`, and `T=4*Ny`.

The reader-facing deliverable is
`purification_lyapunov_hard_soft_note.pdf`.  It embeds both figures and states
the acceptance decisions, so the figure PDFs under `figure_assets/` are source
assets rather than separate reports.

The immutable inputs are deliberately distinct:

- hard/support-truncated: revision v2, 60 result/completion pairs, 300 trajectories;
- soft/untruncated: corrected revision v3, 60 pairs, 300 trajectories.

The checksum-pinned earlier hard-wall `T=2*Ny` v2 analysis is included only as
an independent depth diagnostic in `depth_comparison.csv`; none of its 800
trajectories are pooled with the new `T=4*Ny` ensembles.

Reproduce from the repository root with:

```bash
python 00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification/analyze_completed_campaign.py
cd 00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification/analysis_outputs/hard_v2_soft_v3_4ny_analysis_v1
pdflatex -interaction=nonstopmode -halt-on-error purification_lyapunov_hard_soft_note.tex
pdflatex -interaction=nonstopmode -halt-on-error purification_lyapunov_hard_soft_note.tex
```

The default analysis rehashes all 240 raw files against the immutable Drive
inventory.  `--skip-file-hashes` is a development shortcut and is not valid for
the final acceptance run.
