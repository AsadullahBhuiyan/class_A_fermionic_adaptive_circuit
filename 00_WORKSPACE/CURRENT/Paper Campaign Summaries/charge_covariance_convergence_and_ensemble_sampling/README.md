# Charge, covariance, and ensemble convergence working note

This directory contains the consolidated analysis note requested after the
charge-fluctuation and burn-in discussion.

## Main artifact

- `charge_covariance_convergence_and_ensemble_sampling_working_note.tex`
- `charge_covariance_convergence_and_ensemble_sampling_working_note.pdf`

The note uses the archived characterization figures directly from
`00_WORKSPACE/COLAB/colab_charge_fluctuations/analysis_outputs/` and the two
stationarity figures generated locally in this directory.

## Regeneration

From this directory:

```bash
python make_ensemble_sampling_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build \
  charge_covariance_convergence_and_ensemble_sampling_working_note.tex
cp build/charge_covariance_convergence_and_ensemble_sampling_working_note.pdf .
```

The analysis script is read-only with respect to production campaigns.  It
checks the archived trajectory grids, performs 2,000 whole-trajectory bootstrap
resamples, and writes its compact CSV/PDF/PNG outputs under `tables/` and
`figures/`.

## Source ensembles

- Maximally mixed purification: `N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny`
- Default pure-state streaming: `N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny`
- Sparse reduced-covariance snapshots in `colab_small_system_testing`
- Maximally mixed spatial entropy-contour campaigns in `colab_small_system_testing`

All dynamics were generated earlier through the canonical GPU Markov-circuit
entry point.  No dynamics are rerun by this note.
