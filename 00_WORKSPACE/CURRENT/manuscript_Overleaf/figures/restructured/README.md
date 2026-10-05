# Restructured manuscript figures

Fourteen figure versions: eleven main-text figures, one Figure 2 alternative, and two appendix figures. Each has a vector PDF and 300-dpi PNG preview. All data, protocols, analysis, and reproduction details are collected in one combined note. The manuscript LaTeX, bibliography, original figures, and campaign data were preserved.

[Combined figure notes](FIGURE_NOTES.md) · [Open the overview](overview.png) · [Verification](validation.json)

## Ordered index

| New | Figure | Previous figure | Change | Files |
|---|---|---|---|---|
| 1 | Domain-wall adaptive circuit | 1 | Preserved | [PDF](Figure_01_schematic.pdf) · [Preview](Figure_01_schematic.png) · [Notes](FIGURE_NOTES.md#figure-01-schematic) |
| 2 | Hard-wall support truncation | 2 | Preserved | [PDF](Figure_02_hard_wall.pdf) · [Preview](Figure_02_hard_wall.png) · [Notes](FIGURE_NOTES.md#figure-02-hard-wall) |
| 2 alt | Soft and hard walls | Original attachment | Alternative | [PDF](Figure_02_alt_soft_and_hard_walls.pdf) · [Preview](Figure_02_alt_soft_and_hard_walls.png) · [Notes](FIGURE_NOTES.md#figure-02-alt-soft-and-hard-walls) |
| 3 | Bulk topology | 11 + recent bulk validation | Three-panel composition; 3×1 vertical | [PDF](Figure_03_bulk_topology.pdf) · [Preview](Figure_03_bulk_topology.png) · [Notes](FIGURE_NOTES.md#figure-03-bulk-topology) |
| 4 | Slow purification | 7 | Preserved | [PDF](Figure_04_purification.pdf) · [Preview](Figure_04_purification.png) · [Notes](FIGURE_NOTES.md#figure-04-purification) |
| 5 | Correlation functions | 3 | Remove x=6,14 from B only | [PDF](Figure_05_correlations.pdf) · [Preview](Figure_05_correlations.png) · [Notes](FIGURE_NOTES.md#figure-05-correlations) |
| 6 | Entropy and charge fluctuations | 4A/B | Standalone; omit Ay=1 | [PDF](Figure_06_entropy_charge.pdf) · [Preview](Figure_06_entropy_charge.png) · [Notes](FIGURE_NOTES.md#figure-06-entropy-charge) |
| 7 | Occupation spectrum, energy spectrum, and mode count | Replaces 4C; recent spectra analysis | 3×1; both histograms normalized and pooled over all y0; raw mean count and fit preserved | [PDF](Figure_07_entanglement_spectrum.pdf) · [Preview](Figure_07_entanglement_spectrum.png) · [Notes](FIGURE_NOTES.md#figure-07-entanglement-spectrum) |
| 8 | Central-charge convergence and size dependence | 8 + recent endpoint-size panel | Recent two-panel figure; 2×1 vertical | [PDF](Figure_08_central_charge.pdf) · [Preview](Figure_08_central_charge.png) · [Notes](FIGURE_NOTES.md#figure-08-central-charge) |
| 9 | Entropy carried by each wall | 5 | Omit Ay=1; two-column windows | [PDF](Figure_09_wall_entropy.pdf) · [Preview](Figure_09_wall_entropy.png) · [Notes](FIGURE_NOTES.md#figure-09-wall-entropy) |
| 10 | Modular evolution | 6 | Preserve original three times | [PDF](Figure_10_modular_evolution.pdf) · [Preview](Figure_10_modular_evolution.png) · [Notes](FIGURE_NOTES.md#figure-10-modular-evolution) |
| 11 | Trajectory-averaged dynamics and channel gap | 9 | Original A/B data; dimensionless gap g_C = 1 − ρ(A)² in C and direct inverse-length fits in D; 4×1 | [PDF](Figure_11_mean_channel.pdf) · [Preview](Figure_11_mean_channel.png) · [Notes](FIGURE_NOTES.md#figure-11-mean-channel) |
| A1 | Truncated OW modes | 10 | Preserved appendix figure | [PDF](Figure_A01_ow_truncation.pdf) · [Preview](Figure_A01_ow_truncation.png) · [Notes](FIGURE_NOTES.md#figure-a01-ow-truncation) |
| A2 | Antipodal mutual information | 15 | Preserved appendix figure | [PDF](Figure_A02_mutual_information.pdf) · [Preview](Figure_A02_mutual_information.png) · [Notes](FIGURE_NOTES.md#figure-a02-mutual-information) |

The old Figure 11 tri-junction is incorporated into new Figure 3. Old Figures 12–14 and the old Figure 4C occupation panel are excluded from this collection; Figure 7 contains a new matched-parameter occupation comparison. Their original manuscript assets remain in the parent directory.

## Data and interpretation

- Figure 3 uses a 3×1 vertical layout and the complete S=100 square-system Chern analysis. Its disk estimator and local-marker map are different observables.
- Figure 4 intentionally combines full-system measurements in A/B/D with the separate slab-only size sweep in C.
- Figure 7 uses a 3×1 vertical layout at Ny=32. Both histograms pool all 32 cut origins and 100 trajectories per parameter value: full-range occupation densities (2,048,000 observations each) and conditional energy densities (95,398 retained for alpha_1=1; 83,326 for alpha_1=3), each normalized to unit area. Panel C preserves the raw mean mode count for alpha_1=1, its fit, and trajectory SEM. The combined note discusses Section V of Eisler and Peschel (2010).
- Figure 9 uses two-column wall windows. Its note records the mismatch with the preserved manuscript caption.
- Figure 8 uses a 2×1 vertical layout and separate ensembles: cycle-panel bars are regression errors, while endpoint-panel bars are trajectory SEMs.
- Figure 11 uses a 4×1 vertical layout: mean-state occupation spectrum, squared mean-state correlator, dimensionless channel gap g_C = 1 − ρ(A)² versus alpha_1 at fixed Nx=20, and direct inverse-Ny fits of g_C. The note identifies both replacement PDFs and relates this multiplier gap to the logarithmic rate used in the linked proof.

## Separate diagnostics

[Normalized entanglement-energy comparison, alpha_1=1 versus 3](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.pdf) · [Preview](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.png) · [Data, normalization, and interpretation](FIGURE_NOTES.md#normalized-energy-comparison). Both curves use Ny=32, Ay=16 and the same fixed-origin cut; each integrates to one inside the window |lambda| <= 0.99. This diagnostic is separate from Figure 7. Reproduce it with `python sources/plot_energy_window_comparison.py`.

[Normalized mode fraction versus log chord length](diagnostics/normalized_mode_fraction_vs_log_chord.pdf) · [Preview](diagnostics/normalized_mode_fraction_vs_log_chord.png) · [Notes](FIGURE_NOTES.md#normalized-mode-fraction). Standalone Figure 7(c) variant for alpha_1=1: window count divided by all 2NxAy subsystem modes, averaged over 32 origins within each of 100 trajectories. The existing count fit is divided by the same denominator. Reproduce it with `python sources/plot_normalized_mode_fraction.py`.

## Reproduction

From this directory, run the corresponding renderer below. Modified figures use only the compact inputs in `data/`; no simulation is run. Unchanged/reused figures can also be restored from their checksum-verified source assets in this repository. The original soft/hard-wall alternative is retained inside the bundle for restoration.

| Figure | Command |
|---|---|
| 3 | `python sources/plot_bulk_topology.py` |
| 5 | `python sources/plot_correlations.py` |
| 6 | `python sources/plot_entropy_charge.py` |
| 7 | `python sources/plot_entanglement_spectrum.py` |
| 8 | `python sources/plot_central_charge.py` |
| 9 | `python sources/plot_wall_entropy.py` |
| 11 | `python sources/plot_mean_channel.py` |

Restore one reused figure with `python sources/restore_existing.py --figure Figure_01_schematic`; omit `--figure` to restore all nine unchanged/reused assets. Figures 8 and 11 are regenerated by their plotting scripts to preserve the new layouts and panels. Use `--output-dir /tmp/figure-preview` to redirect restoration. Figures 5, 6, 8, 9, and 11 also support `--output-dir` for isolated plotting. Figures 3 and 7 write beside their bundled sources.

Requirements: Python, NumPy, Matplotlib, and Pillow for the overview. Figure 7 additionally requires LaTeX and Poppler (`pdftoppm`); its TeX compatibility file is included. Font choices are recorded in the plotting scripts. Existing figures retain their selected appearance.

Regenerate this index and overview with `python sources/build_index.py`. Check the bundle with `python sources/verify_bundle.py`. Data provenance and numerical checks are saved in the per-figure `data/` directories. The top-level `manifest.json` binds the delivered assets to checksums.
