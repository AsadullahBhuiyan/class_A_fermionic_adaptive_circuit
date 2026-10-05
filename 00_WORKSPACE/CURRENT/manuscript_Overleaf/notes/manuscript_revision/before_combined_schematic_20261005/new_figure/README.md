# Manuscript figure bundle

Fourteen figure versions, each with a vector PDF and 300-dpi PNG: Figures 1–2 belong to Section II, Figures 3–11 to Section III, and A1–A2 are the only appendix figures. The soft/hard-wall alternative is retained here but excluded from the manuscript. Original figures, the earlier `restructured` bundle, and campaign data remain unchanged; this bundle accompanies the authorized manuscript revision.

[Combined figure notes](FIGURE_NOTES.md) · [Ordered overview](overview.png) · [Verification](validation.json)

## Ordered index

| Figure | Section | Contents | Previous figure | Change | Files |
|---|---|---|---|---|---|
| 1 | II | Domain-wall adaptive circuit | 1 | Outcome label uses upright m | [PDF](Figure_01_schematic.pdf) · [Preview](Figure_01_schematic.png) · [Notes](FIGURE_NOTES.md#figure-01-schematic) |
| 2 | II | Hard-wall support truncation | 2 | Consistent typography; diagram geometry preserved | [PDF](Figure_02_hard_wall.pdf) · [Preview](Figure_02_hard_wall.png) · [Notes](FIGURE_NOTES.md#figure-02-hard-wall) |
| 2 alt | Not included | Soft and hard walls | Original attachment | Consistent typography; diagram geometry preserved; excluded from manuscript | [PDF](Figure_02_alt_soft_and_hard_walls.pdf) · [Preview](Figure_02_alt_soft_and_hard_walls.png) · [Notes](FIGURE_NOTES.md#figure-02-alt-soft-and-hard-walls) |
| 3 | III | Bulk topology | 11 + recent bulk validation | 3×1; taller convergence panel; larger marker map and shorter colorbar; data preserved | [PDF](Figure_03_bulk_topology.pdf) · [Preview](Figure_03_bulk_topology.png) · [Notes](FIGURE_NOTES.md#figure-03-bulk-topology) |
| 4 | III | Slow purification | 7 | Overlined means; original four panels and fit preserved | [PDF](Figure_04_purification.pdf) · [Preview](Figure_04_purification.png) · [Notes](FIGURE_NOTES.md#figure-04-purification) |
| 5 | III | Correlation functions | 3 | Remove x=6,14 from B only; common D chord label | [PDF](Figure_05_correlations.pdf) · [Preview](Figure_05_correlations.png) · [Notes](FIGURE_NOTES.md#figure-05-correlations) |
| 6 | III | Entropy and charge fluctuations | 4A/B | Two panels; omit Ay=1; overlined means; common D chord label | [PDF](Figure_06_entropy_charge.pdf) · [Preview](Figure_06_entropy_charge.png) · [Notes](FIGURE_NOTES.md#figure-06-entropy-charge) |
| 7 | III | Occupation spectrum, energy spectrum, and mode count | Replaces 4C; recent spectra analysis | Compact 3×1; both normalized histograms pool all 32 origins; raw alpha1=1 count preserved; common D chord label; fit annotation repositioned | [PDF](Figure_07_entanglement_spectrum.pdf) · [Preview](Figure_07_entanglement_spectrum.png) · [Notes](FIGURE_NOTES.md#figure-07-entanglement-spectrum) |
| 8 | III | Central-charge convergence and size dependence | 8 + recent endpoint-size panel | Recent two-panel figure; 2×1 vertical | [PDF](Figure_08_central_charge.pdf) · [Preview](Figure_08_central_charge.png) · [Notes](FIGURE_NOTES.md#figure-08-central-charge) |
| 9 | III | Entropy carried by each wall | 5 | Two-column windows; omit Ay=1; overlined means; common D chord label | [PDF](Figure_09_wall_entropy.pdf) · [Preview](Figure_09_wall_entropy.png) · [Notes](FIGURE_NOTES.md#figure-09-wall-entropy) |
| 10 | III | Modular evolution | 6 | Three original snapshot times; overlined mean displacement | [PDF](Figure_10_modular_evolution.pdf) · [Preview](Figure_10_modular_evolution.png) · [Notes](FIGURE_NOTES.md#figure-10-modular-evolution) |
| 11 | III | Trajectory-averaged dynamics and channel gap | 9 | Compact 4×1; dimensionless channel-gap scan and inverse-length fit; common D chord label | [PDF](Figure_11_mean_channel.pdf) · [Preview](Figure_11_mean_channel.png) · [Notes](FIGURE_NOTES.md#figure-11-mean-channel) |
| A1 | Appendix | Truncated OW modes | 10 | Consistent typography; scientific curves and fits preserved | [PDF](Figure_A01_ow_truncation.pdf) · [Preview](Figure_A01_ow_truncation.png) · [Notes](FIGURE_NOTES.md#figure-a01-ow-truncation) |
| A2 | Appendix | Entropy contours and antipodal mutual information | 15 | 2×1: all-origin entropy-contour comparison above unchanged mutual information | [PDF](Figure_A02_mutual_information.pdf) · [Preview](Figure_A02_mutual_information.png) · [Notes](FIGURE_NOTES.md#figure-a02-mutual-information) |

Old Figures 12–14 and the old Figure 4(c) are excluded. The old tri-junction schematic is incorporated into Figure 3; Figure 7 contains the replacement occupation and entanglement-energy comparisons.

## Data and interpretation

- Figure 3 uses the complete S=100 square-system Chern analysis. The finite-disk estimator and local marker are distinct observables; the marker retains periodic-coordinate seam effects.
- Figure 4 retains its original purification data and independent gap-size ensemble. Scientific protocol details remain in the combined note.
- Figure 7 pools all 32 strip origins within each of 100 trajectories per parameter value at Ny=32. The full occupation histograms contain 2,048,000 observations each; the conditional energy histograms contain 95,398 and 83,326 retained observations for alpha1=1 and 3. Both histogram types integrate to one. Panel (c) retains the raw mean window count for alpha1=1 and its unchanged fit and trajectory SEM.
- Figure 8 uses separate ensembles. Panel (a) bars are regression errors of mean-entropy fits; panel (b) bars propagate trajectory sampling fluctuations.
- Figure 9 integrates two-column wall windows x=5,6 and x=14,15. Its displayed Ay=1 points are excluded without changing the fits.
- Figure 10 retains snapshots at modular times 0, 0.1, and 0.2. Absolute direction depends on the explicitly stated correlation-matrix index convention; the numerical curves are preserved.
- Figure 11 uses deterministic outcome-averaged dynamics and the dimensionless channel gap g_C=1−rho(A)^2. The inverse-length fits use fixed Nx=20; they are not a simultaneous two-dimensional thermodynamic extrapolation.
- Figure A2(a) compares origin-averaged half-strip entropy contours for alpha1=1 and 3 at Nx=20, Ny=32, Ay=16, cycle 64: average all 32 origins within each trajectory, then average 100 trajectories. The maps share a square-root color scale. Panel (b) preserves the 63-point mutual-information scan and trajectory SEMs.

All input data, fit windows, averaging order, uncertainties, exclusions, and notation conventions are documented in the combined note. No new circuit simulations were used.

## Reproduction

Run the matching renderer from this directory. Renderers use bundled compact data and write only inside this bundle. The original large trajectory datasets are not needed for plotting.

| Figure | Command |
|---|---|
| 1 | `python sources/plot_schematic.py` |
| 2 | `python sources/plot_wall_schematics.py` |
| 2 alt | `python sources/plot_wall_schematics.py` |
| 3 | `python sources/plot_bulk_topology.py` |
| 4 | `python sources/plot_purification.py` |
| 5 | `python sources/plot_correlations.py` |
| 6 | `python sources/plot_entropy_charge.py` |
| 7 | `python sources/plot_entanglement_spectrum.py` |
| 8 | `python sources/plot_central_charge.py` |
| 9 | `python sources/plot_wall_entropy.py` |
| 10 | `python sources/plot_modular_evolution.py` |
| 11 | `python sources/plot_mean_channel.py` |
| A1 | `python sources/plot_ow_truncation.py` |
| A2 | `python sources/plot_mutual_information.py` |

All fourteen versions now use dedicated renderers to preserve their typography updates. `restore_existing.py` rejects restoring an outdated figure over these products; its `--check-only` mode verifies original source assets without writing files.

Requirements: Python, NumPy, Matplotlib, Pillow, pypdf, Poppler, and a working LaTeX installation with AMS, bm, type1cm/type1ec (cm-super), and dvipng. All figure text and mathematics are rendered through LaTeX in Computer Modern; no font fallback is allowed. The shared sources/manuscript_typography.py enforces final-print sizes: axes and panel letters 9 pt, ticks/legends/annotations 8 pt, prominent schematic labels 10–11 pt. Figure 2 compensates for its 0.8-column inclusion. Per-figure records are in data/typography/. Figures 3, 7, and 11 retain their stacked layouts; A1 retains its original canvas aspect ratio.

After intentional edits, rebuild this index and overview with `python sources/build_index.py`, then record the validated bundle with `python sources/verify_bundle.py --record`. Use `python sources/verify_bundle.py` for a read-only check. The verifier checks the original assets against `notes/manuscript_revision/baseline/original_figure_checksums.json`; it permits the authorized manuscript and bibliography revision. The top-level `manifest.json` binds the delivered bundle to SHA-256 checksums.

## Separate diagnostics

[Earlier fixed-origin energy comparison](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.pdf) and [normalized mode-fraction diagnostic](diagnostics/normalized_mode_fraction_vs_log_chord.pdf) are retained for reference. Neither is included in the manuscript; Figure 7 uses all origins and raw mean counts.
