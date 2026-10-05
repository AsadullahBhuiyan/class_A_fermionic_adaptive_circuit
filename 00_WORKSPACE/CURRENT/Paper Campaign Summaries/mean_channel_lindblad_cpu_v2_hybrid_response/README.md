# `mean_channel_lindblad_cpu_v2_hybrid_response` campaign summary

This folder contains a concise RevTeX working note for the completed deterministic
mean-channel/Lindblad campaign.  The note distinguishes the stationary occupation
spectrum, continuous two-point generator, finite completely-positive channel,
and the trajectory-resolved adaptive circuit. It also places the chiral-looking
occupation branches in the mixed-state topology literature and explains why they
are purity-spectrum flow rather than, by themselves, gapless Liouvillian modes.

The notation follows Bhuiyan--Pan--Jian, *Phys. Rev. Research* **8**, 023147
(2026). In particular,

```text
G_ij = Tr(rho c_i^dagger c_j),       C_arch = G^T = G^*
alpha_PRR = -alpha_run
L_G = L_gain + L_loss
L_cycle = L_G + L_dephas.
```

The campaign computes only the exactly closed Gaussian sector `L_G`; it does not
include the paper's number-dephasing sector or the ordered projective adaptive
circuit. Archived alpha labels and plotted values remain in the run convention.

The compiled working note is
`mean_channel_lindblad_cpu_v2_hybrid_response_working_note.pdf` (9 pages,
double-column RevTeX). Every figure is preceded by the equations defining its
plotted estimators and followed by the corresponding numerical interpretation.

The production data remain authoritative at:

`../../mean_channel_lindblad_cpu_campaign/results/mean_channel_lindblad_cpu_v2_hybrid_response_production_df092e76037c014f/`

Regenerate the overlaid occupation-spectrum and directional-response figures, then
compile the note from this directory with:

```bash
python make_supplementary_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build mean_channel_lindblad_cpu_v2_hybrid_response_working_note.tex
```

The figure generator reads only immutable production NPZ files. Before and after
plotting it verifies all 132 archived case-file hashes against the production
manifest, and it asserts the archived slopes, damping rates, response velocities,
retention, fit quality, finite-channel convergence orders, the transformed
Sylvester equation for `G`, and equality of the `G`- and `C_arch`-convention
density responses. Generated figures
are written under `figures/`; neither the solver, data schema, public APIs, nor the
production archive are modified.

The spectrum generator overlays every $G_{\rm ss}(k_y)$ eigenvalue for
`n_shell=1`, `n_shell=2`, and the full OW frame on a shared axis. It deliberately
draws no wall-branch selection or fit curve; the branch-selection equations in
the note are used only for the separately reported slopes.
