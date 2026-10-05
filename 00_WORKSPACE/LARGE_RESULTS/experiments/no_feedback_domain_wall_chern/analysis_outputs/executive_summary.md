# Executive Summary

Analyzed 6 completed runs (60 per-sample files): three campaigns, two initial states, 10 samples each, 40 cycles.
All run configs record `p_gain_eff=0`, `p_loss_eff=0`, `radius=6`, and `nshell=1`.

## Final Real-Space Chern
- Uniform alpha=1 / Random pure: final Chern -0.04638 ± 0.08099; mean change from cycle 0 = -0.004868.
- Uniform alpha=1 / Maxmix: final Chern 0.02704 ± 0.07391; mean change from cycle 0 = 0.02704.
- DW alpha=1/30, dwtrunc=0 / Random pure: final Chern 0.08243 ± 0.08743; mean change from cycle 0 = -0.05703.
- DW alpha=1/30, dwtrunc=0 / Maxmix: final Chern 0.04206 ± 0.07011; mean change from cycle 0 = 0.04206.
- DW alpha=1/30, dwtrunc=1 / Random pure: final Chern -0.006265 ± 0.06619; mean change from cycle 0 = 0.07162.
- DW alpha=1/30, dwtrunc=1 / Maxmix: final Chern -0.0934 ± 0.07461; mean change from cycle 0 = -0.0934.

## Local Chern Marker
- Uniform alpha=1 / Random pure: full-system mean -0.006654 ± 0.01861.
- Uniform alpha=1 / Maxmix: full-system mean -0.0178 ± 0.01469.
- DW alpha=1/30, dwtrunc=0 / Random pure: full-system mean 0.007007 ± 0.009861; DW slab mean 0.008646 ± 0.01429.
- DW alpha=1/30, dwtrunc=0 / Maxmix: full-system mean 0.004464 ± 0.004593; DW slab mean 0.003608 ± 0.006251.
- DW alpha=1/30, dwtrunc=1 / Random pure: full-system mean -0.01289 ± 0.004252; DW slab mean -0.01983 ± 0.006541.
- DW alpha=1/30, dwtrunc=1 / Maxmix: full-system mean 0.008542 ± 0.008265; DW slab mean 0.01314 ± 0.01272.

## Transfer Spectra
- DW alpha=1/30, dwtrunc=0 / Random pure: transfer gap 0.005201 ± 0.0006839; basis size 800; all Choi trajectories active.
- DW alpha=1/30, dwtrunc=0 / Maxmix: transfer gap 0.003871 ± 0.0008168; basis size 800; all Choi trajectories active.
- DW alpha=1/30, dwtrunc=1 / Random pure: transfer gap 0.002554 ± 0.0005582; basis size 520; all Choi trajectories active.
- DW alpha=1/30, dwtrunc=1 / Maxmix: transfer gap 0.004985 ± 0.0008096; basis size 520; all Choi trajectories active.

## Main Figures
- `figures/chern_vs_cycle_overview.png` and `figures/chern_vs_cycle_overview.pdf`
- `figures/chern_vs_cycle_by_campaign.png` and `figures/chern_vs_cycle_by_campaign.pdf`
- `figures/local_chern_marker_grid.png` and `figures/local_chern_marker_grid.pdf`
- `figures/local_chern_marker_init_differences.png` and `figures/local_chern_marker_init_differences.pdf`
- `figures/transfer_exponent_spectra.png` and `figures/transfer_exponent_spectra.pdf`
- `figures/transfer_gap_summary.png` and `figures/transfer_gap_summary.pdf`
