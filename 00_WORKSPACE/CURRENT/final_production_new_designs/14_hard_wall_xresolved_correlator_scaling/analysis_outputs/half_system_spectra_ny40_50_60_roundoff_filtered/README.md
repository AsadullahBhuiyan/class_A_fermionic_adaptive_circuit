# Half-system spectra with numerical endpoint exclusion

The modular-energy histograms retain only occupation eigenvalues satisfying

$$
10^{-12}<\nu_a<1-10^{-12},\qquad
\epsilon_a=\log\frac{1-\nu_a}{\nu_a}.
$$

This excludes the precision-sensitive tails shaded in the previous figure,
including all exact 0 and 1 occupations. No clipping or finite replacement of
excluded energies is used. The threshold is a practical numerical selection,
not a physical spectral boundary or a rigorous error bound on every retained
energy.

The inputs are the completed hard-wall x-resolved correlator campaign at Nx=20,
Ny=40,50,60, with S=100 independent trajectories per size. They use pure
half-filled initialization, alpha_1=1, alpha_2=30, nshell=1, raster-y
measurements, perfect correction, and T=2Ny cycles. The subsystem contains
both physical orbitals in A=[0,20)x[0,Ny/2). Its dimension is 800, 1,000,
or 1,200 per trajectory. Occupations are transformed per sample before pooling.

The selection retains 52,623 of 300,000 eigenvalues: 17,416 at Ny=40,
17,535 at Ny=50, and 17,672 at Ny=60. For each size and for the pooled
curve, normalization uses only the retained count:

$$
p_j=\frac{n_j}{M_{\mathrm{ret}}\,\Delta\epsilon_j},\qquad
\sum_j p_j\Delta\epsilon_j=1.
$$

Every retained level has equal weight. Modular bins have width 1 in the
interior and shorter boundary bins at approximately +/-27.63; normalization
uses each bin's actual width. Vertical axes are logarithmic and horizontal
axes are linear. The central-energy zoom shows the same density without
renormalizing its restricted view. No fit or uncertainty bars are added.

The occupation panel in the two-panel companion figure still shows all saved
occupations with its original unit-area normalization. The numerical exclusion
applies to the modular histograms requested in this revision. The previous
unfiltered figures are retained in the neighboring
`half_system_spectra_ny40_50_60/` directory.

## Files

- `half_system_modular_filtered_log_density.png` and `.pdf`: updated modular
  histogram with all sizes pooled in black and separate size overlays.
- `half_system_spectra_log_density.png` and `.pdf`: occupation and filtered
  modular histograms side by side.
- `half_system_modular_central_log_density.png` and `.pdf`: central zoom.
- `half_system_occupation_and_modular_spectra.npz`: all original sample spectra,
  `modular_retained_mask_Ny*` masks aligned with those spectra, retained energy
  vectors, histogram edges, counts, and normalized densities. Unfiltered
  energy arrays retain their infinities for provenance; only retained vectors
  enter the modular histograms.
- `histogram_*.csv`: bin bounds, counts, and density for each size and pool.
- `summary.json`: excluded/retained counts, input identity, and normalization.

Regenerate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 00_WORKSPACE/CURRENT/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling/analyze_half_system_spectra.py
```

Checks include all input hashes, scientific metadata, sample coverage, the
occupation-to-energy inverse, retained-bin count closure, and unit histogram
area. All full-resolution production files remain in the owning `gpu_data/`
directory.
