# Fixed-T60 gaps, all seven sizes

Panel C: Campaign 13, Nx=20, Ny=(20, 24, 30, 36, 44, 56, 60), S=100 per size, hard walls,
alpha1=1, alpha2=30, nshell=1, raster-y, complex128, slab-only measurement
with Born-conditioned exterior. All 140 result/receipt pairs independently
verified. Extract exact saved cycle 60, no interpolation or simulation.
Compute min absolute log[(1-nu)/nu] per trajectory after excluding saved
pure caps, divide by 120, then average. Uncertainty is sample SD/sqrt(100).

| Ny | Mean gap ± SEM |
|---:|---:|
| 20 | 0.052272 ± 0.002262 |
| 24 | 0.042493 ± 0.001822 |
| 30 | 0.028034 ± 0.001465 |
| 36 | 0.019143 ± 0.001288 |
| 44 | 0.016207 ± 0.001046 |
| 56 | 0.011828 ± 0.000761 |
| 60 | 0.008625 ± 0.000634 |

SEM-weighted log-space power fit: z=1.566897 ± 0.054378,
chi-square=9.5111 for 5 degrees of freedom.
Fit errors propagate absolute sampling SEMs, with no residual rescaling or
bootstrap. This is a descriptive finite-time fit, not an asymptotic exponent;
fit-window sensitivity is saved separately. Raw modular gaps have the same
exponent because the time divisor is constant across sizes.

A/B retain raw cycles 1..60 and log-log axes; C retains log-log axes and
the latest tick styling. D is unchanged at T60, with time in caption only.
A/B/D are the full-measurement Ny30 Campaigns 21/22, not the slab-only
protocol in C. Sharing an observation time does not remove that distinction.
Numerical arrays in A/B/D are checked for exact equality to prior products.
Previous figures and manuscript are untouched. Reproduce with
python remake_gap_t60_all_sizes.py.
