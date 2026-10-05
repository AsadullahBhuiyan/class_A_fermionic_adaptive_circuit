# Campaign 26: fixed-width, fixed-time gap scaling

## Data and estimator

All 16 result/completion pairs were revalidated against SHA-256, byte count,
task, seed, configuration, source identity and sample coverage. There are 10
independent trajectories at each Ny=20,24,28,32,36,40,44,48, with Nx=20 and
T=40 for every trajectory. Hard walls, maxmix initialization, Born-conditioned
exterior followed by slab-only measurements, perfect correction, raster_y,
alpha1=1, alpha2=30, nshell=1, complex128; no covariance clipping.

For each trajectory, epsilon_j=log[(1-nu_j)/nu_j], g_mod=min_j|epsilon_j|,
and Delta=g_mod/(2T)=g_mod/80. Compute the minimum **before** averaging samples.
This is neither the gap of an averaged spectrum nor the spectrum of an averaged
covariance. It is the active-slab finite-time Lyapunov half gap, not a
half-system entanglement-spectrum gap. The full symmetric gap convention would
double every value without changing the size exponent.

All errors on the means below are sample SD/sqrt(10), with no bootstrap.

| Ny | Raw modular gap ± SEM | Finite-time half gap ± SEM |
|---:|---:|---:|
| 20 | 2.9692 ± 0.3908 | 0.03712 ± 0.00489 |
| 24 | 2.3885 ± 0.5097 | 0.02986 ± 0.00637 |
| 28 | 1.8331 ± 0.3040 | 0.02291 ± 0.00380 |
| 32 | 1.9187 ± 0.2741 | 0.02398 ± 0.00343 |
| 36 | 1.4088 ± 0.2147 | 0.01761 ± 0.00268 |
| 40 | 1.7044 ± 0.2211 | 0.02131 ± 0.00276 |
| 44 | 0.7785 ± 0.3133 | 0.00973 ± 0.00392 |
| 48 | 0.9289 ± 0.2383 | 0.01161 ± 0.00298 |

## Size trend and exploratory fits

The endpoint contrast Delta(20)-Delta(48) is 0.025504
± 0.005722, combining independent SEMs in quadrature
(4.46 combined SEMs). The central decrease is
68.7%. It is not pointwise monotonic.
Since 2T=80 is constant, the raw modular gap falls by precisely the same
fraction. The trend therefore cannot be caused solely by a size-dependent
1/(2T) denominator.

Fit the eight ensemble means in **linear gap space**, minimizing
sum_Ny [(mean Delta - A*(Ny/32)^(-z))/SEM]^2. No logarithmic data fit or
sample-wise power-law fitting is used. The result is z=1.1705
± 0.2127, A=0.021504 ± 0.001296.
Parameter errors use the local weighted-fit covariance with absolute sampling
SEMs, without rescaling by residual chi-square. These are approximate
linearized one-standard-error uncertainties, not bootstrap intervals.

Power law: chi-square=6.027 for 6 degrees of freedom.
Fixed inverse-size model: chi-square=6.680
for 7 degrees of freedom. A constant gap gives
chi-square=32.316 for 7
degrees of freedom. Thus this range favors a decreasing gap and is compatible
with 1/Ny; a precise universal exponent is not established. The chi-square
probabilities in the JSON are only approximate because SEMs are themselves
estimated from ten samples and sample gaps need not be Gaussian.

The alternative Delta=b+a*(32/Ny) gives b=
-0.004278 ±
0.004801.
This diagnostic fit leaves b unconstrained; a negative central b is not a
physical negative gap. Zero is compatible, but finite positive intercepts
are not excluded by these data. No thermodynamic extrapolation is adopted.

| Power-law fit window | z ± propagated fit SEM |
|---:|---:|
| 20–48 | 1.17 ± 0.21 |
| 24–48 | 1.18 ± 0.33 |
| 28–48 | 1.20 ± 0.41 |
| 32–48 | 1.65 ± 0.60 |

## Scope and cautions

This is a fixed-T=40, fixed-Nx=20 finite-time trend. Only endpoint data were
saved, so time convergence cannot be established. Increasing Ny does not
increase wall separation here. This is not a two-dimensional thermodynamic
limit or a demonstrated infinite-time Lyapunov gap closure.

Campaign 25 used square Nx=Ny systems at T=10 with independent samples.
Those data are neither pooled nor used in this fit. Differences from that
pilot confound observation time with geometry except at Nx=Ny=20. This
result also does not prove that the earlier variable-T scaling was artificial.

All 80 gaps are finite. Largest occupation-bound excess:
4.75e-13; largest saved covariance Hermiticity
residual: 0. Occupations selecting the
minimum range from 0.00333 to
0.97531, far from the 1e-9 pure-mode caps.
Every saved modular energy, cap, rate and gap was independently recomputed.

## Outputs

- `gap_vs_Ny_powerlaw.pdf/png`: primary linear-axis plot with mean ± SEM,
  exploratory power law, and fixed inverse-size fit. No extrapolation.
- `gap_vs_Ny_loglog.pdf/png`: the same fits and sampling errors on log axes.
- `gap_with_individual_samples.pdf/png`: all 80 individual gaps; horizontal
  offsets only separate points and are not physical size changes.
- `lyapunov_gap_vs_Ny.pdf/png`, `modular_gap_vs_Ny.pdf/png`: data-only plots.
- `gap_summary.csv`, `sample_gaps.csv`, `sample_spectral_diagnostics.csv`,
  `fit_window_sensitivity.csv`: means, SEMs, samples and checks.
- `size_scaling_manifest.json`: fits, covariances, definitions, hashes, provenance.

Reproduce with `python analyze_size_scaling.py --output-root <downloaded campaign>
--analysis-root <analysis_outputs>`. The original data, Drive deployment and
manuscript remain unchanged.

Suggested caption: Finite-time Lyapunov half gap versus circumference at fixed
Nx=20 and T=40 physical cycles. Hard-wall, slab-only adaptive dynamics begin
from a maximally mixed state with the canonical Born-conditioned exterior.
Points are the mean of ten trajectory-wise minimum absolute modular energies
divided by 2T; bars are ordinary trajectory SEM. The dashed curve fits
A(Ny/32)^(-z) over Ny=20–48 using inverse-SEM-squared weights, with
z=1.17±0.21; the dotted curve is the fixed 1/Ny fit.
The fit is descriptive of this finite-time window, not an infinite-time
extrapolation.
