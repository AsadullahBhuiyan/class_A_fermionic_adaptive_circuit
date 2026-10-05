# Campaign 25: fixed-time square-system gap analysis

Analyzed 2026-09-28. All 14 result/completion pairs were revalidated against
their byte counts, SHA-256 checksums, scientific configuration, task IDs, seeds,
and source identities. Every size contains samples 0–9 exactly once.

## Estimator and results

For each independent Born trajectory, at exactly ten physical cycles,

\[
g_{\mathrm{mod},\xi}=\min_j\left|\log\frac{1-\nu_{\xi,j}}{\nu_{\xi,j}}\right|,
\qquad \Delta_\xi=g_{\mathrm{mod},\xi}/20.
\]

The minimum is taken **before** trajectory averaging. This is not the gap of
an ensemble-averaged spectrum or covariance. Errors below are ordinary sample
SD/sqrt(10), with no bootstrap or fit-residual error. Occupations come from the
active slab; this is not a half-system entanglement spectrum.

| L=Nx=Ny | Mean raw modular gap ± SEM | Mean finite-time half gap ± SEM |
|---:|---:|---:|
| 20 | 0.4162 ± 0.1259 | 0.02081 ± 0.00629 |
| 24 | 0.4846 ± 0.1164 | 0.02423 ± 0.00582 |
| 28 | 0.3278 ± 0.1069 | 0.01639 ± 0.00534 |
| 32 | 0.4260 ± 0.1142 | 0.02130 ± 0.00571 |
| 36 | 0.2352 ± 0.0625 | 0.01176 ± 0.00313 |
| 40 | 0.2716 ± 0.0584 | 0.01358 ± 0.00292 |
| 44 | 0.2749 ± 0.0597 | 0.01375 ± 0.00298 |

## Interpretation

The means suggest a downward tendency, but it is non-monotonic and the scatter
is large. The endpoint difference is
`mean Delta(L20) - mean Delta(L44) = 0.0070635 ± 0.0069656`, combining the
independent size SEMs in quadrature. This is about one combined SEM. The central
values decrease by 34%, but these ten-sample ensembles do not establish a
size exponent or gap closure. The largest three means are mutually close;
this does not establish a nonzero thermodynamic limiting gap either.

The divisor `2T=20` is identical for all sizes, so any size trend in this pilot
is already present in the raw modular gap. The raw and normalized plots contain
the same size information. This removes the explicit size-dependent denominator
of the earlier `T=2Ny` endpoint study, but it does not demonstrate that the old
trend was entirely a normalization artifact. The old sweep kept Nx=20 and used
longer size-dependent times; this sweep also changes Nx and wall separation.

Only one endpoint time was retained, so convergence of the finite-time rate
cannot be tested from this ensemble. Ten cycles is a chosen observation time,
not an established asymptotic regime. A future fixed-size time comparison is
needed before interpreting these numbers as infinite-time Lyapunov gaps.
No power-law fit or thermodynamic extrapolation has been imposed.

## Numerical checks

- All 70 gaps are finite; saved spectra and gaps match recomputation.
- Largest occupation-bound excess: `4.22e-15`; saved maximum Hermiticity
  residual: zero.
- Occupations selecting the minimum gap lie between `0.2342` and `0.7618`,
  far from the pure-mode caps at `1e-9` and `1-1e-9`. Those caps are not
  setting the reported gaps; covariance clipping was disabled.
- Typical sample scatter is much larger than numerical roundoff. For example,
  at L=20 the individual half gaps span `0.00103..0.05923`; at L=44 they span
  `0.000548..0.02801`.

## Files

- `lyapunov_gap_vs_L.pdf/png`: mean finite-time half gap with ordinary SEM.
- `modular_gap_vs_L.pdf/png`: the same result without division by 20.
- `gap_summary.csv`, `sample_gaps.csv`: all means, SEMs and 70 sample gaps.
- `analysis_manifest.json`: input/output hashes and estimator definition.
- `diagnostics/gap_with_individual_samples.pdf/png`: individual trajectories
  in gray, with mean ± SEM in blue. Horizontal offsets separate overlapping
  samples; dashed segments are guides, not fitted curves.
- `diagnostics/sample_spectral_diagnostics.csv` and `spectral_diagnostics.json`:
  numerical checks, sampling scatter, endpoint contrast and provenance.

The simulation data and existing manuscript figures remain unchanged.
