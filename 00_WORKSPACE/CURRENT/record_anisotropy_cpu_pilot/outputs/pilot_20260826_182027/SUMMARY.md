# Record anisotropy CPU pilot results

**Outcome: no infrared spacetime-anisotropy crossing was resolved.**

The lag-zero local variance/contact term is excluded. At every size, the nonzero-lag temporal and `L/2` spatial correlations are noise-scale and do not form a positive monotone matching curve.

| observable | L | C(L/2,0) | C(0,1) | t* | alpha | bootstrap resolved |
|---|---:|---:|---:|---:|---:|---:|
| surprisal | 6 | -0.02496 | -0.02023 | unresolved | unresolved | 0.069 |
| mismatch_fraction | 6 | -0.000287 | -0.0004048 | unresolved | unresolved | 0.000 |
| surprisal | 8 | 0.02254 | 0.0004336 | unresolved | unresolved | 0.110 |
| mismatch_fraction | 8 | 8.338e-05 | 3.49e-05 | unresolved | unresolved | 0.023 |
| surprisal | 10 | 0.03607 | -0.01785 | unresolved | unresolved | 0.019 |
| mismatch_fraction | 10 | 0.0003826 | -0.0001545 | unresolved | unresolved | 0.018 |

Matching rule: `C(L/2,0)=C(0,t*)`, with interpolation beginning at lag 1. Lag 0 is not a time-separated correlator.
Bootstrap resampling is by complete trajectory; the two walls remain paired.
Stationary means and variances pass the first-half/second-half diagnostic, so the failure is lack of an infrared signal rather than visible burn-in drift.

The previously contact-inclusive interpolation is preserved only as `results_contact_inclusive_diagnostic.json`; its sub-cycle crossings are not physical estimates.

Canonical dynamics wall time: 248.0 s.
