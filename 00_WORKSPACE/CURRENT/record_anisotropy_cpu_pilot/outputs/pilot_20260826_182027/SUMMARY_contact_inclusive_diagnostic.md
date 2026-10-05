# Record anisotropy CPU pilot results

Canonical dynamics: `classA_U1FGTN.run_markov_circuit`.

This estimates the Born-record anisotropy, not the prepared-state velocity.

| observable | L | t* | alpha | cycles/L=1/alpha | bootstrap resolved |
|---|---:|---:|---:|---:|---:|
| surprisal | 6 | unresolved | unresolved | unresolved | 0.328 |
| mismatch_fraction | 6 | unresolved | unresolved | unresolved | 0.304 |
| surprisal | 8 | 0.9857 | 2.277 | 0.4392 | 0.662 |
| mismatch_fraction | 8 | 0.9973 | 2.25 | 0.4444 | 0.567 |
| surprisal | 10 | 0.9635 | 2.912 | 0.3434 | 0.700 |
| mismatch_fraction | 10 | 0.9697 | 2.893 | 0.3457 | 0.742 |

Matching rule: `C(L/2,0)=C(0,t*)`, with linear interpolation of the first raw temporal crossing.
Bootstrap resampling is by complete trajectory; the two walls remain paired.
An unresolved entry means the positive spatial target had no crossing in the preregistered lag window.

Total wall time: 248.0 s.
