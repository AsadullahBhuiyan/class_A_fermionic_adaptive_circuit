# Gaussian reference-ancilla anisotropy CPU pilot

Canonical dynamics: `classA_U1FGTN.run_markov_circuit`.

| L | I_space | t* | alpha | bootstrap resolved | alpha 95% interval |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.3383 | 1.448 | 1.163 | 0.761 | [0.3691, 1.564] |
| 8 | 0.1721 | 2.372 | 0.9463 | 0.605 | [0.7482, 1.221] |

Plateau values average the final `L` cycles after following both references for `2L` cycles.
Bootstrap resampling uses matched complete-trajectory indices across spatial and temporal configurations.
These small sizes and four trajectories are a feasibility pilot, not a production uncertainty estimate.

Total wall time: 455.9 s.
