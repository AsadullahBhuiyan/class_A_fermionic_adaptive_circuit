# Gaussian reference-ancilla anisotropy CPU pilot

**Outcome: the exact Gaussian reference probe is feasible, but four trajectories do not calibrate alpha.**

| L | I_space | t* | alpha | bootstrap resolved | estimated S for 20% spatial SEM |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.3383 | 1.448 | 1.163 | 0.756 | 57 |
| 8 | 0.1721 | unresolved | unresolved | 0.266 | 85 |

The `L=6` crossing is a candidate only: its 95% bootstrap alpha interval is broad and the final plateau window still shifts by about one standard error.
The `L=8` mean temporal curve starts below the spatial target at `delta_tau=1`, rises at `delta_tau=2` because of one rare trajectory, and then falls. The ordered CFT crossing rule therefore marks it unresolved.

Plateau values average the final `L` cycles of a `2L` post-insertion follow. Complete trajectories are the bootstrap units.
The raw reference entropies and mutual information are retained for every sample and every follow cycle.
The earlier permissive interpolation is preserved only as a diagnostic artifact and is not an accepted alpha estimate.

Canonical dynamics wall time: 455.9 s.
