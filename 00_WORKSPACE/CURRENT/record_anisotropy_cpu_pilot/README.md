# Born-record spacetime-anisotropy CPU pilot

This isolated experiment calibrates the anisotropy of the wall **Born record**. It
does not claim to measure the propagation velocity of the prepared Gaussian state.
Every trajectory calls the canonical
`classA_U1FGTN.run_markov_circuit(...)` CPU entry point with the explicit-interface
production geometry (`Nx=20`, `DW=True`, `nshell=1`, `alpha_1=1`, `alpha_2=30`,
random schedule, perfect correction, and no slab truncation).

The pilot retains, for every sample/cycle/wall/y/channel, the occupation probability,
realized log probability, outcome bit, and mismatch bit. Two preregistered local
record fields are derived without post-selection:

- site surprisal, `-sum_a log p(realized channel a)` (primary);
- mismatch fraction relative to the four OW target occupations (diagnostic).

For each field it measures the same connected correlator in space and time and solves

```text
C(L/2, 0) = C(0, t*)
alpha(L) = asinh(1) L / (pi t*) .
```

The two walls are averaged within a trajectory. Bootstrap resampling uses whole
trajectories, so neither walls nor time origins are treated as independent samples.
The default pilot uses `L=6,8,10`, four trajectories per size, `2L` burn-in, `6L`
recording, and temporal lags through `3L`. These are exploratory settings, not a
production error bar. Lag zero is excluded from interpolation because it contains
the local contact/variance term; an absent positive non-contact crossing is reported
as unresolved.

Run the unit tests and pilot with:

```bash
pytest -q 00_WORKSPACE/CURRENT/record_anisotropy_cpu_pilot/tests
python 00_WORKSPACE/CURRENT/record_anisotropy_cpu_pilot/run_cpu_pilot.py
```

Each output directory contains an immutable manifest, lossless raw NPZ files,
per-observable correlation/bootstrap NPZ files, JSON and Markdown summaries, and a
300-dpi PDF/PNG diagnostic figure.
