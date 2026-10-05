# OW constraint-flag breakdown CPU pilot

This is a standalone, resumable CPU experiment. It does not modify the PRX Quantum
draft, existing working notes, or GPU/Colab packages. All trajectories call the
canonical `classA_U1FGTN.run_markov_circuit(...)` entry point.

Run a cheap validation:

```bash
python 00_WORKSPACE/CURRENT/ow_constraint_flag_pilot/run_pilot.py smoke --output-root /tmp/ow_flag_smoke
```

Launch the preregistered `16 x 20`, 10-seed pilot after the tests pass:

```bash
python 00_WORKSPACE/CURRENT/ow_constraint_flag_pilot/launch_tmux.py
```

The launcher chooses ten lightly loaded physical cores from CPU IDs 0--55, records
the preflight, and prints the detached tmux session name and log path. Run directories
are timestamped. A completed stage has a `SUCCESS` marker and can be skipped with
`--resume`; incomplete work is retained under a `.partial` directory for diagnosis.

The companion formalism and preregistration are in `docs/constraint_flag_pilot.tex`.
Its generated `docs/generated/results_fragment.tex` says “pilot pending” until the
analysis stage writes measured results.
