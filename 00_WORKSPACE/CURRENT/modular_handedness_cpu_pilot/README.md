# S100 CPU modular-handedness campaign

This is a local, completion-resumable CPU campaign. It runs 100 independent
trajectories for each of two domain-wall constructions at `Nx=20`, `Ny=40`,
`nshell=1`, then reduces every final occupied frame with the legacy translated
half-cylinder modular-packet estimator.

The locked production command is:

```bash
python launch_tmux.py
```

Useful commands:

```bash
python launch_tmux.py --dry-run
python run_campaign.py status
python run_campaign.py all --resume --workers 28
python run_campaign.py smoke
tmux attach -t modular_handedness_s100
```

## Scientific contract

- Hard wall: support truncation and slab-only dynamics.
- Soft wall: untruncated explicit interface and full-system dynamics.
- Both: `DW=True`, `alpha_1=1`, `alpha_2=30`, pure half-filled random initial
  states, legacy `raster_y` ordering, perfect correction, complex128, and 80
  physical cycles through the canonical CPU `run_markov_circuit` entry point.
- Only the cycle-80 occupied frame is retained. No covariance history or cycle
  checkpoint is written.
- Each simulation and each packet reduction has its own NPZ/completion-JSON
  pair. A restart checks configuration identity, source hashes, bytes, and
  SHA-256 before skipping a task.

The packet reduction reconstructs each restricted correlation matrix directly
from the saved frame, converts it to the legacy centered covariance
`G_A = 2 C_A - I`, and analyzes all 40 translated width-20 cuts independently.
It never averages covariances or modular Hamiltonians before propagation.

## Runtime and storage

Measured one-cycle host timings were about 35 s (hard) and 41 s (soft). With 28
one-thread workers on one NUMA node, the expected wall time is roughly 8--10
hours. A killed worker loses at most one approximately 47--55 minute trajectory.
Verified trajectories and completed analysis products are retained.

The 200 uncompressed occupied frames require approximately 3.8 GiB. Compact
packet products and summaries keep the expected campaign total near 5 GiB.
All generated files live below the stable ignored `outputs/` directory.

