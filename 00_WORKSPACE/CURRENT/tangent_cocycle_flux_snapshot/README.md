# Flux-dependent full cocycle snapshot

This experiment samples one 96-cycle Born trajectory at zero flux, replays its
48-cycle burn-in and 48-cycle observation record at 21 twists, and saves the
full scale-separated tangent product.  Every physical trajectory calls the
canonical CPU entry point `classA_U1FGTN.run_markov_circuit`.

The production configuration is in `reference_config.json`.  Each twist file
contains `Q`, `core_hat`, `core_log_scale`, `product_hat=Q@core_hat`, the final
covariance, singular-value/rank diagnostics, and JSON metadata.  The raw factor
`exp(core_log_scale)` is deliberately not formed.

Run a small validation with:

```bash
python tangent_cocycle_flux_snapshot/run_flux_cocycle_snapshot.py --smoke
```

`launch_tmux.sh` is the prepared production launcher.  It does not queue or
start itself; invoke it only after the existing CPU campaign releases the
selected cores.  Set `CPU_LIST` and `THREAD_COUNT` to override its defaults.
The launcher refuses to overlap known CPU-heavy repository campaigns unless
`ALLOW_BUSY=1` is explicitly set.
The runner checkpoints after each twist and accepts `--resume RUN_DIRECTORY`.
