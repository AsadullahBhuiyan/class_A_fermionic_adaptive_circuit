# Fresh-record twist-torus validation

This CPU campaign tests whether occupied frames produced from independently sampled measurement records can be stitched into a well-defined many-body Berry bundle. It uses a 16×16 physical lattice, a 16×16 twist torus, 32 cycles, and the canonical `classA_U1FGTN.run_markov_circuit` occupied-frame backend.

Every twist point constructs a new model and rebuilds its twist-dependent OW orbitals. No full trajectory record is replayed. The two surfaces differ only in whether the random site schedule is also refreshed.

## Launch

```bash
cd /home/abhuiyan/class_A_fermionic_adaptive_circuit
00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/launch_tmux.sh --dry-run
00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/launch_tmux.sh --preflight-only
00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/launch_tmux.sh
```

Resume a named campaign with:

```bash
CAMPAIGN_ID=N16_T16_<timestamp> \
  00_WORKSPACE/CURRENT/fresh_record_twist_torus_validation/launch_tmux.sh --resume
```

The default runner binds 56 single-threaded workers to distinct physical cores. Override the exact logical CPUs with `CPU_LIST=0,1,...` while retaining exactly 56 entries.

## Products

Each twist shard contains complete cycle-resolved marker, entropy, charge, rank, Gram residual, and log-weight arrays, its final occupied frame, and a compact site/outcome/probability record. Analysis writes aggregate NPZ and CSV products, PDF/PNG maps, a machine-readable classification, and `reports/validation_note.md`.

Rank mismatch or a singular neighboring overlap is reported as an undefined stochastic invariant. It is not repaired with padding, truncation, a pseudoinverse, or integer rounding.
