# Production amendment: 10-sample fixed-geometry collaborator campaign

This amendment supersedes only the active `production_25sample_v2_lean` execution
matrix. It does not alter or reinterpret any archived result. The original
25-trajectory P1 campaign remains governed by its frozen `production_25sample_v1`
configuration and is completed only through `01_p1_existing_completion`.

## Locked stochastic contract

- Sampling revision: `production_10sample_v3_fixed_nx20_ny20_30_40_50_60`.
- Ordinary stochastic cases contain 10 independent trajectories in two immutable
  five-trajectory shards.
- The transverse width is fixed to `Nx=20`.
- Scaling families use `Ny=[20,30,40,50,60]`.
- Every physical trajectory runs for exactly `2*Ny` cycles, with zero burn-in,
  independent random schedules, perfect correction unless a control explicitly varies
  it, and `complex128` arithmetic through
  `classA_U1FGTN_gpu.run_markov_circuit`.
- H3 remains a representative one-record deterministic descendant. Validation and
  deterministic mean-channel calculations do not acquire a fictitious trajectory axis.

## Matrix reduction

Dense parameter scans remain at `Ny=40`. Their finite-size arms use `Ny=20,30,50,60`,
with the `Ny=40` point supplied by the dense scan. The M3 dense noise scan remains at
`20x20`; its sparse size trend is rectangular at `20x30`, `20x40`, `20x50`, and
`20x60`. H3 and B1 remain representative `20x40` calculations.

The former transverse-width gate is retired. Bundle 01 becomes a non-gating fixed-width
baseline containing the explicit-interface, matched-trivial, support-terminated, and
support-terminated matched-trivial constructions at all five circumferences. Downstream
bundles obtain `Nx=20` from this immutable contract and do not consume
`accepted_width.json`.

## Provenance and claim boundary

The revised campaign writes to
`classA_final_production_outputs/production_10sample_v3_fixed_nx20_ny20_30_40_50_60`.
Earlier output trees are read-only provenance. A prior five-trajectory shard may be
reused only when its engine hash, complete case configuration apart from the declared
ensemble size, root and shard seeds, shard index, and global sample indices match
exactly. Ten-trajectory estimates carry larger ensemble uncertainty than the retired
25-trajectory matrix and must report trajectory-level uncertainty without silently
pooling incompatible protocols.
