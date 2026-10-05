# 07 — Ambient Log-Gram Alpha Scan

This standalone Colab bundle implements the approved 220-case scan of the
ambient fixed-record covariance Jacobian. It does not modify or depend on the
locked Gate 4 bundle.

The saved checkpoint operator is

\[
H_t=\log(J_{1:t}^{\dagger}J_{1:t}),
\]

and each sample retains the 16 finite eigenpairs with smallest \(|h|\). No
Choi state or Choi-derived transfer matrix is constructed.

## Two exact extraction paths

- **Maxmix:** at a checkpoint, the live active covariance gives
  \(J^{\dagger}J=4C_t(\mathbf{1}-C_t)=\mathbf{1}-G_t^2\). The covariance is
  diagonalized in memory and then discarded by the canonical runner.
- **Pure:** the canonical runner's `lyapunov_basis_mode="pure_occupied_empty"`
  initializes complete occupied and empty bases after cycle-zero preparation,
  propagates both with the existing adjoint half-Jacobian updates, and applies
  independent QR stabilization after every cycle. Exact cumulative cores are
  retained in memory only until each checkpoint eigensystem is extracted by a
  direct SVD. Scalar log singular values are merged and ranked before only the
  selected 16 ambient vectors are embedded or transferred from the GPU.

For the hard protocol, Born-conditioned exterior preparation occurs before the
two tangent blocks are initialized. Its outcomes are saved separately for exact
replay, but its Jacobians are not part of \(J_{1:t}\). Saved hard-arm vectors are
expressed in the declared active-slab basis.

## Contract

- `Nx=20`; `Ny=[20,30,40,50,60]`
- `alpha_1=1.0,1.2,...,3.0`; `alpha_2=30`
- hard `(meas_slab_only,dw_truncation)=(true,true)`
- soft `(false,false)` with `DW=true`
- maxmix and independent Haar-random half-filled pure Slater arms
- 25 Born trajectories per cell; five immutable shards of five
- `nshell=1`, random site order, perfect correction, no postselection
- `2*Ny` cycles in `complex128`
- checkpoints `round(linspace(Ny,2*Ny,6))`
- root seed `2026082607`
- finite singular-value tolerance `1e-12`
- roundoff-degenerate cluster tolerance `64*eps(float64)`
- eigenpair-residual and vector-Gram validation ceilings `1e-8`

`production_queue.json` contains exactly 1,100 deterministic shard jobs. An
existing shard is skipped only after its SHA-256 checksum and complete
identity (revision, source hashes, case, shard, sample count, and active basis)
verify. Orphan or mismatched artifacts fail instead of being overwritten. A
case-specific conservative memory admission check runs against current A100
free memory before frames or tangent cores are allocated.

## Commands

```bash
python run_bundle.py preflight
python run_bundle.py queue
python run_bundle.py pilot --output-root /content/drive/MyDrive/classA_pilot_outputs/ambient_log_gram_alpha_scan_v1 --replay-check
python run_bundle.py production --case-id LG_N20x20_a1.0_hard_pure --shard-index 0 --output-root /content/drive/MyDrive/classA_log_gram_alpha_scan/ambient_log_gram_alpha_scan_v1
```

Omitting `--case-id` and `--shard-index` walks the full resumable production
queue. Pure trajectories always use batch size 1; maxmix uses up to five.

The Colab notebook enforces the A100 runtime check on startup. Both `RUN_PILOT` and
`RUN_PRODUCTION` default to `False`, so opening or running setup cells cannot launch a
campaign accidentally.

## Output schema

Every `.npz` shard contains:

- `log_gram_eigenvalues`: `(samples,6,16)`, `float64`
- `log_gram_eigenvectors`: `(samples,6,active_dimension,16)`, `complex128`
- occupied/empty/mixed block and degenerate-cluster labels (-1=mixed,
  0=occupied, 1=empty)
- residuals, vector Gram errors, finite ranks, null counts, and boundary gaps
- checkpoint cycles and full-space indices defining the active basis
- packed cycle and exterior-preparation outcomes, site-order records, initial
  state seeds, trajectory-stream seed, and globally unique sample IDs

The adjacent manifest records complete parameters, source hashes, runtime,
output checksum, replay status, and `choi_tracked=false`. Tangent products,
covariances, Choi blocks, entropy, topology, correlations, and other state
observables are intentionally absent.
