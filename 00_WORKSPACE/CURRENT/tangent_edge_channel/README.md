# Physical tangent channel at a monitored domain wall

This directory contains the CPU reference campaign for the proposed
steady-ensemble, two-mode edge-channel diagnostic. It asks a deliberately
narrow question: after the physical monitored trajectory reaches its steady
ensemble, how does one fixed realized measurement-and-correction record act on
a predetermined physical chiral edge subspace?

The two input vectors are adjacent Bloch modes of the exact real-space
Dirac/Chern domain-wall Hamiltonian, localized on one of the two periodic domain
walls. Near the finite-cylinder avoided crossing, the constructor resolves the
two-state edge doublet by diagonalizing a left-minus-right wall discriminator.
It then projects the physical modes into the active measurement basis only when
hard truncation makes that unavoidable, reorthonormalizes them, and records the
retained norm. The input plane is therefore fixed before any Lyapunov data are
observed; it is not selected from a slow-vector spectrum after the fact.

For a realized record \(r\), the circuit propagates the fixed-record state
tangent \(D\Phi_r\). The Born draw is held fixed and is not differentiated. At
the end of every observed cycle, a positive-diagonal QR gauge gives

\[
    D\Phi_{r,t} V_0 = Q_t B_t, \qquad B_t = R_t B_{t-1}, \quad B_0=I_2.
\]

Only the final large frame \(Q_t\) is retained. The time series stores the
two-by-two \(R_t\) factors, a Frobenius-normalized restricted core and its log
scale, singular values/rank, wall retention and opposite-wall leakage,
interference phase displacement, branch-probability audits, and predictable
record Fisher information. Exact rank loss is reported as rank loss/NaN rather
than hidden behind a pseudoinverse or a finite transfer-core regularizer.

## Reference campaign

[`reference_config.json`](reference_config.json) specifies the intended
production grid. The primary run is the \(\alpha_1=1\), \(\alpha_2=30\), hard
domain-wall protocol with perfect correction and Born-sampled outcomes. Its
matched hard-wall controls keep the same active cells and four measurement
channels per visited cell:

- `same_chern_hard_cut`: \(\alpha_1=\alpha_2=1\), isolating the hard cut from a
  topological contrast;
- `trivial_hard_wall`: \(\alpha_1=\alpha_2=30\), removing the topological
  sector;
- `domain_wall_raster_order`: the primary geometry with a deterministic raster
  order, testing ordering bias.

`uniform_topological` is a full-lattice baseline, not an event-count-matched
control. Every output records the number of measurement channels per cycle and
also reports the survival exponent per visited channel. The \(n_{\rm shell}=2\)
case is a cutoff check. Postselection is restricted to a small deterministic
validation case and is not the primary physical protocol.

The default burn-in is \(2N_y\) physical cycles and the tangent observation
window is \(4N_y\) cycles. Burn-in and observation occur in one uninterrupted
call to the canonical CPU entry point, with
`lyapunov_start_cycle=burn_cycles+1`; restarting a covariance at the observation
boundary would not sample the same trajectory ensemble.

Run the compact live smoke campaign with:

```bash
python tangent_edge_channel/run_tangent_edge_channel.py \
  --smoke --no-parallel --output-dir /tmp/tangent_edge_channel_smoke
```

Run one selected case or a reduced pilot with:

```bash
python tangent_edge_channel/run_tangent_edge_channel.py \
  --cases domain_wall_primary --ny 24 --samples 8 \
  --burn-cycles 24 --observation-cycles 48 \
  --output-dir tangent_edge_channel/results/pilot_001
```

Run the reference grid with externally parallel single-trajectory workers:

```bash
python tangent_edge_channel/run_tangent_edge_channel.py \
  --cpu-budget 80 --output-dir tangent_edge_channel/results/campaign_001
```

Each run directory contains compressed arrays, per-sample and per-cycle metric
tables, publication-sized diagnostic figures, and a `run_summary.json` with the
canonical entry point, exact edge-frame metadata, seeds, resource decision,
failure/censoring records, and git state. `campaign_summary.csv` compares final
metrics across sizes and controls. The driver computes these quantities but
makes no claim in advance about chirality, protection, or error-correcting
performance; those are conclusions to draw only from converged data and the
matched controls.
