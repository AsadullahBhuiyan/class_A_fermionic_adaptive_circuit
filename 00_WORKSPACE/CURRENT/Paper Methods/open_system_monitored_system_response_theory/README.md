# Open-system / monitored-system response theory

This directory is the canonical home of the standalone methods note that
consolidates the project's discussion of record-conditioned response,
trajectory averaging, Born-score reweighting, density-response chirality,
momentum--frequency spectroscopy, frozen-record flux flow, and possible
quantized-response extensions.  Its Hall proposal is numerical and
microscopic: exact deterministic Born-mean evolution first, ordinary sampled
Born trajectories second, and a local gate compilation for literal electrical
current.  Generating-functional material is optional context, not the proposed
calculation.

The note is a theory and methods source of truth, not a completed-result claim.
In particular:

- the exact normalized covariance tangent and Born-score identities are analytic;
- the current H2 production code still evaluates those derivatives with batched
  central `+epsilon/-epsilon` frozen-record replays at `epsilon=1e-3`;
- H2 production uses 25 independent Born trajectories, while H3 uses one
  preregistered interface record and one matched-trivial record;
- a wall-resolved density-response ridge diagnoses chirality but is not called a
  quantized Hall coefficient.
- a literal electrical Hall coefficient requires a gauge-covariant local
  system--ancilla dilation because the reduced physical circuit exchanges
  particles with feedback ancillas;
- perfect correction makes every linear Born-mean covariance observable close
  exactly under `C_next = Q C Q + s P`, so intrinsic outcome sampling is not
  needed for the first mean-response screen;
- the strongest realistic record-independence target is concentration of the
  pumped-charge or response distribution at an integer for Born-typical
  records, not equality for every finite record;
- the proposed flux-ramp/FCS campaign is a future, versioned experiment; H3's
  current one-record static twist replay is not that physical pump.

The short execution-facing protocol is
[`MICROSCOPIC_NUMERICAL_HALL_PROTOCOL.md`](MICROSCOPIC_NUMERICAL_HALL_PROTOCOL.md).

The standalone implementation handoff that combines the exact
flattened-Hamiltonian wall-pump benchmark, the source-subtracted monitored
circuit observable, the required GPU API, validation gates, staged campaign,
and related literature is
[`wall_resolved_flux_pump_handoff.tex`](wall_resolved_flux_pump_handoff.tex).
Its compiled PDF is
[`wall_resolved_flux_pump_handoff.pdf`](wall_resolved_flux_pump_handoff.pdf).

The current manuscript contains compact versions of parts of this material in
`../../prxq_draft/main.tex` and `../../prxq_draft/appendices.tex`. The audited
campaign contract remains
`../../experiment_review/numerical_campaign_legacy_working.tex`; the older
`../../prxq_draft/numerical_campaign.tex` contains superseded H2/H3 sample counts.

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build open_system_monitored_system_response_theory.tex
```

The canonical PDF is `build/open_system_monitored_system_response_theory.pdf`.
