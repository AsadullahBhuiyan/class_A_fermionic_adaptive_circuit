# Gaussian reference-ancilla anisotropy CPU pilot

This is the direct charge-conserving Gaussian analogue of the reference-qubit
protocol in Zabalo *et al.* It is separate from the unsuccessful local-record-field
pilot.

The active target protocol is support terminated: `dw_truncation=True` and
`meas_slab_only=True`. The completed `pilot_20260826_183458` run used the earlier
untruncated `False/False` geometry and is retained only as an engineering validation
of the reference construction; its candidate anisotropy does not calibrate the
support-terminated campaign.

At cycle `tau1=4L`, both canonical fermion modes of one wall unit cell are measured.
For each mode, a spectator reference fermion is prepared with the complementary
occupation and mixed with the physical mode by a 50:50 number-conserving beam
splitter. The resulting two physical/reference Bell pairs form a maximally entangled
four-level cell/reference state while preserving Gaussianity and total charge. A
second two-mode reference is inserted either `L/2` cells away at the same cycle or at
the same cell after `delta_tau` cycles. The canonical Markov engine then continues,
and the exact Gaussian mutual information between the two reference subsystems is
retained after every subsequent cycle.

The pilot treats a complete circuit trajectory as the independent sample and keeps
the two reference modes in each subsystem together. It estimates plateau mutual
informations, matches the temporal curve to the spatial value, and evaluates

```text
alpha = log(1 + sqrt(2)) L / (pi t_star).
```

The live-state mutation used here is intentionally confined to this CPU pilot. A
production campaign should promote reference insertion to an explicit canonical
engine hook after the observable and plateau protocol pass validation.

The matching implementation requires an ordered downward crossing: the shortest
sampled temporal separation must lie above the spatial target. This prevents a rare
upward fluctuation at a later separation from manufacturing a crossing.

Cycle count is a measured convergence property in the target geometry. The initial
budget is `4L` pre-insertion equilibration, temporal separations through `0.75L`, and
`2L` post-insertion follow, or at most `6.75L` (rounded to `7L`) per trajectory. This
budget is accepted only when matched-seed audits show (i) the reference correlators
are unchanged upon increasing the insertion time and (ii) consecutive `L`-cycle
post-insertion window means agree within their combined trajectory uncertainty.

Run:

```bash
pytest -q 00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/tests
python 00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/run_cpu_pilot.py
```

## Locked L=24 CPU campaign

The failed v1 run is preserved as an engineering diagnostic. It stopped in a branch
worker because a freshly rebuilt worker model failed the engine's strict checkpoint
signature guard. The v3 runner uses fork-inherited deep copies of one canonical model
template, never reconstructs that template between stages, reports field-level
signature differences, validates 20 simultaneous L=24
one-cycle resumes in preflight, and reuses only the independently validated v1
benchmark and 32 burn-in checkpoints—not its partial reference branches.

The production-calibration runner is separate from the small engineering pilot. It
uses the immutable settings in `campaign_config.l24.v3.json`, performs an `S=32`
matched-seed cycle audit, and starts an independent `S=100` measurement only after
the audit passes. Both `dw_truncation=True` and `meas_slab_only=True` are hard
validation conditions. The coarse temporal grid is `1,3,6,9,12,15,18`; only the
integer cycles inside the measured crossing bracket are added.

Launch the complete preflight, CPU benchmark, audit, final measurement, analysis,
and validation chain in a detached session with

```bash
python 00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/launch_tmux.py
```

The launcher samples per-core utilization, reserves 28 distinct physical cores on
one NUMA node, sets all BLAS thread counts to one, and prints exact attach, log, and
resume commands. The runner can also be invoked stage by stage:

```bash
python 00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/run_l24_campaign.py \
  all --run-root /absolute/output/path --cpu-list 0-27 --workers auto --resume
```

Here (t_*) is the insertion-time separation at which the arithmetic Born mean of
the temporal reference mutual information matches the spatial result at separation
`L/2=12`. For `L=24`, `alpha = 6.7332/t_star`; neither `t_star` nor the anisotropy is
identified with the equilibration or post-insertion follow duration.

The complete derivation, protocol, validation, pilot analysis, and next-run
recommendation are in
[`docs/reference_ancilla_anisotropy_pilot_note.pdf`](docs/reference_ancilla_anisotropy_pilot_note.pdf),
with the editable LaTeX source beside it.
