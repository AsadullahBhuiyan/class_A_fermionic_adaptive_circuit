# Frozen-record flux–charge CPU pilot

This local CPU pilot asks how the final charge distribution of one fixed monitored-circuit record changes when the overcomplete-Wannier (OW) functions are rebuilt at a static boundary twist. It is deliberately smaller than the online monitored-channel pump in the handoff document.

## Current recommended calculation: wall-diabatized width sweep

The current production target is
`N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1`.  It replaces the
accidental mesh-dependent branch choice of the older static-projector scan
with an explicit wall-character continuation of the isolated occupied--empty
edge doublet.  It is a spectral-flow diagnostic of saved pure-state endpoint
projectors, not a real-time monitored pump and not a continuation of the
full-rank adaptive Kato campaign.

As of September 9, 2026, endpoint acquisition is complete. The corrected A100
campaign passed its timing and memory gate, and all 230 five-trajectory shards
were imported and independently verified: 1,000 missing primary endpoints and
150 GPU bridge endpoints, in addition to the 600 immutable legacy CPU
endpoints. The first exact-control launch was stopped because it twisted the
zero-flux surrogate `I-2*P_exact(0)` instead of rebuilding the authoritative
exact equilibrium Hamiltonian at every flux. The corrected control now rebuilds
the OW functions and `H_exact(phi)`, then performs the legacy continuation
inside each conserved transverse-momentum block. A live eight-interval check
reproduced `q_x=(+1.000000,-0.99999999)`, instantaneous closure at `1e-8`, and
true-undo error below `3e-17`. The 16-cell production-shaped sample-0 smoke and
the corrected 12-control gate are running; the full S100 sweep remains locked
until both finish successfully. The primary matrix is S100 per soft/hard wall
for both `nshell=1` and dense correction at fixed `Ny=24` and
`Nx=20,24,28,32`.

Use these documents as the authoritative entry points:

- [`notebooks/equilibrium_wall_crossing_explorer/equilibrium_wall_crossing_explorer.ipynb`](notebooks/equilibrium_wall_crossing_explorer/equilibrium_wall_crossing_explorer.ipynb):
  executed, interactive single-state equilibrium demonstration of the avoided
  wall crossing, instantaneous refilling, and previous-overlap continuation;
- [`WALL_DIABATIC_WIDTH_SWEEP_RUNBOOK.md`](WALL_DIABATIC_WIDTH_SWEEP_RUNBOOK.md):
  operator checklist, exact launch order, source matrix, output locations,
  resume/failure behavior, and final acceptance ledger;
- [`docs/wall_diabatic_spectral_pump_methods.pdf`](docs/wall_diabatic_spectral_pump_methods.pdf):
  two-column scientific definition, formalism, controls, thresholds, and
  interpretive boundary, with editable REVTeX source beside it;
- [`WALL_DIABATIC_DATA_DICTIONARY.md`](WALL_DIABATIC_DATA_DICTIONARY.md):
  exact endpoint, checkpoint, spectral-pair, completion, control, sensitivity,
  and analysis fields, axes, schemas, and status vocabulary;
- [`../final_production_new_designs/11_wall_pump_width_endpoints/README.md`](../final_production_new_designs/11_wall_pump_width_endpoints/README.md):
  the self-contained A100 endpoint-bundle contract;
- `campaign_config.wall_diabatic_spectral_pump_s100_v1.json`: the
  machine-readable scientific contract; and
- `local_profile.wall_diabatic_spectral_pump_s100_v1.json`: the
  machine-readable CPU execution and resume contract.

One spectral task is one endpoint state and contains separate raw CCW and CW
paths.  The main calculation has 1,600 such primary pairs (3,200 directional
paths), with 150 optional bridge pairs kept outside the primary ensemble.  It
also preregisters 12 exact-control pairs and 2,050 sensitivity pairs.  A
resolved edge cluster is determined by fixed isolation, wall-polarization,
wall-weight, and link-quality thresholds; an unqualified cluster is saved with
`resolved=False` and its explicit `unresolved_reason`, and is never tuned until
it pumps. Quantization is therefore an outcome, not an acceptance gate.

## Finite-difference parent-Hamiltonian ramp

`N20x24_parent_schrodinger_rk4_s50_tau1e4_v1` tests actual unitary
Schrödinger evolution of saved monitored endpoint states, rather than replacing
their occupied projector at each flux.  It uses 25 soft and 25 hard endpoints
(sample IDs `0,4,...,96`, selected independently of their earlier pump result),
constructs `h0=1-2F0F0^dagger`, and integrates
`i dF/dt=h(phi(t))F` in complex128 with a classical finite-difference RK4 ramp
of duration `T=10000` in both directions.  The 128 observation intervals each
contain 320 RK4 steps, and every path has a rolling checkpoint.

```bash
./launch_parent_schrodinger_rk4_tmux.sh --dry-run
./launch_parent_schrodinger_rk4_tmux.sh
```

The session `parent_schrodinger_rk4_N20x24_s50_tau1e4_v1` uses physical cores
28--55 with 28 one-thread workers.  It writes one checksum-verified result pair
per CW/CCW path, resumes from stable frame checkpoints, and automatically
compares the endpoint response with the prior static projector continuation.
At finite circumference the strict infinitely slow limit can follow an avoided
crossing back to the instantaneous ground-state projector and close with zero
transfer; this campaign asks whether `T=10000` is already in that regime and
how the outcome depends on the saved crossing gap.

The `20x24` calculation is complete: all 100 raw CW/CCW paths verify.  The
soft-wall endpoint means are `+0.71569` (CCW) and `-0.71422` (CW); the hard-wall
means are `+0.53622` and `-0.53624`.  The distinct `24x24` physical-time width
repeat was stopped when the project was redirected to geometric continuation.
Its zero completions, 28 verified interval-104 rolling checkpoints, log, and
configuration remain untouched under
`results/N24x24_parent_schrodinger_rk4_s50_tau1e4_v1/`; the exact stop record is
`cancellation_inventory.json`.

The primary discretization has `dt=0.244140625`.  Its numerical acceptance is
checked independently with the four-path campaign
`N20x24_parent_schrodinger_rk4_s2_tau1e4_dt_half_v1`, which reruns sample 0 for
both walls and both directions with 640 steps per observation interval
(`dt=0.1220703125`).  It has a separate output root and cannot alter the active
S50 checkpoints.  After the primary sample-0 paths exist, run:

```bash
python run_parent_schrodinger_rk4_refinement.py report
taskset -c 28-31 python -u run_parent_schrodinger_rk4_refinement.py run --workers 4
```

The refinement writes `step_halving_summary.json`, including the maximum
pointwise and endpoint differences in `q_x`.  It is a discretization check, not
an additional statistical sample.  The lightweight tmux session
`parent_schrodinger_rk4_dt_half_followup` waits for the primary verified
`analysis_summary.json` and then performs this refinement on cores 0--3; if the
primary session ends without that marker, the follow-up exits visibly instead
of launching.

## Adaptive Kato overlap continuation

`N20x24_N24x24_kato_overlap_continuation_s25_v1` replaces the canceled width
repeat with a different question.  It performs no physical-time evolution.
For each saved endpoint, the twisted state-derived parent is diagonalized at
every adaptive RK4 stage; the fixed-rank spectral branch is chosen by maximum
overlap with the preceding projector, using energy only to break exact ties.
The occupied frame is transported geometrically by
`dF/dphi=[dP_spec/dphi,P_spec]F`.  One full step and two half steps are compared
through their projector Frobenius distance at tolerance `1e-8`.

The campaign uses endpoint IDs `0,4,...,96` for soft/hard walls at both
`20x24` and `24x24`, and runs CW/CCW separately: 200 paths total.  Each path
has one atomic NPZ/completion pair and one stable rolling frame checkpoint
every eight of the 128 flux intervals.  The launcher first runs all eight
sample-0 paths at `1e-8`, repeats them at `2.5e-9`, requires raw-`q_x`
agreement below `1e-4`, and only then starts production:

```bash
python run_kato_overlap_continuation_s25.py --report-only
./launch_kato_overlap_continuation_tmux.sh
tmux attach -t kato_overlap_continuation_N20x24_N24x24_s25_v1
```

The launch is pinned to physical cores 28--55 with 28 one-thread workers.
Raw CW and CCW paths and endpoint distributions are primary; quantization and
agreement with the older 64/128-grid classifications are outcomes, not gates.
See
[`docs/kato_overlap_continuation_methods.pdf`](docs/kato_overlap_continuation_methods.pdf)
for the derivation, adaptive rule, numerical gates, and interpretation.

## State-derived adiabatic projector pump

The campaign `N16x20_state_adiabatic_projector_pump_s10_v1` is the direct
state-only analogue of the equilibrium flattened-Hamiltonian calculation.  It
does not rerun the monitored dynamics.  Each of the twenty verified S10 burn-in
frames defines a trajectory-specific occupied projector and flattened parent,

```text
P_xi = F_xi F_xi^dagger,
h_xi = 1 - 2 P_xi.
```

A minimum-image Peierls phase threads one flux quantum through this parent.
The saved occupied rank is followed over 64 intervals by maximum neighboring
projector overlap, separately clockwise and counterclockwise.  The principal
observable is the continued subsystem charge
`q_x=(delta_N_right-delta_N_left)/2`; independently refilling the lowest-rank
projector at every flux provides the instantaneous closure control.  This is a
static spectral/topological diagnostic of the state prepared by one monitored
trajectory, not an additional monitored-time pump.

Open `state_adiabatic_projector_pump.ipynb` for the editable launch, live resume
inventory, and result displays, or use:

```bash
python run_state_adiabatic_projector_pump.py report
taskset -c 56-63 python -u run_state_adiabatic_projector_pump.py run --resume --workers 8
python analyze_state_adiabatic_projector_pump.py
```

The notebook runs this static calculation directly; no tmux wrapper is needed.
Every CW/CCW trajectory is one checksum-verified NPZ/completion-JSON task, so a
rerun immediately verifies and skips the completed S10 result.

## S100 static-projector ensemble

The versioned campaign `N20x24_state_projector_pump_s100_v1` determines whether
the near-unit sample-0 soft-wall result is representative of an ensemble or one
member of a record-dependent mixture.  It prepares 100 independent raster-y
trajectories for each of the soft and hard topological walls at `20x24`, using
48 cycles, pure half filling, `nshell=1`, `alpha=(1,30)`, perfect correction,
no postselection, and complex128.  It then applies the same 64-interval static
projector continuation in both directions to every final occupied frame.

There are 200 independently resumable burn-in tasks and 400 pump-path tasks.
Each task is one atomic NPZ/completion-JSON pair; pump completions additionally
pin their burn-in checksum.  The production launcher first validates sample 0,
then resumes the full ensemble and runs the preregistered analysis:

```bash
python run_state_projector_pump_s100.py report
./launch_state_projector_pump_s100_tmux.sh --dry-run
./launch_state_projector_pump_s100_tmux.sh
```

The default tmux session is `state_projector_pump_N20x24_s100_v1`.  It waits
until the fixed-flux campaign releases physical cores 28--55, then uses 28
one-thread workers.  The methods and interpretation boundary are fixed in
[`docs/state_projector_pump_s100_methods.pdf`](docs/state_projector_pump_s100_methods.pdf).
The descriptive event statistic `|q_x^odd|>0.5` exposes near-zero versus
order-one sectors but is neither a topological definition nor an acceptance
gate.

The campaign completed on September 3, 2026: all 200 burn-ins and 400 pump
paths verify.  The endpoint distribution is sharply bimodal, rather than a
broad distribution of fractional transfers.  Soft-wall trajectories contain
71/100 near-unit events and 29/100 closures; hard-wall trajectories contain
36/100 near-unit events and 64/100 closures.  The mean direction-odd endpoints
are `0.70728` (95% trajectory-bootstrap CI `[0.61777, 0.79622]`) and `0.35938`
(`[0.26947, 0.45905]`), respectively.  Among classified events, the means are
`0.99617` and `0.99828`; every non-event closes within `1.60e-8`.  The full
statistics and figures are under
`results/N20x24_state_projector_pump_s100_v1/analysis/`, and the updated
two-column note explains why these fractional ensemble means are mixture
weights rather than fractional single-trajectory pumps.

The version-3 follow-up size series keeps `Nx=20` and adds
`Ny=28,30,32,34,36`, with a fresh S100 soft ensemble and S100 hard ensemble at
every circumference. There is no additional burn-in: each monitored
trajectory runs once for exactly `2*Ny` cycles. The endpoint occupied frame is
saved because it is the input to the static projector pump; no strip entropy,
central-charge fit, fit residual, or other CFT information is calculated or
saved. The endpoint state receives the same 64-interval CW and CCW projector
continuation used at `Ny=24`.

The earlier version-2 size run was canceled and its partial output is preserved
without reuse. The completed `Ny=24` campaign and its post-hoc central-charge
analysis also remain unchanged. Inspect or launch version 3 with:

```bash
./launch_state_projector_pump_size_series_tmux.sh --dry-run
./launch_state_projector_pump_size_series_tmux.sh
```

The session `state_projector_pump_N20_Ny28_36_s100_v3` first runs sample zero
for all five sizes, then completes each size in ascending order.  Per-size
analysis is automatic; the final cross-size product includes pump statistics
for `Ny=24,28,30,32,34,36`. The added workload is 1000 monitored endpoint
trajectories and 2000 pump paths, all using independent version-3 configuration
identities and completion pairs.

A representative post-hoc S100 plot of one near-unit trajectory per wall is
available at
`results/N20x24_state_projector_pump_s100_v1/analysis/figures/state_projector_pump_representative_quantized_paths.pdf`.

### Dense-OW comparison at Ny=24

`N20x24_state_projector_pump_s100_dense_v1` repeats the completed Ny=24
state-projector pump with one scientific change: `nshell=None`, the canonical
dense/infinite-shell OW construction.  It retains the same soft/hard walls,
48-cycle raster-y monitored preparation, S=100 independent trajectories per
wall, complex128 arithmetic, and 64-interval CW/CCW projector continuation.
The independent root seed and output directory prevent pooling with the
`nshell=1` ensemble.

```bash
./launch_state_projector_pump_dense_tmux.sh --dry-run
./launch_state_projector_pump_dense_tmux.sh
```

The session `state_projector_pump_N20x24_dense_s100_v1` uses physical cores
0--27 with one BLAS thread per worker.  It verifies and runs sample zero for
both walls first, then resumes into the full campaign.  Each endpoint and pump
path remains an independently checksummed result/completion pair.
It uses soft sample 19 and hard sample 25, selected as the classified-event
endpoints closest to `q_x^odd=1`, and plots both directions against signed
`phi/(2*pi)`.

## Online S10 flux-ramp campaign

The versioned campaign `N16x20_online_flux_ramp_s10_v1` is the dynamical
follow-up.  It uses raster-y ordering and ten independent pure-state burn-ins
for each of the soft and hard topological walls.  Every parent runs for 40
zero-flux cycles, after which its exact occupied frame is branched into
independently Born-sampled counterclockwise and clockwise 16-cycle ramps.  The
two descendants share the same verified burn-in checksum but have different
continuation seeds.

The ramp uses the exact physical schedule

```text
phi_0 = 0
phi_j = sigma * 2*pi*j/16,  j=1,...,16,  sigma=+1 or -1
```

There is no `1e-7` offset in the online experiment: that offset selects a
spectral branch in the equilibrium/static protocol, whereas the physical ramp
starts from the actual zero-flux circuit state.  At each ramp cycle the
canonical CPU engine rebuilds the OW projectors at the new absolute twist
before applying any site update.  Each run records direct regional charge,
feedback-source-corrected charge, total charge/rank, continuity checks, and the
x-resolved density at all 17 flux values.

Inspect or launch the resumable production campaign with:

```bash
python run_online_flux_ramp.py report \
  --config campaign_config.online_ramp_s10_v1.json \
  --output-root results/N16x20_online_flux_ramp_s10_v1

./launch_online_flux_ramp_tmux.sh --dry-run
./launch_online_flux_ramp_tmux.sh
```

The launcher uses tmux session `online_flux_ramp_N16x20_s10_v1`, physical
cores 28--55, 28 workers, and one BLAS thread per worker.  Rerunning the same
launcher verifies all NPZ/completion-JSON pairs and skips completed tasks.
Only the interrupted burn-in or ramp is repeated.  Separate `tqdm` bars report
burn-in and ramp progress; analysis starts only after all 20 burn-ins and 40
ramps verify.

The combined equilibrium, completed static-response, online-ramp, and
continuous frozen-record methods/results note is
[`docs/charge_pump_results_working.pdf`](docs/charge_pump_results_working.pdf),
with editable REVTeX source beside it.

## Continuous frozen-record S10 ramp

The campaign `N16x20_frozen_record_continuous_ramp_s10_v1` reuses the twenty
verified cycle-40 burn-in frames above.  From every frame it generates one
64-cycle zero-flux raster-y measurement record, then resets to the same frame
and replays the identical record continuously along CW and CCW schedules with
`M=16,32,64`.  The state is propagated from one flux point into the next; it
is never restarted at an intermediate twist.  CW and CCW therefore have the
same outcomes, corrections, rank history, and injected charge, leaving the
twist path as their only difference.

The primary observable is the direct regional transfer
`q_x=(delta_N_right-delta_N_left)/2`.  No tangent modes, source subtraction,
or separate hold replay are used.  The zero-flux record-generation history is
retained as a diagnostic.  Inspect or launch with:

```bash
python run_frozen_continuous_ramp.py report \
  --config campaign_config.frozen_continuous_ramp_s10_v1.json \
  --output-root results/N16x20_frozen_record_continuous_ramp_s10_v1

./launch_frozen_continuous_ramp_tmux.sh --dry-run
./launch_frozen_continuous_ramp_tmux.sh
```

The tmux launch first completes sample 0 for both walls and all ramp lengths,
validates the paired histories, and then resumes the remaining S10 tasks.  It
uses cores 28--55, 28 one-thread workers, atomic NPZ/completion-JSON pairs,
and persistent `tqdm` progress.  Quantization is measured but is not an
acceptance gate.

The campaign is complete: 20/20 reference records and 120/120 ramp tasks
verify, with no failures.  The paired direction-odd endpoint response decreases
from `M=16` to `M=64`: soft `0.7122 -> 0.1475` and hard
`0.9139 -> 0.3199`.  Thus the rapid ramps show a reproducible handed response,
but it does not stabilize toward a nonzero adiabatic pump in this range.

## Fixed-flux quench S10 campaign

The follow-up campaign `N16x20_frozen_record_fixed_flux_quench_s10_v1`
removes the ramp entirely.  It reuses the same twenty burn-in frames and the
same twenty 64-cycle zero-flux records.  From each burn-in frame it replays the
entire record while holding every OW measurement at one fixed twist:
`+/-pi/2`, `+/-pi`, `+/-3pi/2`, or `+/-(2pi-1e-7)`.

The primary response is the direct paired state difference from the same
record's zero-flux history,
`response_q_x=((N_right(phi)-N_right(0))-(N_left(phi)-N_left(0)))/2`.
This is not the analytic feedback-source subtraction used in an earlier
diagnostic.  It is a sudden fixed-flux quench of a conditioned measurement
channel, not an adiabatic pump.  Inspect or launch with:

```bash
python run_fixed_flux_quench.py report \
  --config campaign_config.fixed_flux_quench_s10_v1.json \
  --output-root results/N16x20_frozen_record_fixed_flux_quench_s10_v1

./launch_fixed_flux_quench_tmux.sh --dry-run
./launch_fixed_flux_quench_tmux.sh
```

The 160 deterministic tasks use atomic NPZ/completion-JSON pairs.  The tmux
launcher validates all eight twists for sample 0 before resuming the remaining
nine samples, uses cores 28--55 with 28 one-thread workers, exposes persistent
`tqdm` progress, and runs the analysis only after every pair verifies.

## Pure-state tangent-flux extension

The versioned sibling campaign `campaign_config.pure_tangent_v1.json` studies the
projective spectrum of the same finite-depth circuit idea without changing the
completed static-response outputs.  At the user's request it uses a **raster-y**
schedule, so it generates new zero-twist hard- and soft-wall reference records
under `results/N16x20_frozen_flux_pure_tangent_v1/references`; it does not
mislabel or reorder the earlier random-serial records.

For each reference it replays 17 signed twists in both orientations and
accumulates the full 40-cycle occupied/empty tangent cocycle.  It saves up to
64 finite low-gap particle--hole candidates, requires at least 32, and
continuously follows 32 branches using
the product of occupied- and empty-leg overlaps after undoing the uniform twist
gauge.  The four wall weights, final-projector occupation diagnostics, and the
derived excitation charge are saved separately.  These are spectral
particle--hole observables, not fractional occupations of the pure state and
not a quantized monitored pump.

Inspect the resolved workload without starting it:

```bash
python run_pure_tangent_flux.py --report-only
./launch_pure_tangent_tmux.sh --dry-run
```

The production launcher uses physical cores 28--55, 28 workers, and one BLAS
thread per worker.  Every twist is one resumable NPZ/completion-JSON task.  On
successful completion it runs `analyze_pure_tangent_flux.py` and writes the
continued branches under the new result directory's `analysis/` folder.

The dedicated two-column tangent methods writeup is available as
[`docs/pure_tangent_flux_pilot.pdf`](docs/pure_tangent_flux_pilot.pdf), with
editable REVTeX source beside it.  It includes the exact equilibrium
flattened-Hamiltonian benchmark as context while reserving all circuit-pilot
results for a later amendment.

The campaign uses two independent `16×20` reference trajectories, one for the soft explicit interface and one for the hard support-terminated wall. Each trajectory runs for `40 = 2*Ny` cycles with `nshell=1`, `alpha_1=1`, `alpha_2=30`, pure half-filled initialization, random serial ordering, perfect correction, no postselection, and complex128. Every circuit evolution calls the canonical CPU entry point `classA_U1FGTN.run_markov_circuit`.

## Twist convention and the `1e-7` offset

Let `sigma=+1` denote increasing/counterclockwise flux and `sigma=-1` decreasing/clockwise flux. The static replay grid is

```text
phi_j = -sigma * 1e-7 + sigma * 2*pi*j/16,  j=0,...,16.
```

Consequently, increasing flux begins at `-1e-7`, while decreasing flux begins at `+1e-7`. Each wall uses one reference record generated at exactly zero twist, and that same record is imposed at every positive- and negative-direction replay. Its measurement outcomes, feedback targets, and total injected charge are therefore identical across all twists.

The offset is not a Hamiltonian regulator. In the exact flattened-Hamiltonian `20×40` benchmark, the two physical-wall levels form an exponentially small avoided crossing at zero flux: the closest levels are approximately `±5.49e-10` for the soft wall and `±7.51e-10` for the hard wall. Starting infinitesimally to one side selects an unambiguous wall branch. The quantization-favoring choice is the sign opposite to the winding direction:

| Exact `20×40` reference | Increasing flux | Decreasing flux |
|---|---:|---:|
| `phi_0=-1e-7`, soft wall | `+0.9999999976` | `-0.9839718413` |
| `phi_0=+1e-7`, soft wall | `+0.9839718590` | `-0.9999999924` |
| `phi_0=-1e-7`, hard wall | `+0.9999999987` | `-0.9808096918` |
| `phi_0=+1e-7`, hard wall | `+0.9808096944` | `-0.9999999994` |

The `0.98` values reflect the small initial weight on the opposite wall, not flux-grid error. These nearly integer transfers belong to **continuously tracked occupied branches of the exact flattened Hamiltonian**. This frozen-record pilot evaluates every twist independently. It must close after a `2*pi` loop and has no quantization acceptance gate. It measures intermediate conditional wall-to-wall redistribution, not an online monitored-time pump.

The full derivation and distinction are in [`wall_resolved_flux_pump_handoff.pdf`](../Paper%20Methods/open_system_monitored_system_response_theory/wall_resolved_flux_pump_handoff.pdf).

## Endpoint observable

Only the final state after all 40 replayed cycles is measured. With

```text
L: x=0,...,7
R: x=8,...,15
```

the direction-specific response is

```text
Delta N_a(phi_j) = N_a^final(phi_j) - N_a^final(phi_initial)
q_wall = (Delta N_R - Delta N_L)/2.
```

The reference record fixes `Delta N_inj = sum_e(target_e-outcome_e)`. Every replay must reproduce this same global injection, and `Delta N_L + Delta N_R` must vanish within tolerance. No cycle-resolved charge or covariance history is saved.

## Launch and resume

Inspect the complete resolved workload without writing output:

```bash
00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/launch_tmux.sh --dry-run
```

Launch the default 56 single-thread worker campaign on one hardware thread from each physical core:

```bash
CAMPAIGN_ID=N16x20_frozen_flux_charge_v1 \
CPU_LIST=0-55 \
WORKERS=56 \
00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/launch_tmux.sh
```

The launcher prints the tmux session, log, output, attach, and follow commands. To resume the same campaign:

```bash
CAMPAIGN_ID=N16x20_frozen_flux_charge_v1 \
CPU_LIST=0-55 \
WORKERS=56 \
00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/launch_tmux.sh --resume
```

Each twist is an independent result/completion pair. Restart verifies hashes and reruns only missing or invalid points. Worker progress is summarized by one `tqdm` bar so parallel output stays readable.

Override `OUTPUT_ROOT`, `CONFIG`, `SESSION`, or `BLAS_THREADS` through environment variables. Do not use more workers than CPUs in `CPU_LIST`.

## Products

`results/<campaign-id>/` contains:

- the two compressed reference records and parent states;
- one compact NPZ and completion JSON per static twist;
- `charge_vs_phi.csv` and `charge_vs_phi.npz`;
- a manifest containing the exact configuration, source hashes, offset rationale, injection checks, replay probabilities, and closure residuals;
- a double-column PDF and 300-dpi PNG diagnostic figure; and
- `analysis_summary.json`.

The expected output is below 100 MiB. On the repository host, the 68 replay points should finish in roughly 20–35 minutes with 56 physical-core workers, subject to the first completed wave's measured timing.

## Post-hoc regional source subtraction

The completed records are sufficient to separate direct correction-source placement
from the final wall-charge response; the expensive covariance dynamics do not need to
be rerun. For every correction event, the analysis reconstructs

```text
A_a(phi) = sum_e [target_e - outcome_e] tr[R_a P_e(phi)]
q_a(phi) = N_a^final(phi) - N_a^final(0) - [A_a(phi) - A_a(0)]
```

Run:

```bash
python 00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/analyze_source_subtracted.py \
  --campaign-dir 00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/results/N16x20_frozen_flux_charge_v1
```

The source-subtracted products are written to the campaign's
`source_subtracted_analysis/` directory. In the completed pilot, the largest
twist-dependent direct-source contribution is `2.93e-7` for the soft wall and
`1.96e-7` for the hard wall, whereas the corresponding source-subtracted wall
responses reach `0.628069` and `0.433350`. Thus direct spatial displacement of
the feedback source explains less than `4.7e-7` of the raw response. The signal
is a genuine source-subtracted conditional endpoint redistribution. Near zero
twist it is direction-odd: the centered finite-difference values are
`2*pi*dq_x/dphi = 0.99723` (soft) and `0.99899` (hard), consistent with a
handed, spectral-flow-like response in these individual records. The static
family still closes around the `2*pi` circle, so it is not a measured
microscopic current or a quantized dynamical pump. A robust chirality claim
requires multiple independent records and matched trivial/reversed-Chern
controls.
