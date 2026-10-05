# Bundle-level clarifications and amendments

The audit PDF remains the controlling experimental specification.  This file records
places where the executable bundles must resolve an omission, contradiction, or storage
constraint.  It is not permission for silent protocol drift.

## A1 — H2 schedule control

The H2 acceptance paragraph mentions random and reversed schedules, but the fixed
production choices, H2 run matrix, Colab allocation, and subsequent user instruction retain
only the production random-serial law.  The bundles therefore run `sequence="random"`
only at `S=25`.  Schedule replay/reversal may appear in `00_validation` as a diagnostic,
but it is not a production family and is not pooled with H2.

## A2 — M2 imperfection grid

The PDF requests a “declared imperfect-correction grid” without enumerating it.  The bundle
declares correction-error probabilities
`[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5]`, implemented as
`perfect_correction=False` and `p_gain=p_loss=1-error_probability`.  Perfect correction is
a separate baseline; forced postselection is a deterministic one-record endpoint.

## A3 — M3 dense noise grid

The PDF specifies the BPJ onsite U(1) noise convention and a dense grid but not its points.
The bundle declares `sigma=0,0.025,...,0.7`, resolving both external reference scales near
0.2 and 0.55.  Wall scans launch only after the domain-wall-free bulk gate identifies the
finite-size transition region.

## A4 — corrected covariance memory accounting

The PDF's numerical example understates the covariance size by a factor of four.  The
canonical engine has `Nlayer=2*Nx*Ny`, hence `Nlayer=3200` at `20x80`.  A dense complex128
covariance is 163.84 MB (decimal), and three snapshots for 25 trajectories are 12.288 GB.
Even Hermitian-triangle storage is approximately 6.146 GB before records, tangents, and
other live buffers.
The executable preflight uses the canonical dimension and therefore requires shard-local
parent processing; it plans zero permanent covariance bytes.

## A5 — streamed observables instead of permanent covariance snapshots

The newest user instruction prefers computing observables on the fly rather than retaining
ensemble covariance matrices.  Production bundles therefore treat parent covariances as
transient in-memory or local-scratch objects only.  P1/W1/S1/T1/P2 and scan observables are saved
per trajectory in compact arrays. H1/H2 consume the required five-sample parent state in
`/content` before it is discarded; H3 regenerates each state from the compact parent record
inside its replay stage. Covariance matrices are not copied to Drive, even as
nominally temporary production products.  The permanent archive retains the initial RNG state, complete
ordered record, source/engine hashes, and replay residuals so the covariance can be
regenerated; it does not retain the covariance by default.

## A6 — speed-oriented stage fusion

Removing permanent covariance products makes the most efficient unit a complete live
five-trajectory shard. `02_pure_wall_master` therefore evaluates S1/T1/B2 and the H1/H2
descendants while the relevant parent states are already resident, archiving compact stage
products atomically. This avoids replaying the common prefix for every H1 snapshot and H2
time origin. H3 remains in `03_chirality_replay`: its 65–129 full twist replays dominate
runtime and need independent checkpoint/restart boundaries. This is an execution-layout
change only; trajectory independence, sample counts, and estimators are unchanged.

## A7 — RNG and validation precision

The canonical GPU engine currently uses one stateful Torch RNG stream per immutable
five-trajectory shard. The manifest records the global sample indices, derived shard seed,
and pre/post generator states. Exact replay and restart are therefore guaranteed for that
fixed shard, but a one-batch 20-sample run is not claimed to be bitwise identical to four
five-sample runs. Repartitioning is a versioned ensemble change. The executable V0 gate
tests 256-branch normalization, frozen replay and likelihoods, observer neutrality,
CPU-device/GPU-device frozen-word parity, tangent-QR residuals, and zero covariance bytes.

## A8 — compact noise and response records

M3 archives per-cycle phase-field moments and SHA-256 digests, not the full onsite phase
array; the pre-run RNG state, engine hash, and case configuration regenerate it exactly.
H2 translates eight sources within each trajectory, saves their within-trajectory mean and
source sample variance, and only then treats the parent trajectory as the bootstrap unit.
H1 and H2 each write an atomic compact Drive checkpoint before the next fused stage begins.

## A9 — compact ordered-record tiers

Every stochastic family archives the ordered site word, bit-packed outcomes and targets,
and per-cycle self-information, so every trajectory remains replay-capable. Per-event
float64 probabilities and reset diagnostics are consumed live and discarded. This retains
the stronger provenance requested after the user clarified that Drive will be offloaded
between experiments, while still avoiding unnecessary event-float duplication. H2/H3/M1
replay uses the same compact format.

## A10 — operational Drive ceiling

Drive is recycled between experiments: 12 GB is the planning ceiling for one active
experiment and 14 GB is an exceptional hard edge. Completed, checksummed archives are
offloaded to the cluster before Drive is cleared. Storage optimization must therefore
remove covariance histories and redundant event floats, but it must not replace the
bit-packed replay word with a weaker digest-only record.

## A11 — P2 bridge and R1 placement

P2 now uses the canonical reduced-Choi tracker in the same max-mix trajectory call. At
cycles `1,2,4,8,16,Ny,3Ny/2,2Ny`, it compares `sigma_LR^dagger sigma_LR` directly with
`4*C*(1-C)` and saves recordwise Frobenius, eigenvalue, singular-mode, and entropy
residuals; neither dense matrix is archived. R1 is placed in `02_pure_wall_master` because
its only new stochastic inputs are the primary wall/control records at
`Ny=16,20,28`. Its SCGF and charge-sector products are merged from the compact replay
records after all five shards complete.

## A12 — executable deterministic M1 controls

The fixed-schedule M1 descendants remain paired one-for-one with compact Born parents.
Bundle `05` additionally exposes two non-trajectory jobs: exact Kraus-branch Fock-space
validation at `2x2` and `2x3` with canonical covariance-sweep checks at `4x6` and `6x8`,
and matrix-free complex128 Lanczos for the Poissonized homogeneous generator on the
declared `Nx={8,12}`, `Ny={16,24,32,48,64}` grid. The latter retains only leading Ritz
values, decay rates, residuals, and momentum-sector checks; it never materializes or saves
a dense covariance superoperator.

## A13 — enforced downstream launch decisions

The `04` and `05` production runners now require a schema-v1 accepted decision file in
addition to the joint W1/B0 width artifact. P2 requires `T1_viable`; each S2 or M1–M3 case
checks the subset of `P1_viable`, `S1_viable`, and `T1_viable` declared in its production
configuration. The shipped templates are deliberately `pending` and cannot launch a run.

## A14 — successive-cycle covariance convergence

Every primary and fixed-schedule-mean trajectory now archives the per-trajectory diagnostic
`frobenius(G_cycle-G_previous_cycle)/Nlayer` at every physical cycle. The observer initializes
from cycle 0, retains only the preceding live covariance batch on the GPU, and releases that
scratch buffer at the final cycle. The permanent product is therefore only one float64 value
per trajectory and cycle; no covariance history is retained.

## A15 — corrected P1 shell sweep

The subsequent user instruction supersedes the pre-v2 working PDF's lean P1 matrix. The versioned
P1 bulk baseline now runs `n_shell=2,1,None` for every declared pure and max-mix bulk
point. Finite shell depths use the canonical local backend; `None` uses the canonical
dense backend. P1 case IDs carry the `rev-shell_sweep_v2` tag so the corrected ensemble
cannot overwrite or be silently pooled with the superseded lean P1 artifacts.

## A16 — simplified optional B1 surrogate-attraction test

The earlier CPU B1 pilot mixed charge-resolved static residual profiles using empirical
sector frequencies and compared them with held-out wrong-outcome activity. That is no
longer the production question. Bundle `06_b1_controller_frame` instead constructs the
signed controller-frame operator once and tests whether 25 canonical GPU trajectories at
`Nx=20, Ny=48` approach its nominal half-filled ground-state manifold over exactly `2*Ny`
cycles. The all-trivial full-support arm is the matched control. Instantaneous integer
charge appears only in the rank-matched Ky Fan lower bound; no sector frequency, fitted
weight, or training/test split is defined. B1 is an optional final appendix diagnostic and
has no downstream launch gate or infinite-time interpretation.

## A17 — paired-initialization S2 phase-boundary scan

S2 runs the identical eight-mass, four-circumference phase-boundary grid for two separate
Born ensembles: independently randomized half-filled pure Gaussian starts and maximally
mixed starts. Pure cases retain the stationary late checkpoints and stabilized tangent QR
products. Max-mix cases additionally retain early/logarithmic purification checkpoints and
the recordwise reduced-Choi Gram comparison with `4*C*(1-C)` at
`1,2,4,8,16,Ny,3Ny/2,2Ny`. Case IDs carry `init-pure` or `init-maxmix`; their archives,
bootstraps, and finite-size fits are never pooled. S2 requires the accepted T1 decision as
well as the P1 and S1 decisions so both arms use the validated tangent convention.

## A18 — dense Choi tracking removed from production

The subsequent user instruction supersedes the Choi-tracking portions of A11 and A17.
P2 and both S2 initialization arms now run with `track_choi=False`; they neither propagate
nor archive a reduced Choi covariance. P2 retains the independently valuable same-record
purification curves and physical tangent QR products, and tests their finite-time and
finite-size scaling statistically rather than imposing the former exact Gram identity.
Max-mix S2 retains its early/logarithmic purification grid and physical tangent spectrum.
The generic canonical-engine Choi implementation and all legacy Choi datasets remain
unchanged for historical diagnostics, but they are outside the production contract.

## A19 — deterministic mean-channel/Lindblad work moved to CPU

The subsequent CPU-campaign instruction supersedes the earlier bundle-07 plan and the M1
schedule-descendant plan. The finite-success averaged channel is available analytically,
so production evaluates that exact completely positive affine covariance map directly and
does not draw trajectories or random schedules. Its continuous-time weak-success generator
is a second deterministic arm, not a fitted surrogate. Both arms merit compact main-text
discussion, with derivations and robustness controls in an appendix.

The local campaign at `../mean_channel_lindblad_cpu_campaign/` uses complex128 NumPy/SciPy
CPU execution. The main matrix fixes `Nx=20`, scans the S2 eight masses and four
circumferences, runs `n_shell=1,2,None` at every point, starts from the maximally mixed mean
covariance, and observes through `2*Ny` with no burn-in. Exact finite-channel strengths
`p=1,1/2,1/4,1/8,1/16` are compared at fixed physical time with the continuous solution;
there is no fitted clock conversion. Empty and filled starts, wall/filling conventions,
uniform phases, and representative full-dephasing cases are deterministic controls.

Only `n_shell=None` may calculate, archive, plot, or support claims about a `k_y`-resolved
occupation spectrum, wall branch, branch weight, or momentum-resolved slope. The
`n_shell=1,2` products are restricted to real-space or momentum-integrated occupation,
purity, wall-localization, entropy, variance, relaxation, convergence, and residual
diagnostics. Configuration validation, schemas, analysis, and tests enforce this boundary.

The older `CI_Lindblad_DW.py` generator remains scientifically useful historical
infrastructure, but its narrower wall convention, number-covariance convention, optional
quartic-dephasing closure, and incompletely archived outputs are explicitly labelled. The
production spectral-flow arm first reconstructs the controlled no-dephasing gain/loss
sector and compares canonical and legacy wall widths. Representative full-dephasing
calculations are explicitly labelled and run at all three shell choices. No covariance
matrix or history is archived; permanent products are only shell-appropriate compact
observables, timings, hashes, manifests, convergence arrays, and numerical residuals. The
mixed wall flow is an interface sector of a mixed state, not preparation of a pure chiral
critical state.

## A20 — B1 execution is required

The subsequent user instruction supersedes only the optional scheduling status stated in
A16 and the locked working PDF. Bundle `06_b1_controller_frame` is a required final
campaign and runs after bundle `01` in the first three-session Colab lane. Its already
locked numerical contract is unchanged except for the later sampling amendment A21: both
`20x48` interface/control cases, five-trajectory shards, exactly `2*Ny` cycles, and the declared
rank-matched controller-frame observables. B1 still has no downstream launch gate and its
result cannot reinterpret preceding campaigns.

This is a scheduling-scope amendment, not a change of Hamiltonian, estimator, ensemble,
duration or output schema. The superseded 100-sample contract remains provenance and must
not be pooled with the versioned 25-sample campaign below.

## A21 — versioned 25-trajectory production campaign

The subsequent user instruction replaces the 100-trajectory production ensemble with
`S=25` independently initialized trajectories at every ordinary stochastic parameter
point. The execution unit remains five complete trajectories, so each ordinary case has
exactly five immutable shards with indices `0..4`. All geometries, parameter grids,
initialization laws, cycle counts, estimators, convergence diagnostics, dtype, controls,
and gate definitions are unchanged. Configurations and manifests carry the sampling
revision `production_25sample_v1`; superseded 100-trajectory archives are provenance only
and are never pooled automatically.

H3 remains the explicit exception established by the flux-pilot promotion decision: it
uses the two agreed 65-point cases and parent shards `0,1`, giving 10 frozen records per
protocol. Deterministic one-record controls likewise retain their declared sample count.

## A22 — H3 representative-record scope

The subsequent H3 scope decision replaces the 10-record-per-protocol replay with one
preregistered frozen record from parent shard `0`, record index `0`, for each of the
explicit-interface and matched-trivial protocols. Each record is followed around the
complete 65-point closed twist circle. This is a representative branch-level spectral-flow
diagnostic, not an estimate of a Born-ensemble frequency; consequently the former
quantized-fraction and 10-of-10 acceptance requirements are retired. The record-level
requirements remain finite branch weight, gauge-correct closure, unambiguous opposite wall
flows in the interface case, and zero flow in the matched-trivial control. Additional
records, denser grids, or larger sizes require a separately reviewed campaign revision.

## A23 — versioned 10-trajectory campaign and 33-point H3 circle

The subsequent runtime-budget decision replaces the active 25-trajectory production
target by `S=10` independently initialized trajectories at every ordinary stochastic
parameter point. The execution unit remains five complete trajectories, so a new case
has exactly two immutable shards with indices `0,1`. Explicit deterministic and forced-
postselection controls remain one-sample cases. Geometries, schedules, initializations,
cycle counts, estimators, convergence diagnostics, dtype, and acceptance thresholds are
otherwise unchanged. Configurations and manifests carry the sampling revision
`production_10sample_v1` and write beneath the versioned output namespace
`classA_final_production_outputs/production_10sample_v1`.

Checksum-verified 25-sample archives remain read-only provenance and valid statistical
supersets. A legacy shard may satisfy the new minimum only when its case, model, run
parameters other than the declared total sample count, observation contract, global
sample indices, shard seed, canonical engine hash, and archive checksum match exactly.
All verified trajectories in such a case remain available to analysis; they are neither
truncated nor recomputed. Incompatible archives are never pooled or silently adopted.

H3 remains one preregistered frozen record from parent shard `0`, record index `0`, for
each of the interface and matched-trivial protocols. Its production circle is reduced
from 65 to 33 points, including both endpoints, with flux indices `0..32` and
`phi=2*pi*index/32`. This remains a representative branch-level spectral-flow
diagnostic and makes no Born-ensemble frequency claim.

## A24 — two-construction S2 entanglement-transition scan

The active production revision is `production_10sample_v2`, written beneath
`classA_final_production_outputs/production_10sample_v2`. It preserves every pre-existing
S2 case dictionary and adds a pure-state support-terminated mirror at the same eight
non-singular masses, four circumferences, accepted width, two five-trajectory shards, and
`2*Ny` duration. These 32 new cases use `DW=true`, `dw_truncation=true`,
`meas_slab_only=true`, `n_shell=1`, `alpha_1=alpha_in`, and `alpha_2=30`. They retain the
same random schedule, perfect correction, complex128 covariance, rank-16 tangent frame,
three entropy checkpoints, correlator products, and final local marker as the explicit-
interface pure arm. No support-terminated max-mix cases are added.

The resulting queues contain 148, 50, and 677 shards in lanes 1--3 respectively: 875
shards before reuse, with bundle 05 contributing 653. Reuse searches the read-only v1
namespace before the old unversioned tree, collapses byte-identical duplicates, and fails
on multiple distinct compatible candidates at one priority. Reuse still requires exact
case physics, observation contract, sample indices, seeds, engine, audit, checksum, and
receipt agreement after ignoring only the declared total sample count. All 10--25
compatible trajectories are retained.

CPU postprocessing fits every final-time trajectory over `2 <= Ay <= Ny/2-1` to both a
constant area-law model and
`S(Ay)=b+m*log[(Ny/pi)*sin(pi*Ay/Ny)]`, reporting `c_eff=3m` and
`Delta AICc=AICc_area-AICc_log`. Whole trajectories are bootstrapped with 2,000
deterministic resamples; checkpoint stability, actual sample counts, receipts, and input
hashes remain in the analysis products. A common transition is claimed only if both
constructions approach `c_eff=1` at `alpha=1`, approach zero at `alpha=3`, show the
corresponding model preference, and have overlapping finite-size crossover brackets
between 1.875 and 2.125. Otherwise the result is reported as distinct or inconclusive.

This revision retains the existing A100-only execution contract. H100 remains unsupported
until a separate validation and numerical-equivalence amendment is approved.

## A25 — restored 25-trajectory floor, lean size allocation, and bundle execution

The subsequent user decision retires A23--A24 as the active sampling and orchestration
contract while preserving all of their outputs as provenance.  Every ordinary stochastic
point again uses `S=25` independently initialized trajectories, stored as five immutable
five-trajectory shards.  The new sampling revision is
`production_25sample_v2_lean`, written beneath
`classA_final_production_outputs/production_25sample_v2_lean`.  H3 remains the explicit
representative-record exception.  Forced and partial postselection are absent.

The transferred `production_25sample_v1` P1 matrix is not redefined.  Its original audit,
engine, case dictionaries, seeds, and unversioned output directory are frozen in
`01_p1_existing_completion`; 226 verified shards are retained and only the 14 missing
shards may be computed before that runner stops at P1.  The active bundle-01 configuration
therefore disables P1 and contains only the revised W1 matrix.

Large circumference is allocated to the observables that require it.  The primary
S1/T1 wall/control family uses `Ny={20,40,60,80}` so entropy and subsystem-charge fits can
extract and extrapolate the central charge `c` and U(1) current level `k`.  The
support-terminated family and R1 Born-record family use `Ny={20,40,60}`.  W1 uses
`Nx={20,24,28}` and `Ny={40,60}`.  P2 uses `Ny={20,40,60}`.  H1 and H2 descendants run at
`Ny={20,40,60}`, and H3 replays the preregistered record at `20x40` on the 33-point circle.

S2 performs the full eight-mass scan at `Ny=40` and repeats only
`alpha={1,1.875,2.125,3}` at `Ny={20,60}` for finite-size trends.  M2 performs its full
perfect/imperfect grid at `Ny=40` and repeats the perfect, 10%, and 50% error points at
`Ny={20,60}`.  M3 performs its dense noise grid at `20x20`, repeats
`sigma={0,0.2,0.55,0.7}` at square sizes `16,24,28,32`, and uses `Ny={20,40,60}` for the
post-gate wall bracket.  B1 uses the required wall/control comparison at `20x40`.

The three monolithic Colab lanes are retired.  Every generated numbered notebook launches
one bundle, defaults to a receipt/checksum report, resumes only missing shards, and stops
at the bundle boundary.  Distinct bundles or explicitly disjoint case-prefix queues may
run concurrently.  Gate artifacts continue to enforce scientific dependencies.  A
filtered queue runs a finalizer only when it is exactly the complete 24-case W1 matrix or
the complete 45-case M3-bulk matrix; all other filtered queues stop without finalization.

## A26 — H1 wall-retention indexing correction

The first `02_pure_wall_master` launch completed its five parent trajectories but exposed
a postprocessing-only NumPy advanced-indexing error in the live H1 modular-transport
descendant.  Selecting a packet and a Boolean wall mask in one expression moved the three
selected x columns ahead of the 161-point modular-time axis, producing incompatible
retention and displacement shapes `(3,)` and `(161,)`.

The corrected reducer selects the packet first, then applies the wall mask, sums only the
transverse and wall-window axes, and asserts one retention value per modular-time point.
A regression test fixes the production geometry at 161 times and a three-column wall
window.  The synchronized bundle source manifests and distributable manifest carry the
new helper hash.  No physical model, run configuration, seed, trajectory word, estimator,
fit rule, or acceptance threshold changes.  The failed launch emitted no completed shard
archive, so it is rerun from shard zero after replacing the package.

## A27 — support-terminated tangent-frame basis correction

The first support-terminated W1 shard exposed an observer-only basis mismatch.  For
`Nx=Ny=20`, the canonical tangent engine correctly evolves its frame in the full
800-row one-particle lattice basis, while the support-terminated writer declares the
440-row active topological slab.  The writer previously assigned the full
`(5,800,16)` frame directly into `(5,440,16)` active-slab storage.

The corrected writer accepts either representation.  A full-lattice frame is restricted
in the declared `active_top_layer_indices` order before storage; a frame already expressed
in that basis passes through unchanged; every other row dimension is rejected.  The saved
NPZ now includes `frame_basis_indices`, making each reduced row's full-lattice identity
explicit.  Regression coverage uses the exact 20-by-20, 800-to-440, five-trajectory,
rank-16 geometry from the failed shard.

This changes neither tangent evolution nor any physical trajectory.  The first four W1
shards completed before the failure and remain checksum-resumable.  The failed fifth item
emitted no complete archive and resumes at that item after package replacement.

## A28 — B1 charge-roundoff and persistence correction

The first B1 `20x40` interface shard completed all five trajectories and 80 cycles, then
the post-run observer check rejected a maximum total-charge integer residual of
`4.0870418160920963e-10`.  The original implementation incorrectly reused the
`1e-10` Ky Fan numerical tolerance as an absolute trace-integrality threshold.  At active
one-particle dimension 1600 this residual is complex128 accumulation roundoff, not
physical charge leakage.

The corrected B1 revision keeps the Ky Fan and other numerical tolerance at `1e-10` and
introduces a separate, bounded `charge_integer_tolerance=1e-8`.  It records both the
observed residual and both tolerances in the manifest.  The sample count, shard seeds,
geometry, dynamics, trajectories, observer definitions, duration, and scientific
classification are unchanged.

Compact ordered records and per-sample, per-cycle observer histories are now written
before acceptance validation.  A future validation failure is archived under the
`_rejected_validation` tier before the child raises, so a completed trajectory is never
silently discarded.  Accepted corrected archives use the versioned output directory
`06_b1_controller_frame_chargefix_v1`; the failed session log remains provenance and is
not represented as a completed shard.  Because the original failure occurred before any
archive or receipt was emitted, shard zero must be rerun after replacing the B1 package.

## A29 — standalone lean P1 Chern-dynamics revision

The subsequent P1 redesign supersedes all earlier P1 run matrices and gate language while
preserving their bundles and archives as immutable provenance. The new standalone bundle is
`01_p1_chern_dynamics`, and its output revision is
`production_25sample_p1_chern_v1`. It does not modify, resume, or pool data from
`01_p1_existing_completion` or `01_bulk_width_gate`, and it does not redesign W1.

P1 now contains exactly 12 uniform square cases: `L={16,24,32,64}` crossed with
`n_shell={1,2,None}`. Every case uses 25 random half-filled Slater trajectories in five
immutable five-trajectory shards, `T=L`, no burn-in, `DW=False`,
`alpha_1=alpha_2=1`, `n_a=0.5`, random serial order, Born sampling, perfect correction,
and complex128 arithmetic through `classA_U1FGTN_gpu.run_markov_circuit`.

The only scientific product is the repository-`+1` real-space Chern estimator at every
cycle including cycle zero. Ten distinct centers are drawn afresh per trajectory-cycle by
a counter-derived seed that excludes `n_shell`, so corresponding shell cases use identical
centers. Each radius-`0.4L` tripartition uses periodic minimum-image coordinates. Archives
retain center coordinates, ten raw values, their within-trajectory mean, sample IDs and
seeds, configuration/source hashes, timing, and checksums; all covariance, Bott, density,
entropy, tangent, convergence, purity, and ordered-record products are retired.

Merging averages centers first and the 25 independent trajectories second. The compact
figure has no uncertainty band; raw trajectory variability and a summary table remain
available. No P1 pass/fail threshold is introduced. Before production, one full
five-trajectory `L=64`, `n_shell=1` A100 shard must create a version-matched safety receipt.
Production refuses to launch if the receipt is missing, stale, or reports peak reserved
memory above 80% of the A100 capacity. No scientific parameter is silently reduced.
