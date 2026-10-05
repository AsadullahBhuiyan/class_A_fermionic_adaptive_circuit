# Wall-diabatized S100 width sweep runbook

This runbook covers the campaign
`N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1`. Older endpoint,
fixed-grid, physical-time RK4, and Kato outputs remain separate and must not be
rewritten.

Status on September 9, 2026: implementation, local validation, notebook
generation, and endpoint acquisition are complete.  The corrected A100 run
passed its locked timing and memory gate, completed all 230 five-trajectory
shards, and was imported without modifying Drive.  The repository import
contains 1,000 primary endpoints and 150 bridge endpoints; all 230 NPZ/JSON
pairs pass byte-count, SHA-256, embedded-identity, and scientific-contract
verification. The first exact-control launch exposed an implementation
mismatch and was stopped: it twisted `I-2*P_exact(0)` rather than rebuilding
the exact OW Hamiltonian family. The corrected control reconstructs
`H_exact(phi)` at every flux and continues eigenvectors independently within
each conserved transverse-momentum block, exactly as in the authoritative
equilibrium benchmark. The production spectral sweep and its sensitivity
analysis remain pending until the corrected controls and the
production-shaped smoke test pass.

The authoritative import receipt is
`imported_endpoints/wall_pump_width_endpoints_s100_v1/DOWNLOAD_MANIFEST.json`.
It records the authenticated account `abhuiyan2398@gmail.com`, 1,150
trajectories, 461 downloaded files, 16,905,617,976 bytes, 230 verified pairs,
and zero failed pairs.  The accompanying A100 receipt selected execution batch
80, retained more than 8 GiB of measured headroom, and projected 30.043 hours
for the 1,000 missing endpoints against the locked 40-hour cutoff.

## Document and code map

| Purpose | Authoritative artifact |
|---|---|
| Scientific definition and interpretation | `docs/wall_diabatic_spectral_pump_methods.tex` and compiled PDF |
| Field-level file/schema reference | `WALL_DIABATIC_DATA_DICTIONARY.md` |
| Locked scientific configuration | `campaign_config.wall_diabatic_spectral_pump_s100_v1.json` |
| Local execution/resume profile | `local_profile.wall_diabatic_spectral_pump_s100_v1.json` |
| Primary spectral runner | `run_wall_diabatic_spectral_pump_s100.py` |
| Exact controls and sensitivities | `run_wall_diabatic_controls_and_sensitivities.py` |
| Control ledger | `validate_wall_diabatic_controls.py` |
| Statistical analysis | `analyze_wall_diabatic_width_sweep.py` and `wall_diabatized_io.py` |
| Local tmux orchestration | `launch_wall_diabatic_local_tmux.sh` and `wall_diabatic_local_tmux_entrypoint.sh` |
| Current control/smoke handoff | `wall_diabatic_gated_queue_entrypoint.sh` |
| A100 endpoint bundle | `../final_production_new_designs/11_wall_pump_width_endpoints/` |
| Canonical CPU fallback | `run_wall_pump_endpoint_cpu_fallback.py` and `launch_wall_pump_endpoint_cpu_fallback_tmux.sh` |
| Optional spectral GPU benchmark | `benchmark_a100_spectral_eigensolver.py` |

## Locked data lineage and workload

All primary cells contain S100 soft-wall and S100 hard-wall endpoint states.
The three reused cells supply 600 immutable CPU endpoints; the five missing
cells require 1,000 new endpoints.

| `Nx x Ny` | `nshell=1` source | dense source |
|---|---|---|
| `20 x 24` | reuse verified CPU S100 | reuse verified CPU S100 |
| `24 x 24` | reuse verified CPU S100 | generate S100 |
| `28 x 24` | generate S100 | generate S100 |
| `32 x 24` | generate S100 | generate S100 |

The preferred GPU route also creates independent S25-per-wall bridge cohorts
for `nshell=1` at `Nx=20,24` and dense correction at `Nx=20`, adding 150
endpoints. Bridge samples are a backend sensitivity cohort and never enter the
primary S100 estimates. The all-CPU fallback contains no bridge.

Endpoint preparation is fixed to 48 raster-y cycles, pure half filling,
`alpha_1=1`, `alpha_2=30`, perfect correction, no postselection, complex128,
walls at `Nx/4` and `3Nx/4`, and root seed `2026090701` for new independent
trajectories. The GPU revision is `wall_pump_width_endpoints_s100_v1`.

The spectral workload is:

- 1,600 primary endpoint pairs, each containing separate CCW and CW paths;
- 3,200 primary raw directional paths;
- 150 optional GPU bridge pairs (300 paths), stored under `bridge_pump/`;
- 12 exact-control pairs; and
- 2,050 sensitivity pairs formed from five locked variants on 410 unique
  endpoint states.

The locked sensitivities are `M128`, `M512`, `seam_M256`, `radius3_M256`, and
`rank4_M256`. The selected endpoint set is IDs `0,4,...,96` in every cell,
plus the thirteen historical `M64 -> M128` classification-flip samples in the
`nshell=1`, `20 x 24` cell: soft IDs `4,8,34,59,69,80,98` and hard IDs
`3,6,31,85,86,87`.

## 0. Pre-run audit

From the repository root, run the maintained focused suite before uploading or
launching anything:

```bash
pytest -q \
  tests/test_wall_pump_width_endpoints_bundle.py \
  tests/test_colab_bundle_layout.py \
  tests/test_wall_diabatic_analysis.py \
  tests/test_wall_diabatic_spectral_pump.py \
  tests/test_wall_pump_endpoint_cpu_fallback.py \
  tests/test_occupied_frame_gpu.py \
  tests/test_slab_exterior_preparation.py \
  tests/test_b1_gpu_bundle.py
```

The deliberately skipped tests are multi-hour live M=128/256/512 endpoint
calculations and a real-CUDA parity check. They are not silently counted as
production evidence. The ordinary test suite covers exact task counts, source
identities, CPU/GPU frozen-record parity on small frames, atomic completion,
checkpoint restoration, control definitions, analysis outputs, and
notebook/source synchronization.

## 1. Generate the missing endpoints

Upload the directory

`00_WORKSPACE/CURRENT/final_production_new_designs/11_wall_pump_width_endpoints`

unchanged to

`MyDrive/final_production_new_designs/11_wall_pump_width_endpoints`, open
`run_wall_pump_width_endpoints.ipynb` on a 40-GB A100, and run top to bottom.
Use `BENCHMARK_ONLY=True` first. Production is allowed only when the measured
projection for the 1,000 missing primary endpoints is at most 40 hours and the
selected batch retains at least 8 GiB of GPU memory. The same output root is
resumable by five-trajectory result shards and five-cycle checkpoints.

After an accepted benchmark, run once with `REPORT_ONLY=True`. The report must
show 230 durable shards in the resolved table: 200 primary shards and 30
bridge shards. Then set `REPORT_ONLY=False`; use
`MAX_NEW_EXECUTION_BATCHES=1` for the first real batch if a bounded launch is
desired. Remove the limit only after the first checkpoint/result pair is
visible and verifies through a report-only restart.

The Drive output is

`MyDrive/classA_final_production_outputs/wall_pump_width_endpoints_s100_v1`.

After completion, copy that directory's contents without renaming subfolders
to

`imported_endpoints/wall_pump_width_endpoints_s100_v1`

inside this pilot directory. The strict loader verifies every shard against
the locked GPU revision, configuration hash, source hashes, dtype, backend,
wall flags, sample IDs, byte count, and SHA-256. Do not rename `endpoints/`,
`bridge/`, protocol, size, wall, or shard paths. The downstream configuration
addresses those exact relative names. Keep the A100 benchmark receipt with the
imported endpoint tree because it records the selected execution batching and
measured gate.

If the A100 benchmark is rejected, preserve its receipt and explicitly launch
the two-lane canonical-CPU fallback instead:

```bash
./launch_wall_pump_endpoint_cpu_fallback_tmux.sh /path/to/a100_endpoint_benchmark.json
```

The fallback supplies the 1,000 missing primary endpoints but intentionally
does not fabricate the independent 150-endpoint GPU bridge. It is a distinct
provenance route using `classA_U1FGTN.run_markov_circuit`. It runs two
28-worker NUMA-local lanes on cores `0--27` and `28--55`, writes the same
five-sample shard interface, and uses canonical state/RNG checkpoints every
five cycles. Never merge GPU and fallback files into one source tree.

## 2. Validate and run the spectral calculation

Inspect the exact controls without computation:

```bash
python -u run_wall_diabatic_controls_and_sensitivities.py report-controls
python -u run_wall_diabatic_spectral_pump_s100.py report
```

Launch the local calculation on all 56 physical cores, numbered 0--55:

```bash
./launch_wall_diabatic_local_tmux.sh
```

The tmux entry point performs the stages in this order:

1. verify the saved exact flattened-Hamiltonian reference;
2. compute the new topological, trivial, conjugated-Chern, seam-gauge, and
   M=128/256/512 exact controls;
3. run or resume the 1,600 primary endpoint pairs, plus the 150 independent GPU
   bridge pairs when present;
4. run the preregistered mesh, seam, radius-three, and rank-four sensitivities;
5. validate the controls and write the raw CW/CCW analysis.

Each endpoint and both directions form one atomic NPZ/completion pair. The
primary calculation never pools the independent GPU bridge with the reused CPU
ensembles. Quantization of monitored endpoints is reported, not used as a
completion gate. Numerical validity and the exact/trivial/sign/gauge controls
are gates.

The exact controls are a separate symmetry-resolved specialization. They
rebuild the canonical OW `H_exact(phi)` at every flux, transform it into the
40 conserved `k_y` blocks, match each block's complete eigenbasis by maximum
overlap, polar-align degenerate clusters, and keep the initial occupation
labels. They never flatten only the zero-flux occupied projector and then
twist that surrogate. Their wall band contains neighboring momentum states,
so the monitored endpoint's global rank-two `Delta_ext >= 0.1` qualification
is not imposed on this exact positive control.

The primary flux path is
`phi_j=-sigma*1e-7+sigma*2*pi*j/256`. The runner transports the isolated edge
doublet, diagonalizes `R_R-R_L` within that doublet, preserves the entering
wall label through the avoided crossing, and rejoins it to the occupied
spectator projector. The fixed qualification thresholds are:

- external edge-to-complement gap at least `0.1` and larger than the internal
  edge splitting;
- wall-polarization eigenvalues at most `-0.8` and at least `+0.8`;
- radius-two combined wall weight at least `0.8`; and
- neighboring edge-cluster principal overlap at least `0.8`.

A miss is classified as an unresolved edge cluster and saved with
`resolved=False` plus its explicit `unresolved_reason`; it is not retried with
a different window or mesh. Raw CCW and CW `q_x`, `Delta N_L`, and
`Delta N_R` are stored and analyzed separately. No `q_x^odd` replacement is
used.

Set `WALL_PUMP_ENDPOINT_ROOT=/absolute/path` only when intentionally selecting
a different verified endpoint root. The standard A100 import is preferred;
the CPU-fallback root is selected automatically only when the standard import
is absent.

### Local spectral result layout

The base path for a primary task is

`<output>/pump/<protocol>/N<Nx>x24/<wall>/sample_NNN.npz`

with a sibling `sample_NNN.completion.json`. Optional bridge pairs use
`<output>/bridge_pump/`; controls use `<output>/controls/`; sensitivities use
`<output>/sensitivity/`; failed task diagnostics use `<output>/failures/`;
and per-task logs use `<output>/logs/tasks/`. The root
`campaign_identity.json` records the complete configuration and source hashes.

Every spectral NPZ contains both directions and 257-point primary histories
for flux, raw regional charges, `q_x`, density, internal/external gaps,
wall-polarization eigenvalues and weights, link quality, selected status,
projector/charge residuals, ordinary-overlap and instantaneous-refill controls,
source real-space Chern number, endpoint-defect eigenmodes/densities, multi-cut
charge, center displacement, and true forward/reverse undo diagnostics.

The analysis writes:

- `analysis/samplewise_raw_qx.csv`;
- `analysis/all_verified_raw_qx.csv`;
- `analysis/raw_directional_statistics.csv`;
- `analysis/raw_directional_curves.csv`;
- `analysis/analysis_summary.json`; and
- PDF plus 300-dpi PNG figures for raw paths, endpoint histograms, and
  numerical diagnostics under `analysis/figures/`.

The summary includes resolved/unresolved fractions, bootstrap confidence
intervals, the exponential internal-gap fit versus wall separation `W=Nx/2`,
source-Chern controls, all five sensitivity statuses, and the unpaired
backend-adjusted bridge comparison.

### Spectral interruption cost

Spectral output is completion-based at one endpoint pair. Verified pairs are
skipped on restart; a killed worker repeats only its current endpoint pair.
There is no spectral rolling checkpoint because the natural endpoint task is
the documented unit of work. A worker exception writes a diagnostic JSON and
does not create completion. The endpoint acquisition stage has the separate
five-cycle rolling checkpoint described above.

## 3. Optional spectral A100 benchmark

`benchmark_a100_spectral_eigensolver.py` is benchmark-only. It cannot change
the production backend. A future GPU spectral port requires complex128
eigenvalue and projector agreement within `1e-10` and measured throughput at
least twice a same-host 56-affinity-CPU reference. A small Colab CPU is not
accepted as that baseline. Until both conditions are recorded, the spectral
calculation remains local CPU work.

The full formalism and preregistered gates are in
`docs/wall_diabatic_spectral_pump_methods.tex` and its compiled PDF.

## 4. Final acceptance checklist

Do not call the campaign complete until all applicable boxes are satisfied:

- [x] A100 endpoint gate accepted at `<=40 h` with `>=8 GiB` headroom, or a
  rejected receipt is preserved and the canonical CPU fallback is used.
- [x] Exactly 1,000 missing primary endpoints verify; on the GPU route, all
  150 bridge endpoints also verify.
- [ ] `report` shows 1,600/1,600 verified primary spectral pairs and either
  1,750/1,750 GPU-route base pairs or 1,600/1,600 CPU-route base pairs.
- [ ] All 12 exact-control pairs verify. Topological `|q_x|>0.97`, trivial
  `|q_x|<0.05`, conjugated-Chern orientation reversal, M=128/256/512
  agreement below `1e-4`, seam/uniform agreement below `1e-8`, and true
  forward/reverse undo below `1e-8` all pass.
- [ ] All 2,050 sensitivity pairs verify; unresolved edge clusters remain
  explicitly listed rather than rewritten.
- [ ] Charge and projector residuals satisfy the locked `1e-10` gates, input
  Gram residuals satisfy `1e-8`, fixed rank is preserved, and no invalid
  completion pair enters analysis.
- [ ] Analysis contains separate raw CW and CCW samplewise distributions for
  every size, wall, and shell cell, with no bridge pooling.
- [ ] The final methods note is amended with dated result tables/figures only
  after the corresponding products verify. Quantization and width/shell trends
  are reported as outcomes, never retroactively turned into gates.

## 5. Recovery guide

- **Colab disconnect:** remount Drive and rerun the same notebook. An accepted
  benchmark and exact execution map are reused; verified shards skip; the
  matching execution-batch checkpoint resumes.
- **Partial or corrupt endpoint pair:** leave it in place for diagnosis and
  rerun. Verification marks it pending, and successful publication atomically
  replaces the stable product before writing completion last.
- **A100 gate rejected:** do not loosen the gate. Preserve the receipt and use
  the CPU fallback command in section 1.
- **Missing imported source:** copy the completed endpoint tree to the exact
  documented import root or set `WALL_PUMP_ENDPOINT_ROOT` intentionally. A
  report never treats an unavailable source as a completed task.
- **Spectral task failure:** inspect `failures/<task-id>.json` and the matching
  task log. Fix only numerical/implementation defects in a new revision if a
  scientific contract change is required; rerun unchanged tasks with
  `--resume` after an operational failure.
- **Unresolved edge cluster:** this is a valid measured diagnostic, not an
  operational failure. Do not change its window, rank, or threshold after
  seeing the result; use the preregistered sensitivity outputs.
