# Three-lane hard-wall endpoint alpha sweep

Nx=20, Ny=30, T=60; **90 additional Born trajectories for each of 21 alphas**.
The current notebooks extend the completed ten samples to 100 per alpha:
original IDs 0--9 remain unchanged, and the new run produces IDs 10--99.
`meas_slab_only=True` in both ensembles, not False. Do not change this flag
when extending the ensemble. All other scientific settings remain unchanged.
Hard/support-truncated walls at x=5,15; alpha2=30, nshell=1, trial orbitals X,
maximally mixed with canonical Born-conditioned exterior preparation, then
slab-only raster-y measurements, perfect correction, no postselection.
Canonical `classA_U1FGTN_gpu.run_markov_circuit`, complex128 covariance path.
No evolving covariance spectral clipping. The initial state is not maximally
mixed in the exterior after its canonical preparation.

## Three simultaneous lanes

| Notebook | Alpha values (execution order is nearest 2 first) |
|---|---|
| `run_alpha_endpoint_lane_A.ipynb` | 1, 1.6, 1.85, 1.975, 2.05, 2.2, 2.6 |
| `run_alpha_endpoint_lane_B.ipynb` | 1.2, 1.7, 1.9, 2, 2.1, 2.3, 2.8 |
| `run_alpha_endpoint_lane_C.ipynb` | 1.4, 1.8, 1.95, 2.025, 2.15, 2.4, 3 |

Put the complete folder under `MyDrive/final_production_new_designs/` and open
each notebook on a **different A100 40-GB-class runtime**. Concurrent runtime
availability depends on your Colab account. Never launch the same lane twice.
Ninety trajectories are resident per alpha, under a 38-GiB PyTorch allocator cap
(not a guarantee about CUDA allocations outside that allocator).
Each lane has seven execution batches and 126 five-sample result shards;
all lanes together give 1,890 NEW samples and 378 result/completion pairs.
There is no automatic batch resizing: changing batch composition changes the
random-number stream. If an actual A100 memory failure occurs, preserve the
checkpoint and revise the execution contract explicitly, rather than silently
reassigning sample IDs or RNG streams.

The full configuration is visible in one cell. `REPORT_ONLY=True` verifies
inventory without simulation; `MAX_NEW_EXECUTION_BATCHES=1` limits a first run.
Defaults run the full lane. Code runs under `/content`; stdout, stderr, task
starts, resume inventory, cycle progress, memory and measured ETA are visible.
The final cell disconnects the runtime. No automatic cross-lane analysis runs.

Revision: `hard_wall_alpha21_nx20_ny30_add90_t60_endpoint_modes_v2`.
Root seed: `2026100131`. Output:
`MyDrive/classA_final_production_outputs/hard_wall_alpha21_nx20_ny30_add90_t60_endpoint_modes_v2`.
Task IDs include alpha; deterministic batch seeds do not depend on lane.
The 90-sample batch composition and sample range are part of the sampling
identity, yielding new seeds disjoint from the original 21 batch seeds.
The original S10 configuration is preserved in `original_s10_config.json`;
`default_config()` also retains that historical contract for analysis.
The add90 collection and original S10 collection can be explicitly combined
to S100 after verification; no old file, receipt, or checkpoint is rewritten.
The Colab run does not need access to the old S10 output folder.

## Endpoint products

At cycle 60 only, diagonalize the Hermitized active centered covariance with
660 modes. Save each sample's raw sorted centered spectrum a, occupations
nu=(1+a)/2, signed rates lambda=-atanh(a)/60, and gap=min(abs(lambda)).
The minimum is taken **per sample**, never after spectral averaging.
These are occupation-derived finite-time rates, not the tangent/Choi spectrum.

Centered modes within 1e-9 of either endpoint are numerically capped, with
infinite signed rates. Raw values are preserved; bound excess above 1e-9 fails.
This is a centered tolerance, not an occupation tolerance. Each sample stores:

- `gap_raw`: finite gap or +inf when all modes are capped.
- `gap_value`: actual finite gap, or **100** for an infinite gap.
- `gap_is_infinite`: authoritative flag; never infer it by comparing to 100.
- `gap_mode_valid`, `gap_mode_index`, `gap_mode_vector`, matching occupation,
  signed rate, residual, cap counts, and full-spectrum tie mask/count.
- Active indices and `(x,y,orbital)` coordinates. Full-space index is
  `2*(y*Nx+x)+orbital`. Sum squared amplitudes over orbitals for cell probability.

The vector's largest component is real and nonnegative. Ties within 1e-12 in
absolute rate select the first ascending-spectrum index. Degenerate modes are
basis-dependent; tied-mode indices flag this ambiguity. The selected mode's
absolute rate can differ from the exact minimum by at most that tie tolerance.
All-capped samples use mode_valid=False, index=-1, and a zero vector/scalar
padding; **that padding is not a physical mode**. Infinite under this numerical
cap rule does not prove an infinite physical gap. NaNs and failed solvers are
errors, never placeholder results.

## Resume and durability

Rolling covariance plus NumPy/Torch CPU/all CUDA RNG checkpoints are published
every ten cycles, including 60. Restart restores RNG immediately before the
next canonical call and skips repeated exterior preparation. The final
checkpoint survives eigensolve/publication failure until all 18 five-sample
results verify. No histories or final covariance archives are retained.

Each checkpoint/result uses local scratch, a Drive temporary copy, byte/SHA-256
readback, atomic replacement, and completion JSON last. Source/config identities
must match. Invalid or partial pairs are pending; corrupt checkpoints restart
the batch deterministically. DriveFS readback is **not independent cloud-server
verification**. No Drive API, leases, generations or dashboards are used.
Allow 8 GiB free on local and mounted filesystems. One full rolling covariance
is about 2.07 GB uncompressed per lane; temporary replacement needs additional space.
Only current checkpoints persist; finished outputs are compact spectra/modes.

## Analysis and timing

After downloading all three lanes, run:

```bash
python analyze_campaign.py --config campaign_config.json --output-root /path/to/downloaded/add90_collection --destination /path/to/add90_analysis
```

All 378 add90 pairs must verify. Omit `--config` to analyze the original 42-pair
S10 collection; do not point this S10 mode at add90 data. These invocations
analyze each collection separately, not the combined S100 ensemble.
CSVs contain raw gaps, placeholder flags, finite-sample
counts, means and ordinary SD/sqrt(n) SEM (no bootstrap). Infinite placeholders
never enter statistics. Mixed configurations are labeled finite-subset estimates,
not ensemble means. No SEM is reported for fewer than two finite samples.
The PDF/PNG uses a symlog y axis to retain zero and placeholder 100, with distinct
markers for full ensembles, finite subsets and saturation. No fits are made.

The completed S10 sweep measured 775--777 seconds of dynamics per ten-sample
alpha batch. Linear scaling to 90 samples gives about **1.94 hours per alpha
and 13.6 hours per lane, plus checkpoint/setup overhead**. This is a planning
baseline, not a measured 90-sample runtime: larger resident batches can improve
throughput, but an optimal batch or speedup has not been benchmarked on A100.
The previous ten-sample peak was about 3.35 GiB, suggesting roughly 30 GiB
under linear scaling to 90; the enforced allocator ceiling remains 38 GiB.
At linear timing, each ten-cycle checkpoint bounds lost dynamics to about
20 minutes. Three simultaneous A100s reduce wall time, not total GPU-hours.
The runner prints
measured remaining time after each batch. No upload or production launch is
part of repository installation.

## Repository validation

### October 1 endpoint-ordering hotfix

The original GPU launch completed dynamics but could abort at endpoint
extraction with `Spectra must be in ascending centered-eigenvalue order`.
This is an extractor ordering bug, not evidence of failed purification or
exhausted GPU memory. The extractor now stably sorts the returned eigenvalues
and matching eigenvector columns together before computing spectral products.
Physical bounds, cap tolerance, residual checks and all scientific parameters
remain unchanged. Tests explicitly permute solver eigenpairs, including
near-degenerate cap modes, to exercise the previously untested failure path.

For historical compatibility, the runner accepts the exact pre-sort and
completed-S10 source tuples pinned in `PRE_SORT_SOURCE_HASHES` and
`S10_SOURCE_HASHES`, only when both engine/helper hashes still match.
Configuration, task, seed, checksum and numerical validations remain required.
Already completed shards keep their original bytes and provenance; saved
cycle-60 checkpoints resume at endpoint extraction without rerunning dynamics.
No receipt is rewritten to pretend old data used new source. Unknown source
changes remain incompatible. The notebooks, output revision and sampling
contract do not change for this eigenpair-ordering repair.

Run `python -m pytest -q tests/test_alpha_endpoint_ny30.py` from the repository
root. Coverage includes the locked grid/lane partition, full synthetic analysis,
caps/placeholder statistics, complex gap eigenmodes, every checkpoint boundary,
final-checkpoint/publication crash windows, corruption/readback/atomic-replace
failures, notebook streaming, runtime enforcement and source/manifest identity.
Small engine tests exercise the GPU class's CPU-backed covariance path; these
are validation fixtures, not production data. A real A100 timing/memory launch
is still required to verify the runtime estimate on Colab.
