# Hard-wall full-measurement purification at Ny=30

## Occupation-check recovery hotfix (2026-09-26)

The initial alpha1=3 run stopped after its cycle-30 checkpoint. The displayed
lower eigenvalue, -3.732e-13, was within the 1e-9 allowance; the upper eigenvalue
triggered the check but was rounded to `1.000e+00`. The lost runtime means its
precise value and the cause (eigensolver error versus actual covariance drift)
cannot be established from the traceback. This is a numerical-validation failure,
not evidence for a physical failure to purify.

The hotfix keeps the 1e-9 tolerance, unmodified covariance evolution, RNG,
scientific configuration, revision, and output folder. Only suspect eigenpairs
are independently recomputed with CPU LAPACK (`scipy.linalg.eigh`, driver `evr`).
If they pass the same bound, that observation uses the CPU eigenpairs. If the
violation is confirmed, the queue still stops: it prints full-precision extrema,
excess, cycle, and sample ID, and publishes a small `diagnostics/<task>/`
NPZ/JSON containing the offending unmodified sample covariance and spectrum.
No projection, larger tolerance, or silent acceptance of unphysical states is
introduced. Thus this is a safe recovery/diagnostic fix, **not yet a demonstrated
fix for the production numerical failure**. Genuine drift requires investigation
and a separately versioned numerical change, not repeated blind restarts.

Exact pre-hotfix source hashes are explicitly allowlisted in the runner. All
other configuration, sample, filename, byte-count, checksum, and shape checks
remain active. Completed original results stay unchanged; their recorded hashes
remain truthful. Continued checkpoints/results record the updated sources, with
the accepted prior source identity documented by `PRE_HOTFIX_SOURCE_HASHES`.
The canonical engine and helper are unchanged. This diagnostic-only compatibility
does not permit resuming a different campaign or engine version.

Use a fresh A100 runtime and run `REPORT_ONLY=True` first: expect 20 completed
shards and one alpha1=3 checkpoint at cycle 30, provided those saved files have
not changed. Then set `REPORT_ONLY=False` and rerun config/run cells. Do not erase
outputs or restart alpha1=1. A CPU-confirmed failure preserves the good checkpoint
and should be diagnosed from the new report before another attempt.

Run `run_hard_wall_full_measurement_purification.ipynb` on an A100 40-GB-class
runtime. Copy this whole bundle folder to
`MyDrive/final_production_new_designs/21_hard_wall_full_measurement_purification/`.
Open the notebook in Colab and run in order. No Drive API authentication is
required beyond the standard mount. Upload is not part of the local build.

## Scientific contract

- Nx=20, Ny=30, 100 independent Born trajectories per alpha1=1 and alpha1=3.
- Hard/support-truncated walls at x=5,15; alpha2=30, nshell=1, X trial orbitals.
- **meas_slab_only=False**: every cycle visits all 600 physical unit cells.
  Cycle zero is globally maximally mixed on 1,200 physical modes. The canonical
  hard-wall exterior product-state preparation is NOT performed.
- Canonical `classA_U1FGTN_gpu.run_markov_circuit`, perfect correction,
  no postselection, raster_y, n_a=0.5, complex128; endpoint T=2Ny=60.
- Revision `hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1`,
  root seed 2026092521. Alpha-specific deterministic seeds; old outputs are
  neither overwritten, imported, nor pooled with these new ensembles.

This changes the preparation and measurement protocol, not just the observer.
It is not a shorter resume of campaigns 07 or 20. Hard support truncation
remains on even though measurement sites cover the full system.

## Saved observables (trajectory first; no sample averaging during acquisition)

Both alphas save raw ascending occupation eigenvalues at every cycle 0..60,
with shape `(5,61,1200)` per five-sample shard. Also save scalar total entropy,
charge, intrinsic charge variance, numerical residuals, and every-cycle Born
log probabilities/event counts. Raw occupations are not clipped; the entropy
kernel alone retains the prior 1e-12 regularization, so the near-zero entropy
tail is an estimator floor. Spectra are of the full physical system, not a
half-system restriction and not the spectrum of an averaged covariance.

Alpha1=1 additionally saves:

- Entropy and intrinsic charge-variance contours `(5,61,20,30)`, summing both
  orbitals in each cell, from a shared eigendecomposition at each cycle.
- One **final-cycle** slow-mode vector `(5,1200)` (complex128), its orbital-
  summed density `(5,20,30)`, occupation, sorted-spectrum index, signed rate,
  absolute rate, residual and minimum-magnitude multiplicity.

The finite-time rate is `lambda_j=log((1-nu_j)/nu_j)/(2T)`. Selection minimizes
its ABSOLUTE value, not the signed rate, matching the earlier purification
figure. Only `1e-9 < nu < 1-1e-9` modes are eligible. If none survive,
`slow_mode_resolved=False`, index=-1, and mode/rate products are NaN. A tie
within absolute/relative rate tolerances 1e-10 is reported in
`slow_mode_min_abs_multiplicity`; the stored eigensolver vector is then one
nonunique representative, not a uniquely resolved physical mode. Choose only
resolved multiplicity-one modes for the earlier single-mode-density estimator.
The phase is fixed by making the largest-magnitude component real positive.
The physical row index is `2*Nx*y + 2*x + orbital`.

Alpha1=3 uses `eigvalsh` and does not compute or save spatial contours or
eigenvectors. Both alphas retain final centered covariance `G_final=2C-I`,
shape `(5,1200,1200)`, so alternate endpoint analyses remain possible.
No covariance/eigenvector histories are saved.

## Execution and resume

One 100-sample dynamics batch per alpha (unchanged Ny=30 batching); observer
chunks of ten. Forty immutable five-sample NPZ/completion-JSON result pairs
in alpha-specific directories. Outer shard and inner physical-cycle progress
are displayed directly in the notebook kernel. The endpoint eigensolver also
shows progress. No shell stdout/buffer forwarding is used.

After every ten cycles, publish a rolling covariance/RNG/partial-observer
checkpoint pair, using local `/content` staging, DriveFS temporary-copy and
SHA-256 readback, atomic rename, and JSON last. The final checkpoint is kept
through endpoint mode extraction and all result publication. Restarting after
an endpoint interruption repeats only the endpoint extraction/publication,
not the dynamics. Completed pairs are verified before skipping. An interruption
between replacement of the two checkpoint files can leave an invalid pair;
that batch must then rerun deterministically. This is the simple DriveFS
contract, not a claim of independent cloud-server verification.

GPU budget is 38 decimal GB device usage and at most 35 decimal GB PyTorch
allocator residency, leaving CUDA-library workspace. A100 peak usage and speed
still need measurement for this full-measurement protocol. Do not silently
change batch composition on OOM; it is part of the ensemble identity.
The notebook exposes REPORT_ONLY and MAX_NEW_EXECUTION_BATCHES. The default
runs both alphas in sequence; setting the latter to 1 runs the next pending
batch only. Rerun the same notebook to continue.

Final arrays are about 4.9 GB uncompressed, dominated by the two final covariance
ensembles; allow roughly 10 GB on Drive while replacing the current ~2.5 GB
checkpoint. No multi-GB intermediate history is kept. Local scratch is cleaned
only after that batch's complete results pass readback verification.

## Rebuild and tests

Run `python build_notebook.py` here to regenerate the notebook, visible config,
and deployment checksums. Canonical GPU sources are bundled byte-identically;
the old campaigns are not runtime dependencies. The local CPU-backed miniature
tests validate formulas and exact checkpoint continuation, not A100 throughput.
No production simulation has been launched by building this notebook.
