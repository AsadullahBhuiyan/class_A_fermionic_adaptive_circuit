# Cycle-30 clipped continuation of hard-wall alpha1=3 purification

This is a **numerically stabilized continuation**, not a new independent ensemble
and not an unchanged v1 trajectory. Nx=20, Ny=30, alpha1=3, alpha2=30, nshell=1,
100 samples, hard support truncation, full-system measurements, raster_y,
perfect correction, complex128, total horizon 60 cycles. The original seed
2026092521 and batch of 100 are retained. Completed alpha1=1 results stay in v1.

## Why this version exists

The unmodified v1 covariance of sample 70 developed occupation-bound excesses
8.72e-12, 1.96e-10, and 4.32e-9 at cycles 29,30,31. CPU and GPU independently
confirmed the last violation. Fast purification exposing an unstable direction
is a hypothesis, not an established physical cause. The successful older slab-only
campaign used the same canonical engine, but different preparation and measurements.

With the user's authorization, the canonical GPU engine now has an opt-in
`covariance_spectral_clip` flag (default false for all old callers). After each
complete cycle it Hermitian-diagonalizes G=2C-I and clips its eigenvalues to
[-1,1], equivalently clipping occupations to [0,1]. It adds only the spectral
correction; interior occupations are unchanged. This is not entrywise clipping,
flattening, or a switch to a pure-state projector. The updated covariance is
used for the next cycle, observer products and checkpoints.

The clipped output is separately versioned:
`hard_wall_full_measurement_nx20_ny30_alpha3_s100_2ny_clipped_v2`.
It must not overwrite or be silently pooled with the original ensemble.

## Read-only handoff and exact continuation

`V1_OUTPUT_ROOT` points to the original campaign. On the first production launch,
the runner validates the pinned v1 configuration/source identity, task, sample IDs,
cycle 30, byte count and SHA-256. It loads the original covariance, all RNG states,
and partial observer arrays without modifying the source files. The state is
projected once at handoff; the handoff correction and source checksum are stored
in `fork_provenance.json` and embedded in every final result. A verified checkpoint
is then published in the new output folder. Subsequent launches resume only v2.
Missing or incompatible v1 input is an error, not a fresh-run fallback.

Cycles 0..30 retain the original **unclipped** spectra/observations. Cycles 31..60
use cycle-end stabilization. `unclipped_prefix_cycles=30` distinguishes the two;
the handoff correction is separate from the historical cycle-30 observation.
No original alpha1=1 result is copied or recomputed. This creates twenty new
five-sample result/completion pairs for alpha1=3, with ten-cycle rolling checkpoints.

## Saved diagnostics and limitations

In addition to every-cycle spectra, entropy, charge, variance, record probabilities,
and final covariance, save per sample and cycle:

- `clip_pre_min`, `clip_pre_max`: occupation extrema before projection;
- `clip_max_correction`: largest absolute occupation correction;
- `clip_covariance_frobenius`: Frobenius norm of the spectral correction to G;
- `clip_charge_change`: signed occupation trace correction;
- `clip_mode_count`: number of clipped eigenvalues.

Unclipped-prefix corrections/counts are zero by protocol, not measurements of
what hypothetical earlier clipping would have done. Before-bound values for
that prefix are derived from its retained spectra. Handoff diagnostics live
separately in the provenance record. Large occupation corrections (>1e-6),
nonfinite states and Hermiticity violations (>1e-9) still stop visibly. The
observer retains its original 1e-9 physicality check after stabilization.

Clipping changes subsequent states and potentially subsequent Born outcomes.
Saved record probabilities are those actually sampled along the stabilized
path; do not interpret them as an exact unmodified Kraus-product likelihood
or infer that clipping fixes the underlying algorithm. Tiny-gap conclusions
require checking correction magnitudes and stabilization sensitivity.

Every cycle now includes an additional chunked eigendecomposition (chunk 10),
followed by the usual observer. This first implementation does not reuse observer
eigenvectors. Runtime overhead is unbenchmarked on A100; do not reuse the old
remaining-time estimate as a guarantee. No new covariance histories are saved.

## Run

Open `run_hard_wall_full_measurement_clipped.ipynb` in a fresh A100 40-GB runtime.
Keep the old Drive output folder. Review configuration and paths. `REPORT_ONLY`
does not evolve or copy state; the first production launch performs the checked
fork, then starts its cycle bar at 30/60. Rerun the same notebook after interruption.
Local /content staging, DriveFS temporary-copy/readback/checksum publication,
native tqdm, and the standard runtime-disconnect cell are retained.

Local synthetic/CPU-backed tests are necessary but not a substitute for confirming
that the real sample-70 continuation remains stable through cycle 60 on A100.
