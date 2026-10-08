# B200 endpoint entropy and wall-contour campaign

Fresh 600-trajectory ensemble: Nx=20, Ny=24,28,32,40,50,60, S=100 per
size, endpoint T=2Ny. Hard support-truncated walls at x=5,15; alpha1=1,
alpha2=30, nshell=1, random-pure half-filled preparation, Born-conditioned
product exterior before cycle zero, slab-only raster-y measurements, perfect
correction, periodic geometry, and no postselection. Every dynamics segment
calls the canonical `classA_U1FGTN_gpu.run_markov_circuit`, complex128 native
occupied frames, with no covariance materialization.

## Retained products

Every five-sample NPZ retains only three float64 scientific arrays:

- `endpoint__contour_von_neumann_y0avg`: (5, Ny//2+1, 20, Ny//2).
- `endpoint__entropy_von_neumann`: (5, Ny//2+1).
- `endpoint__charge_variance`: (5, Ny//2+1).

Each trajectory averages all Ny periodic origins after independently computing
the strip observables; contour coordinates are relative dy. Width zero and
padding beyond Ay are exactly zero. Metadata binds sample identities, geometry,
cycle, configuration, seeds, and frozen sources. One trajectory is the
independent sampling unit. No finished native frames, time histories, spectra,
individual-origin contours, Renyi observables, or mean-charge curves are saved.
Temporary native-frame/RNG checkpoints are deleted only after result readback.

## Launch and resume

Use the existing `/blue/panhaining/ab2398.cornell/.conda/classA-b200/bin/python`
to run `submit_campaign.py`. `--dry-run` prints the dependency graph without
writing or submitting. `--output-root` selects a fresh campaign directory.
`--resume --output-root ROOT` uses frozen sources, validated benchmarks,
partitions, and seeds; it refuses concurrent campaign writers.

The dependency chain is GPU validation, six benchmark tasks capped at two B200s,
CPU schedule freeze, two independent B200 production workers, and CPU analysis.
Each GPU task requests 4 CPUs, 112 GiB host memory, and 24 hours. A wall-time
interruption resumes from the most recent five-cycle or completed-width checkpoint.
Logs are under `ROOT/logs/`. Benchmark/production progress is visible in tqdm.

Dynamics candidates are 10,20,25,50,75,100 trajectories; endpoint candidates are
8,16,32,64,128,256 matrix pairs. Fresh subprocesses isolate candidate allocators.
Safe candidates retain 16 GiB free GPU memory and stay below 85% reserved memory;
endpoint calibration additionally budgets worst-case production-frame residency.
Measured initialization, cycles, checkpoint publication, endpoint eigensolves,
and endpoint-progress publication determine the partition and worker assignment.
The scheduler compares every whole-size split with decreasing-cost batch
assignment, choosing the shorter makespan and whole-size assignment on ties.
Benchmark samples never enter production; batch partitions and seeds are frozen.

## Figures and validation

`analyze_campaign.py --output-root ROOT` verifies all 120 result/receipt pairs,
sample IDs 0..99 per size, widths, cycles, and contour closure. It produces two
stacked-panel figures under `ROOT/analysis/`, not in the manuscript bundle:
entropy/charge variance and two-column wall-contour weights at x={5,6}/{14,15}.
Both use the same trajectories, per-trajectory half-strip subtraction,
through-origin joint fits over 8<=Ay<=Ny/2, equal total weight per size, ordinary
trajectory SEMs, and full covariance between widths. Display excludes Ay=1
only. `R0_squared` is the uncentered through-origin statistic.

PDF/300-dpi PNG figures use the shared manuscript LaTeX typography helper;
font and clipping checks are mandatory. CSV curves, fit JSON, source/input
hashes, and typography records accompany them. `--data-only` verifies and fits
without rendering. Completion requires all outputs, not any target coefficient.

The CPU plotting job loads texlive/2023 and uses the separate
/blue/panhaining/ab2398.cornell/.conda/endpoint-pdf-tools Poppler environment
for PDF font verification; it does not change the GPU Python environment.

GPU validation compares independent NumPy dense endpoint calculations and tests
actual interruption after five cycles and two endpoint widths. Native state,
ranks, and RNG must resume exactly; endpoint parity and closure use 2e-8.
