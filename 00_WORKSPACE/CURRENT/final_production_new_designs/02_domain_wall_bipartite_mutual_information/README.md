# Domain-wall bipartite mutual-information sweep

This independent Colab/A100 campaign measures trajectory-resolved bipartite
mutual information between two opposite, full-`x` strips in dynamical
domain-wall states. It is the fixed-`Nx` analogue of the paper's quarter-width
strip diagnostic. This v2 runner can import checksum-verified trajectory values
from the original five-sample v1 output without modifying that output.

## Locked scientific contract

- `Nx = 20`, `Ny = 20, 24, 28`, and strip width `w = Ny // 4 = 5, 6, 7`
- `nshell = 1`, `alpha_2 = 30`, and 21 nonuniform `alpha_1` values swept from
  `3` down to `1`, concentrated symmetrically around the expected critical
  point `alpha_1 = 2`
- hard/support-truncated and soft/untruncated domain-wall constructions
- 100 independent trajectories per `(Ny, wall, alpha_1)` case
- random pure half-filled initialization, perfect correction, no postselection,
  and raster-`y` site order
- exactly `2*Ny` cycles, with the covariance materialized in memory only for the
  endpoint observer
- `complex128` arithmetic on an NVIDIA A100 with 40-GB-class memory

For each trajectory and translation `y0`, the regions are

```text
A = [0,Nx) x [y0,y0+w)
B = [0,Nx) x [y0+Ny/2,y0+Ny/2+w)
```

with periodic wrapping in `y` and both orbitals retained. The observer computes
`I(A:B) = S(A) + S(B) - S(A union B)` before either the translation or
trajectory average. Only `Ny/2` placements are evaluated because the other half
exchanges `A` and `B` exactly.

There are 126 cases, 12,600 trajectories, and 420 durable macro-tasks. The
locked GPU batch sizes are 50 trajectories for `Ny=20` and 25 for `Ny=24,28`.
Each macro produces one compact NPZ plus one completion JSON. The completion
JSON binds the task/configuration identity, global sample indices, current
source hashes, imported v1 provenance, byte count, and SHA-256 checksum.

### Batching design note

The original five-trajectory calls used only about 1 GB of a 40-GB A100 and an
observed `Ny=20` cycle took about 4.5 seconds, implying more than 150 A100-hours
for the full task table before larger-size and endpoint costs. The size-aware
exception here follows the completed bulk-validation batching scale. Its raw
occupied frames are approximately balanced at no more than 251 MB per macro,
and its largest endpoint covariance is approximately 512 MB before workspaces.
This favors GPU throughput while retaining a bounded completion-based restart
unit. There is deliberately no intra-task rolling checkpoint: a disconnect
reruns only the active macro.

### v1-compatible resume

The v1 and v2 output roots are separate. For every pending v2 macro, the runner
verifies the constituent v1 five-sample result/completion pairs. Valid v1
trajectory values are copied unchanged into the self-contained v2 macro; all
missing indices are evolved together in one deterministic GPU call. That call's
seed is derived from the v2 macro identity and its exact missing-index set. The
historical source hashes recorded by each imported completion are retained as
provenance; they are not compared to the redesigned v2 source hashes. Corrupt,
partial, or identity-mismatched v1 pairs are ignored and recomputed. Neither
root should have another writer while this notebook is running.

## Run on Colab

1. Upload this folder to
   `MyDrive/final_production_new_designs/02_domain_wall_bipartite_mutual_information`.
2. Open `run_domain_wall_bipartite_mutual_information.ipynb` in an A100 runtime.
3. Review the single configuration cell and run the notebook from the top.

The read-only legacy directory ends in
`domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v1`; the new output ends in
`domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25`.
Set `REPORT_ONLY = True` to inspect v2 completion and reusable v1 coverage
without writing or starting GPU work. Set `MAX_NEW_TASKS` to a positive integer
to bound a Colab session. Reruns verify and skip completed v2 macros.

## Result format

Each task NPZ retains one value per trajectory for:

- `mutual_information_y0avg`
- `entropy_a_y0avg`
- `entropy_b_y0avg`
- `entropy_union_y0avg`

It also records geometry, parameters, sample indices, macro/computation seeds,
per-sample v1/v2 origin, endpoint cycle, and numerical diagnostics. Imported v1
task IDs and result checksums are retained as provenance. Raw covariance
matrices and individual-`y0` values are not saved.

After every task verifies, the runner writes a consolidated NPZ with trajectory
arrays of shape `(2, 3, 21, 100)` in `(wall, Ny, alpha_1, sample)` order. It also
writes a two-panel hard/soft comparison as vector PDF and 300-dpi PNG. The plot
contains sample means only; the consolidated trajectory values support later
uncertainty analysis.

## Local hard-wall analysis figures

The repository snapshot contains the completed hard-wall lane: 210 verified
result/completion pairs and 6,300 trajectories. The soft-wall lane is not yet
local, so `make_hard_wall_figures.py` deliberately plots only the sampled hard
wall at `nshell=1`; it does not overlay flattened-ground-state benchmark data.
It reuses the benchmark figure grammar and verifies all 420 downloaded files
against `DOWNLOAD_MANIFEST.json` before averaging.

Validate without writing derived products:

```bash
python make_hard_wall_figures.py --check-only
```

Generate the vector PDF, 300-dpi PNG, geometry schematic, and compact mean/SEM
CSV under `analysis_outputs/hard_wall_nshell1/`:

```bash
python make_hard_wall_figures.py
```

The MI figure displays the conventional increasing horizontal axis even though
the simulation grid was evaluated from `3` down to `1`. Error bars are one
standard error of the mean over `S=100` independent trajectories. Mutual
information is formed trajectory by trajectory and translation by translation
before either average.
