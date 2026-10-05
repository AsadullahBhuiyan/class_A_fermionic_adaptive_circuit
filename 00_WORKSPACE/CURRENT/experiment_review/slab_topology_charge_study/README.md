# Offline slab topology and charge study

This study consumes the completed square and rectangular hard-wall Chern
campaigns without changing their data, Colab bundles, or the technical report.
The cohorts are independent (100 trajectories per geometry, 600 altogether).
`config.json` fixes the analysis contract. `results/v1/` is the canonical result
directory; `results/setup_attempt_01/` preserves setup metadata from an initial
input-schema adjustment and contains no scientific results.

## Run and resume

From this directory:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -m pytest -q test_study.py
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -u run_study.py --phase benchmark
tmux new-session -d -s slab-topology-charge-v1 'bash /home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/slab_topology_charge_study/run_local.sh'
```

The launcher runs the batch analysis and then the report. `results/v1/run.log`
contains progress and errors. Repeating the launcher checksums and skips completed
batch products. There is one analysis NPZ/completion JSON per original input
batch. Source/config changes require a new output directory; incomplete or
corrupt pairs are recomputed. There are no Drive APIs or dynamics calls.
The run uses eight BLAS/OpenMP threads and processes endpoint frames sequentially.
Reference projectors are cached once per geometry; full trajectory covariance
matrices are never saved again.

## Scientific conventions

- The spectral projector is `Gamma = (V V†)^T`. All nonlinear observables are
  evaluated on each trajectory before averaging. Means and sample SEMs use
  `ddof=1`; spatial centers and time steps are not independent trajectories.
- Slab boundaries are inclusive: (5,15), (8,22), and (10,30) for Nx20,30,40.
  Exterior occupations are measured once in the canonical orbital basis and
  stay fixed during slab-only evolution. Endpoint product-state/decoupling
  checks must pass before reconstructing slab charge at earlier cycles.
- Regional half filling is half that region's number of orbitals. The variance
  decomposition retains `2 Cov(Q_slab,Q_exterior)`.
- The radius scan uses every integer `2 <= R < distance_to_wall`, all integer y
  centers, and periodic minimum-image coordinates. R4 is the fixed-radius
  comparison; R=0.2Nx is the original production radius.
- Correlations are sums of squared projector entries over both orbital indices,
  averaged over admissible starting positions. The primary core excludes
  columns at distance <=2 from either wall; buffers1 and3 are sensitivity checks.
  Transverse pairs have both endpoints inside that core, with no new slab PBC.
- Reference OW modes are built with the canonical CPU constructor on the
  original geometry, then restricted to active centers/rows. The signed OW
  parent is diagonalized for nsh1 and dense modes. Its occupied projector uses
  the same transpose convention. The frozen exterior is deliberately excluded
  from this equilibrium-reference comparison.
- A cutoff-degenerate cluster is excluded/included in full to give explicitly
  labeled `below`/`above` sensitivity projectors; these are not falsely described
  as unique half-filled ground states. Expected particle/hole mismatch and the
  squared-projector-distance map are not automatically physical defect counts.

## Products and interpretation

`input_inventory.json` binds all38 raw result/completion pairs and campaign
identities. `benchmark.json` records one measured endpoint per cohort/geometry.
`references/` includes spectra, projectors, observables, and particle/hole and
product-state controls. `batches/` preserves every trajectory's radius-by-center
values, regional correlations, reference mismatch maps/counts, and charge and
Chern histories. The original sample IDs and cohort names remain attached.

`tables/` contains charge statistics, the variance budget, Chern quantiles and
radius statistics, correlation profiles, reference mismatch summaries, and
paired trajectory comparisons. Spearman associations are descriptive; no
significance threshold or fit is used. `figures/` contains vector PDFs, 300-dpi
PNGs, and explanatory captions. `findings.txt` reports measured results and
limitations; `summary.json` marks completion only after all38 batches and600
trajectories are present and verified.

Square endpoint frames are cycle40. Rectangular endpoint frames are cycles40,
60,80; only saved scalar observations permit a common cycle40 or cycles21–40
comparison. The code does not manufacture missing frame histories or pool the
two independent20x20 ensembles. No scaling exponent, mobility-gap proof, or
theorem for arbitrary monitored trajectories is inferred from this finite study.

The scientific test suite covers periodic-sector seams, explicit versus
factorized Chern contractions, projector mismatch identities, cutoff degeneracy,
trajectory-first SEMs, covariance accounting, invalid exterior inference, and
completion-based restart/corruption handling. Real endpoint validation additionally
checks frame orthonormality, saved-observer agreement, and frozen exterior state.
