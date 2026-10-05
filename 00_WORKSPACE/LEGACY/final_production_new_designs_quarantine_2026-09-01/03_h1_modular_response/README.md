# 03_h1_modular_response

Standalone, versioned H1 campaign for trajectory-resolved retarded modular response and
supplemental localized-packet drift. It preserves the older H1 implementation,
`03_chirality_replay`, and `02_wall_cft_windows`; outputs are written only to
`MyDrive/classA_final_production_outputs/production_25sample_h1_modular_response_v1/03_h1_modular_response`.

The immutable matrix contains four cases: hard (`dw_truncation=true`,
`meas_slab_only=true`) and soft (both false) walls, crossed with
`alpha_1=1,3`. Every case uses `Nx=20`, `Ny=40`, `alpha_2=30`, `n_shell=1`,
25 independent Haar-random half-filled Slater trajectories in five shards of five,
perfect correction, Born sampling, random serial order, `n_a=0.5`, `complex128`, and
80 cycles. Observations occur at cycles `40,48,56,64,72,80`.

At each trajectory-checkpoint, the lower half-cylinder is traced out and one modular
Hamiltonian is constructed on `[0,20) x [20,40)`. Ten distinct retained cells are freshly
and deterministically selected on each wall. The same source coordinates are used across
all four physical cases, but the trajectory RNG streams remain independent. The observer
computes the signed response to `-i[P_source,C_A]` and reuses the same eigensystem for the
supplemental equal-orbital packet. Modular Hamiltonians are never averaged.

Static susceptibility is deliberately absent. Full covariance histories, eigenvectors,
entropy, Chern, Bott, tangent, and CFT-window data are also absent. Each archive contains
source-resolved primary-cutoff response fields, compact multi-cutoff summaries, packet
products, RNG state, and a compact exact-replay Born record.

## Running

Open `run_production_bundle.ipynb` in an A100 Colab runtime after uploading the complete
campaign package. The production queue runs or reuses the version-matched A100
qualification automatically before its first missing shard; the optional
`RUN_A100_PREFLIGHT=True` cell runs it early for inspection. The measured five-trajectory
soft shard must stay below 80% of A100 memory, 1 GiB per shard, and a 12 GiB projected
active output. Unsafe qualification stops production without changing any scientific
parameter.

The notebook otherwise starts in checksum-verification/report-only mode. After reviewing
the exact 20-shard queue, set `RESUME_REPORT_ONLY=False`. Enable `RUN_ANALYSIS` only after
all archives verify. The analysis writes PDF/300-dpi PNG figures, CSV summaries, Fourier
diagnostics, and a two-column RevTeX result table.

Local read-only queue inspection:

```bash
python run_bundle.py --drive-root /path/to/drive --list-cases-json
```

Targeted tests:

```bash
pytest -q tests/test_h1_modular_response_bundle.py
```
