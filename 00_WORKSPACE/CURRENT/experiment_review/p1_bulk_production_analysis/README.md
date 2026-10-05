# P1 bulk production analysis

`P1_bulk_production_analysis.ipynb` is the editable, local CPU notebook for the frozen `production_25sample_v1` P1 matrix. It verifies all 240 archive/receipt pairs in memory, analyzes trajectory-resolved observables, and produces the legacy Fig. 1(c,d) candidates plus seven diagnostics, including comprehensive cycle-resolved Chern plots, a logarithmic $|C_G-1|$ view over $L\leq s_{\rm cycle}\leq2L$, and a compact reference-style convergence panel with a logarithmic inset.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/p1_bulk_production_analysis/build_notebook.py --execute
```

Set `P1_CPU_RANGE` to control CPU affinity and `P1_ARCHIVE_ROOT` to point to a different verified copy of the frozen `01_bulk_width_gate` directory. Generated products are written to `outputs/production_25sample_v1/`. The frozen production tree is read-only input and must not be modified.
