# Perturbative Expansion Notes

This folder collects the single-Wannier-projector perturbative expansion notes and validation notebooks.

## Files

- `handwritten_perturbative_expansion_about_fixed _point.pdf`
  Handwritten derivation.

- `perturbative_fixed_point_validation_no_dw.ipynb`
  Checks the rank-1 fixed-point expansion for one Wannier projector:
  `delta G -> (I-P) delta G (I-P)`, including no-DW project-code validation.

- `raster_y_Q_product_wannier_sequence.ipynb`
  Builds the one-cycle product `L = product_i (I-P_i)` following
  `run_markov_circuit(..., sequence="raster_y")`, then analyzes the induced
  perturbation map `delta G -> L delta G L^dagger` for the requested no-DW and
  DW-truncated settings. It also plots the cycle-averaged log singular-value
  spectra `log sigma_i(L^C) / C` for `C = 20, 50, 100`.

- `pump_in_purity_check.ipynb`
  Shows algebraically and numerically that deterministic pump-in
  `G' = (I-P)G(I-P) + P` preserves `G^2 = I` when the input mode is already a
  measured vacuum mode, `GP = PG = -P`.

## Main Result

For a single rank-1 Wannier projector `P = |W><W|`, expanding the fixed branch
about a fixed point satisfying `G*P = PG* = sP` gives

```text
delta G' = (I-P) delta G (I-P)
```

independent of whether the fixed branch is occupied (`s=+1`) or unoccupied
(`s=-1`). For a raster sweep, the one-cycle linearized map is the ordered
composition of these factors.
