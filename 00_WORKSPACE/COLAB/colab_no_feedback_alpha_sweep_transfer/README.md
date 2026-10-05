# Colab Alpha Sweep Transfer

Colab-ready GPU bundle for the DW-truncated alpha sweep with both no-feedback and perfect-correction variants. The data-generation artifact is the notebook under `notebooks/data_generation/`, matching the structure used by `colab_regularized_choi_transfer_matrix`.

## Files

- `notebooks/data_generation/run_no_feedback_alpha_sweep_dwtrunc1_N20_Ny20-30-40_S10_cycles2Ny_gpu.ipynb`: production Colab notebook.
- `src/classA_U1FGTN_gpu.py`: bundled GPU dynamics engine, synced from `src/fgtn/classA_U1FGTN_gpu.py`.
- `src/choi_transfer_observables_gpu.py`: bundled GPU Choi transfer observer.

## Colab Usage

Upload the whole `colab_no_feedback_alpha_sweep_transfer` folder to Google Drive, open the notebook in `notebooks/data_generation/`, and run it on an A100 40 GB runtime.

The production campaign contains 54 configurations: `3 Ny values * 9 alpha values * 2 correction modes`.

Default production settings are `Nx=20`, `Ny=20,30,40`, ten samples per configuration, the standard nine-point alpha grid, `dwtrunc1`, and two correction modes: `correction_label=none` with `perfect_correction=False`, and `correction_label=perfect` with `perfect_correction=True`. Both modes keep `p_gain=0` and `p_loss=0`.

For a small GPU smoke campaign, edit the campaign configuration cell to:

```python
NY_VALUES = [4]
ALPHA_VALUES = [1.0]
SAMPLES = 2
BATCH_SIZE = 1
CAMPAIGN_ID = "smoke_dwtrunc1_gpu"
OVERWRITE = True
```


Each per-sample NPZ stores the final-cycle transfer spectrum plus all finite final-cycle Lyapunov/transfer modes in the reduced Choi slab basis:

```text
finite_lyapunov_mode_indices
finite_lyapunov_exponents
finite_lyapunov_a_eigenvalues
finite_lyapunov_modes
finite_lyapunov_mode_count
lyapunov_mode_basis_indices
```

`finite_lyapunov_modes[:, k]` is the mode vector for entry `k` of the finite-mode arrays.

Outputs are written under:

```text
colab_no_feedback_alpha_sweep_transfer/gpu_data/alpha_sweep_transfer_dwtrunc1_no_feedback_and_perfect_correction/
```
