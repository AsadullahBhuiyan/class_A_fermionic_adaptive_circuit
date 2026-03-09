# Dependencies

This project uses a small core scientific Python stack plus a few optional tools for notebooks and plotting workflows.

## Core runtime dependencies

Required for `src/fgtn/classA_U1FGTN.py` and most scripts in `scripts/`:

- `python` (3.10+ recommended)
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `joblib`
- `threadpoolctl`

## Notebook / interactive extras

Used by some notebooks and a few utility scripts:

- `jupyter` (or `jupyterlab`)
- `ipython`
- `pandas`
- `nbformat` (only needed if programmatically editing notebooks)

## Optional system/runtime tools

- `ffmpeg` (recommended for some animation export workflows)

## Install example (pip)

```bash
pip install numpy scipy matplotlib tqdm joblib threadpoolctl jupyter ipython pandas nbformat
```

## Install example (conda)

```bash
conda install numpy scipy matplotlib tqdm joblib threadpoolctl jupyter ipython pandas nbformat
```

## Notes

- Scripts pin BLAS/OpenMP threads via environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) for reproducible CPU behavior.
- No strict version pinning is enforced in-repo yet; if you want fully reproducible environments, add a pinned `requirements.txt` or `environment.yml`.
