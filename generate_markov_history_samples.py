#!/usr/bin/env python3
"""Driver for Markovian top-layer covariance histories."""

import os


def _int_env(name, default=None):
    """Parse an environment variable as int, returning default on failure."""
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Cap threaded math libraries before importing NumPy/SciPy.
_CPU_LIMIT = (
    _int_env("SLURM_CPUS_PER_TASK")
    or _int_env("MY_CPU_COUNT")
    or (os.cpu_count() or 1)
)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[var] = str(_CPU_LIMIT)
os.environ.setdefault("MY_CPU_COUNT", str(_CPU_LIMIT))

try:
    os.sched_setaffinity(0, set(range(_CPU_LIMIT)))
    print(f"[info] CPU affinity pinned to {_CPU_LIMIT} cores.")
except Exception as e:
    print(f"[warn] Could not set CPU affinity: {e}")

import time
import numpy as np

from classA_U1FGTN import classA_U1FGTN

# ---------------- config ----------------
Nx, Ny        = 8, 8
cycles        = 20
SAMPLE_COUNTS = [10000]
nshell        = None
DW            = True
filling_frac  = 0.5
backend       = "loky"   # safer in constrained environments
progress      = True
n_a           = 0.5
alpha_triv    = 30        # trivial mass profile
INIT_MODES    = ["default"]


# -------------- runner --------------
def run_one(init_mode: str, samples: int):
    print(f"\n[run] init={init_mode} | N=({Nx},{Ny}) | cycles={cycles} | samples={samples} | nshell={nshell} | DW={DW} | alpha_triv={alpha_triv}")
    t0 = time.time()
    
    Nlayer = int(2*Nx*Ny)

    model = classA_U1FGTN(
        Nx=Nx,
        Ny=Ny,
        DW=DW,
        nshell=nshell,
        filling_frac=filling_frac,
        G0=None,
        alpha_triv=alpha_triv,

    )

    model.construct_OW_projectors(nshell=nshell, DW=DW)

    if init_mode == "default":
        G_init = model._build_initial_covariance()
    elif init_mode == "maxmix":
        G_init = np.zeros((Nlayer, Nlayer)) + 1e-8*np.eye(Nlayer)
    elif init_mode == "rerun":
        # Load from previous run
        data = np.load("cache/G_history_samples/N16x16_C20_S500_nshNone_DW1_init-default_n_a0.5_markov_circuit_alpha_triv=30.npz", allow_pickle=True)
        G_hist = data[data.files[0]]
        G_init = G_hist[-1, -1, :, :]  # last sample, last cycle
        print('[info] Loaded G_init from previous run.')
    else:
        raise ValueError("init_mode must be 'default' or 'maxmix'.")
    model.G0 = np.array(G_init, dtype=np.complex128, copy=True)

    n_jobs = _CPU_LIMIT
    res = model.run_markov_circuit(
        G_history=True,
        cycles=cycles,
        samples=samples,
        n_jobs=n_jobs,
        backend=backend,
        parallelize_samples=True,
        init_mode=init_mode,
        G_init=G_init,
        progress=progress,
        n_a=n_a,
        save=True,
        save_suffix=f"_alpha_triv={alpha_triv}_rerun",
        save_final_G_only=False
    )

    save_path = res.get("save_path")
    S, T, Nlayer, _ = res["G_hist"].shape
    print(f"[done] samples={S} | cycles={T} | Nlayer={Nlayer} | elapsed={time.time() - t0:0.2f}s | alpha_triv={alpha_triv}")
    if save_path:
        print(f"      saved history: {save_path}")

    if init_mode == "default":
        model.plot_corr_y_profiles(
            res["G_hist"],
            save=True,
            save_suffix="_markov_circuit_alpha_triv={alpha_triv}_rer",
        )
    elif init_mode == "maxmix":
        model.entanglement_contour_suite(
            res["G_hist"],
            save=True,
            save_suffix="_markov_circuit_alpha_triv={alpha_triv}",
        )
    elif init_mode == "rerun":
        model.plot_corr_y_profiles(
            res["G_hist"],
            save=True,
            save_suffix="_markov_circuit_alpha_triv={alpha_triv}_rerun",
        )

def main():
    t0 = time.time()
    for samples in SAMPLE_COUNTS:
        for mode in INIT_MODES:
            print(f"samples = {samples} | init_mode = {mode}")
            run_one(mode, samples)
    print(f"\nTotal elapsed: {time.time() - t0:0.2f}s")


if __name__ == "__main__":
    main()
