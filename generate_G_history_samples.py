#!/usr/bin/env python3
"""Minimal generator for classA_U1FGTN histories (S,T,Ntot,Ntot) with the new class API."""

import os
from pathlib import Path
import time


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

from classA_U1FGTN import classA_U1FGTN

# ---------------- config ----------------
Nx, Ny       = 16, 16
cycles       = 20
samples      = 500
nshell       = None          # None for no truncation
DW           = True
filling_frac = 0.5
backend      = "loky"
progress     = True
alpha_triv   = 30        # trivial mass profile

# The new class expects "default" or "maxmix" for init_mode
INIT_MODES = ["default"]

# -------------- helpers --------------
def hms(sec: float) -> str:
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"

def cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_mode) -> str:
    nsh = "None" if nshell is None else str(int(nshell))
    return (f"N{int(Nx)}x{int(Ny)}"
            f"_C{int(cycles)}"
            f"_S{int(samples)}"
            f"_nsh{nsh}"
            f"_DW{int(bool(DW))}"
            f"_init-{init_mode}")

# -------------- runner --------------
def run_one(init_mode: str) -> str:
    print(f"\n[run] init={init_mode} | N=({Nx},{Ny}) | cycles={cycles} | samples={samples} | nshell={nshell} | DW={DW} | alpha_triv={alpha_triv}")
    t0 = time.time()

    # Build model (no auto-generation)
    model = classA_U1FGTN(
        Nx=Nx, Ny=Ny,
        DW=DW,
        nshell=nshell,
        filling_frac=filling_frac,
        G0=None,
        alpha_triv=alpha_triv,
    )

    # Prepare projectors once
    model.construct_OW_projectors(nshell=nshell, DW=DW)

    # Parallel multi-sample run; ask for FULL history
    res = model.run_adaptive_circuit(
        G_history=True,
        cycles=cycles,
        samples=samples,
        n_jobs=None,          # let the class clamp based on env caps
        backend=backend,
        parallelize_samples=True,
        store="full",           # returns (S,T,Ntot,Ntot)
        init_mode=init_mode,    # "default" or "maxmix"
        progress=progress,
        save=True,
        save_suffix=f"_alpha_triv={alpha_triv}",
    )

    G_hist = res.get("G_hist")
    if G_hist is None or G_hist.ndim != 4:
        raise RuntimeError("run_adaptive_circuit did not return a full G_history array.")

    save_path = res.get("save_path")
    if not save_path:
        # Mirror the cache key when the class did not write to disk.
        save_path = f"cache/G_history_samples/{cache_key(Nx=Nx, Ny=Ny, cycles=cycles, samples=samples, nshell=nshell, DW=DW, init_mode=init_mode)}.npz"
    save_path = Path(save_path)

    S = res.get("samples", G_hist.shape[0])
    T = res.get("T", G_hist.shape[1])
    Ntot = model.Ntot
    del G_hist  # free memory before leaving

    print(f"[done] history saved to {save_path} | S={S}, T={T}, Ntot={Ntot} | elapsed={hms(time.time() - t0)}")
    return save_path


def main():
    t0 = time.time()
    paths = [run_one(mode) for mode in INIT_MODES]
    print("\nArtifacts:")
    for p in paths:
        print(f"  - {p.resolve()}")
    print(f"\nTotal elapsed: {hms(time.time() - t0)}")

if __name__ == "__main__":
    main()
