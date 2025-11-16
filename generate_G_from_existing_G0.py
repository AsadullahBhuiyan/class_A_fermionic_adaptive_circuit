#!/usr/bin/env python3
'''Run the adaptive circuit starting from a cached trajectory snapshot.'''

import time
from pathlib import Path

import numpy as np

from classA_U1FGTN import classA_U1FGTN

# ---------------- configuration ----------------
Nx, Ny = 16, 16
cycles = 40
samples = 100
nshell = None
DW = True
filling_frac = 0.5
backend = "loky"
progress = True

OUT_DIR = Path("cache/G_history_samples")
SOURCE_PATH = OUT_DIR / "N16x16_C20_S100_nshNone_DW1_init-maxmix.npz"
SOURCE_SAMPLE_INDEX = 0          # which trajectory to use
SOURCE_CYCLE_INDEX = -1          # which time slice; -1 means final snapshot

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def hms(sec: float) -> str:
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02}:{m:02}:{s:02}"

def cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_label: str) -> str:
    nsh = "None" if nshell is None else str(int(nshell))
    return (f"N{int(Nx)}x{int(Ny)}"
            f"_C{int(cycles)}"
            f"_S{int(samples)}"
            f"_nsh{nsh}"
            f"_DW{int(bool(DW))}"
            f"_init-{init_label}")

def load_initial_snapshot(path: Path, sample_idx: int, cycle_idx: int):
    if not path.exists():
        raise FileNotFoundError(f"Cached history file not found: {path}")
    data = np.load(path)
    if "G_full" not in data:
        raise KeyError(f"Expected 'G_full' array in {path}")
    G_hist = np.asarray(data["G_full"], dtype=np.complex128)
    if G_hist.ndim != 4:
        raise ValueError(f"Unexpected cached shape {G_hist.shape}; expected (S,T,N,N)")

    S, T, N, M = G_hist.shape
    if N != M:
        raise ValueError(f"Non-square covariance blocks in cache: {G_hist.shape}")
    if sample_idx < 0:
        sample_idx += S
    if cycle_idx < 0:
        cycle_idx += T
    if not (0 <= sample_idx < S):
        raise IndexError(f"sample_idx={sample_idx} out of range (S={S})")
    if not (0 <= cycle_idx < T):
        raise IndexError(f"cycle_idx={cycle_idx} out of range (T={T})")

    return G_hist[sample_idx, cycle_idx], sample_idx, cycle_idx

def main():
    print("Loading cached trajectory...", flush=True)
    G_init, resolved_sample, resolved_cycle = load_initial_snapshot(
        SOURCE_PATH, SOURCE_SAMPLE_INDEX, SOURCE_CYCLE_INDEX
    )
    Ntot = G_init.shape[0]

    print(
        f"Loaded snapshot (sample={resolved_sample}, cycle={resolved_cycle}) with dimension {Ntot}",
        flush=True,
    )

    model = classA_U1FGTN(
        Nx=Nx,
        Ny=Ny,
        DW=DW,
        nshell=nshell,
        filling_frac=filling_frac,
        G0=G_init,
    )

    model.construct_OW_projectors(nshell=nshell, DW=DW)

    t0 = time.time()
    print(f"Running adaptive circuit for {cycles} cycles across {samples} samples...", flush=True)

    res = model.run_adaptive_circuit(
        G_history=True,
        cycles=cycles,
        samples=samples,
        n_jobs=25,
        backend=backend,
        parallelize_samples=True,
        store="full",
        init_mode="default",
        progress=progress,
    )

    if res is None or "G_hist" not in res:
        raise RuntimeError("run_adaptive_circuit did not return history data")

    G_hist = np.asarray(res["G_hist"], dtype=np.complex128)
    if G_hist.ndim != 4:
        raise RuntimeError(f"Unexpected output shape {G_hist.shape}; expected (S,T,N,N)")

    init_label = f"default_fromcache-rerun-s{resolved_sample}-t{resolved_cycle}"
    out_path = OUT_DIR / f"{cache_key(Nx=Nx, Ny=Ny, cycles=cycles, samples=samples, nshell=nshell, DW=DW, init_label=init_label)}.npz"

    np.savez_compressed(
        out_path,
        G_full=G_hist,
        source_file=np.array([str(SOURCE_PATH)], dtype='<U256'),
        source_sample=np.array([resolved_sample], dtype=np.int64),
        source_cycle=np.array([resolved_cycle], dtype=np.int64),
    )

    elapsed = time.time() - t0
    S, T, N, _ = G_hist.shape
    print(f"Wrote {out_path} | S={S}, T={T}, N={N} | elapsed={hms(elapsed)}", flush=True)

if __name__ == "__main__":
    main()
