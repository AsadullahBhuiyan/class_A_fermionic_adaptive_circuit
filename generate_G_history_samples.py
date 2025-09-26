# generate_G_history_samples.py
import os
import time
import importlib
import numpy as np

import classA_U1FGTN
importlib.reload(classA_U1FGTN)
from classA_U1FGTN import classA_U1FGTN


def format_interval(seconds: float) -> str:
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02}:{minutes:02}:{seconds:02}" if days else f"{hours:02}:{minutes:02}:{seconds:02}"


# ----------------------------
# Config (edit these)
# ----------------------------
Nx, Ny  = 24, 24
cycles  = 20
samples = 10
nshell  = 5                 # None for untruncated
DW      = True
backend = "loky"            # "loky" or "threading"
filling_frac = 0.5
seed_tag = None             # optional, becomes part of cache key

# Which caches to build:
GENERATE_DEFAULT     = True       # uses your random G0 (top not maximally mixed)
GENERATE_MAXMIX_TOP  = True       # uses G_tt = 0 at t=0 (entanglement suite)

# Parallelism
n_jobs = min(samples, os.cpu_count() or 1)

# ----------------------------
# Run
# ----------------------------
t0 = time.time()
artifacts = []

def run_one(init_kind: str):
    print(f"\n[run] Generating histories for init_kind='{init_kind}' ...")
    t1 = time.time()

    # Instantiate; __init__ will try to load, else generate & save FULL G.
    m = classA_U1FGTN(
        Nx, Ny,
        cycles=cycles,
        samples=samples,
        DW=DW,
        nshell=nshell,
        filling_frac=filling_frac,
        init_kind=init_kind,        # "default" or "maxmix_top"
        backend=backend,
        n_jobs=n_jobs,
        seed_tag=seed_tag,
        prompt_on_miss=False,       # don't pause; just generate on miss
    )

    # Touch data to ensure it’s loaded in memory (no generation here).
    full_hist = m.get_full_histories()
    S = len(full_hist)
    T = len(full_hist[0]) if S > 0 else 0
    Ntot = m.Ntot

    # Compute the cache path we just produced/used.
    cache_path = m._cache_path_Ghist(
        samples=samples,
        cycles=cycles,
        nshell=nshell,
        seed_tag=seed_tag,
        init_kind=init_kind,
    )
    exists = os.path.isfile(cache_path)
    print(f"[done] kind='{init_kind}' | {S} samples x {T} cycles | Ntot={Ntot} | "
          f"saved={'yes' if exists else 'no'} | elapsed={format_interval(time.time() - t1)}")

    artifacts.append((init_kind, cache_path, exists, S, T, Ntot))


if GENERATE_DEFAULT:
    run_one("default")

if GENERATE_MAXMIX_TOP:
    run_one("maxmix_top")

# ----------------------------
# Report
# ----------------------------
print("\nArtifacts produced:")
for (kind, path, ok, S, T, Ntot) in artifacts:
    status = "OK" if ok else "MISSING"
    print(f"  kind={kind:11s}  [{status:8s}]  S={S:3d}  T={T:3d}  Ntot={Ntot:5d}  ->  {os.path.abspath(path)}")

print(f"\nTotal time elapsed: {format_interval(time.time() - t0)}\n")