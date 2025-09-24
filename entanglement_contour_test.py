# entanglement_contour_test.py
import os
import time
import importlib
import numpy as np
from matplotlib import pyplot as plt

import classA_U1FGTN
importlib.reload(classA_U1FGTN)
from classA_U1FGTN import classA_U1FGTN


def format_interval(seconds: float) -> str:
    """Convert seconds to H:MM:SS (or D:HH:MM:SS if >1 day)."""
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02}:{minutes:02}:{seconds:02}" if days else f"{hours:02}:{minutes:02}:{seconds:02}"


# ----------------------------
# Config
# ----------------------------
Nx, Ny   = 16, 16
cycles   = 10
samples  = 10
nshell   = 5
DW       = True
backend  = "loky"   # or "threading"

t0 = time.time()

# ----------------------------
# Build model
# ----------------------------
m = classA_U1FGTN(Nx, Ny, cycles=cycles, samples=samples, DW=DW, nshell=nshell)

# ----------------------------
# Unified entanglement contour suite
# ----------------------------
print("\n[run] entanglement_contour_suite ...")
t1 = time.time()
artifacts = m.entanglement_contour_suite(
    samples=samples,
    n_jobs=min(samples, os.cpu_count() or 1),
    backend=backend,
    save_data=True,
    filename_profiles=None,
    filename_prefix_dyn=None,
)
t1e = time.time() - t1

# ----------------------------
# Report
# ----------------------------
print("\nArtifacts produced:")
for k, v in artifacts.items():
    if v is not None:
        print(f"  {k:20s}: {os.path.abspath(v)}")

print(f"\nTotal time elapsed: {format_interval(time.time() - t0)}")