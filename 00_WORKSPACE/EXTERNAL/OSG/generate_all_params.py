"""
Run locally before submitting to OSG.
Generates params.json — one entry per job index ($(Process)).

Based on real A100 timings (10 samples each, cycles=Ny/2):
  Ny=40: 20min, Ny=60: 60min, Ny=80: 150min, Ny=100: 360min, Ny=120: 750min
Scaling: t ~ Ny^3.29

Usage:
    python generate_all_params.py
    # check output, then:
    condor_submit all_system_sizes.submit
"""
import json
import math
import numpy as np

# ── Timing fit ────────────────────────────────────────────────────────────────
Ny_measured = np.array([40, 60, 80, 100, 120])
t_measured  = np.array([20, 60, 150, 360, 750])   # minutes for 10 samples
alpha, c    = np.polyfit(np.log(Ny_measured), np.log(t_measured), 1)

# ── Campaign settings ─────────────────────────────────────────────────────────
NY_VALUES     = [40, 60, 80, 100, 120, 140, 160, 180, 200]
TOTAL_SAMPLES = 100
TARGET_H      = 15     # max hours per job

# ── Build params list ─────────────────────────────────────────────────────────
params = []
for Ny in NY_VALUES:
    t_per_sample = np.exp(c) * Ny**alpha / 10      # minutes
    max_per_job  = max(1, min(int(TARGET_H * 60 / t_per_sample), TOTAL_SAMPLES))
    start = 0
    while start < TOTAL_SAMPLES:
        stop = min(TOTAL_SAMPLES, start + max_per_job)
        params.append({
            "Ny":          Ny,
            "block_start": start,
            "block_stop":  stop,
            "cycles":      Ny // 2,
            "fit_cycle":   Ny // 2,
        })
        start = stop

with open("params.json", "w") as f:
    json.dump(params, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Total jobs: {len(params)}")
print(f"Update 'queue N' in all_system_sizes.submit to: {len(params)}\n")

jobs_per_ny = {}
for p in params:
    jobs_per_ny.setdefault(p["Ny"], []).append(p)

for Ny, jobs in jobs_per_ny.items():
    t_per_sample = np.exp(c) * Ny**alpha / 10
    max_s = max(j["block_stop"] - j["block_start"] for j in jobs)
    print(f"  Ny={Ny:3d}: {len(jobs):2d} job(s), "
          f"up to {max_s} samples/job, "
          f"~{max_s * t_per_sample / 60:.1f}h/job")
