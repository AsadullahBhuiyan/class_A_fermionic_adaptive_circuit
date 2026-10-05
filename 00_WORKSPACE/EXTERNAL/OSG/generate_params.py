"""
Run this locally before submitting to OSG.
Generates params.json — one entry per job index ($(Process)).

Edit the sweep variables below to match your campaign, then:
    python3 generate_params.py
    # check params.json looks right, then:
    condor_submit batch_gpu.submit
"""

import json
import itertools

# ── Small system sweep (colab_small_system_testing) ──────────────────────────
small_system_params = []
for Ny, nshell in itertools.product([30, 40], [1, 2]):
    small_system_params.append({
        "Nx": 16,
        "Ny": Ny,
        "nshell": nshell,
        "alpha_1": 1,
        "alpha_2": 30,
        "dw_truncation": True,
        "protocol": "perfect_correction",
        "init_mode": "default",
        "dtype": "complex128",
        "samples": 10,
        "cycles": 50,
    })

# ── N=20 entanglement scaling sweep (colab_large_entanglement_scaling_N20) ───
n20_params = []
for Ny in [30, 40, 50, 60, 80, 100, 120]:
    n20_params.append({
        "Nx": 20,
        "Ny": Ny,
        "nshell": 1,
        "alpha_1": 1,
        "alpha_2": 30,
        "dw_truncation": True,
        "protocol": "perfect_correction",
        "init_mode": "default",
        "dtype": "complex128",
        "samples": 100,
        "cycles": 50,
    })

# ── Choose which campaign to submit ──────────────────────────────────────────
# Swap in n20_params or combine as needed.
params = small_system_params   # 4 jobs
# params = n20_params          # 7 jobs
# params = small_system_params + n20_params  # 11 jobs

with open("params.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"Wrote {len(params)} job configs to params.json")
print("Update 'queue N' in batch_gpu.submit to match:", len(params))
for i, p in enumerate(params):
    print(f"  [{i}] Nx={p['Nx']} Ny={p['Ny']} nshell={p['nshell']} "
          f"samples={p['samples']} cycles={p['cycles']}")
