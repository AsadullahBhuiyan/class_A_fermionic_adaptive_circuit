#!/bin/bash
set -euo pipefail
JOB_ID=$1

echo "=== Job ${JOB_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 - <<'EOF'
import sys, os, json
import torch
import numpy as np

job_id = int(os.environ.get("JOB_ID", sys.argv[1] if len(sys.argv) > 1 else "0"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Job {job_id} | device: {device} | GPU: {torch.cuda.get_device_name(0)}")

# Load this job's parameters
with open("params.json") as f:
    all_params = json.load(f)
p = all_params[job_id]
print(f"Params: {p}")

# Add classA_U1FGTN_gpu to path and run
sys.path.insert(0, ".")
from classA_U1FGTN_gpu import classA_U1FGTN_gpu

circuit = classA_U1FGTN_gpu(
    Nx=p["Nx"],
    Ny=p["Ny"],
    nshell=p["nshell"],
    alpha_1=p.get("alpha_1", 1),
    alpha_2=p.get("alpha_2", 30),
    dw_truncation=p.get("dw_truncation", False),
    dtype=p.get("dtype", "complex128"),
    device="cuda",
)

results = circuit.run_markov_circuit(
    samples=p["samples"],
    cycles=p["cycles"],
    protocol=p.get("protocol", "perfect_correction"),
    init_mode=p.get("init_mode", "default"),
)

out_path = f"output_{job_id}.npz"
np.savez_compressed(out_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
print(f"Job {job_id} done. Saved {out_path}")
EOF
