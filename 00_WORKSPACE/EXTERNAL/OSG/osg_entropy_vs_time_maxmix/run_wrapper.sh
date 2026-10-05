#!/bin/bash
set -euo pipefail
JOB_ID=$1

echo "=== Job ${JOB_ID} starting on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash install_packages.sh

export PYTHONPATH=".:${PYTHONPATH:-}"
mkdir -p output

python3 run_entropy_vs_time.py --job-id "${JOB_ID}" --output-dir ./output

echo "=== Job ${JOB_ID} done at $(date) ==="
