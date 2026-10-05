#!/bin/bash
# Wrapper run on each OSG node. Installs missing packages, then dispatches
# to the correct Python script based on $1.
set -euo pipefail

SCRIPT_NAME=$1
JOB_ID=$2

echo "=== OSG job wrapper: ${SCRIPT_NAME} job_id=${JOB_ID} ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash install_packages.sh

export PYTHONPATH=".:${PYTHONPATH:-}"
mkdir -p output logs

case "${SCRIPT_NAME}" in
    slope_vs_system_size)
        python3 run_slope_vs_system_size.py --job-id "${JOB_ID}" --output-dir ./output
        ;;
    slope_vs_cycle_block)
        python3 run_slope_vs_cycle_block.py --job-id "${JOB_ID}" --output-dir ./output
        ;;
    *)
        echo "Unknown script: ${SCRIPT_NAME}"
        exit 1
        ;;
esac

echo "=== Done: ${SCRIPT_NAME} job_id=${JOB_ID} at $(date) ==="

echo "=== Done: ${SCRIPT_NAME} job_id=${JOB_ID} ==="
