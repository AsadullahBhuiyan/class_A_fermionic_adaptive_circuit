#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$1"
OUTPUT_ROOT="$2"
CAMPAIGN_ID="$3"
CPU_LIST="$4"
WORKERS="$5"
BLAS_THREADS="$6"
LOG_PATH="$7"
shift 7

export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"
export PYTHONUNBUFFERED=1

taskset -c "${CPU_LIST}" python -u "${HERE}/run_campaign.py" all \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --campaign-id "${CAMPAIGN_ID}" \
  --cpu-list "${CPU_LIST}" \
  --workers "${WORKERS}" \
  --blas-threads "${BLAS_THREADS}" \
  "$@" 2>&1 | tee -a "${LOG_PATH}"
