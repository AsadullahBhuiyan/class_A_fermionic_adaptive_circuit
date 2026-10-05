#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/campaign_config.fixed_flux_quench_s10_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/N16x20_frozen_record_fixed_flux_quench_s10_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
LOG="${LOG:-${OUTPUT_ROOT}/logs/tmux_fixed_flux_quench.log}"
PYTHON_BIN="${PYTHON:-/home/abhuiyan/anaconda3/bin/python}"

mkdir -p "$(dirname "${LOG}")"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

BASE=(
  taskset -c "${CPU_LIST}"
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_fixed_flux_quench.py"
)
COMMON=(
  --resume
  --config "${CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --workers "${WORKERS}"
)
if command -v numactl >/dev/null 2>&1 && [[ "${CPU_LIST}" == "28-55" ]]; then
  BASE=(numactl --cpunodebind=1 --membind=1 "${BASE[@]}")
fi

echo "[tmux] started $(date --iso-8601=seconds)"
echo "[tmux] CPU_LIST=${CPU_LIST} WORKERS=${WORKERS} BLAS_THREADS=${BLAS_THREADS}"
echo "[tmux] CONFIG=${CONFIG}"
echo "[tmux] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[tmux] phase=sample-0 structural smoke"
"${BASE[@]}" all "${COMMON[@]}" --sample-ids 0 2>&1 | tee -a "${LOG}"
echo "[tmux] phase=full S10 resume"
"${BASE[@]}" all "${COMMON[@]}" --analyze 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
