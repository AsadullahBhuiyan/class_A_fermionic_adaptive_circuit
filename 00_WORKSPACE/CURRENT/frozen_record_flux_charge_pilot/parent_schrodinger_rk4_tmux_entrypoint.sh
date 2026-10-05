#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
CONFIG="${PROJECT_ROOT}/campaign_config.parent_schrodinger_rk4_n20x24_s50_tau1e4_v1.json"
OUTPUT="${PROJECT_ROOT}/results/N20x24_parent_schrodinger_rk4_s50_tau1e4_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_parent_schrodinger_rk4_s50.log}"
PYTHON_BIN="${PYTHON:-/home/abhuiyan/anaconda3/bin/python}"

mkdir -p "$(dirname "${LOG}")"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

BASE=(taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_parent_schrodinger_rk4_s50.py")
if command -v numactl >/dev/null 2>&1; then
  BASE=(numactl --cpunodebind=1 --membind=1 "${BASE[@]}")
fi

echo "[tmux] started $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] T=10000; 50 endpoint states; CW+CCW; RK4; cores=${CPU_LIST}" | tee -a "${LOG}"
"${BASE[@]}" report --config "${CONFIG}" --output-root "${OUTPUT}" 2>&1 | tee -a "${LOG}"
"${BASE[@]}" run --resume --analyze --config "${CONFIG}" --output-root "${OUTPUT}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
