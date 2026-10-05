#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/campaign_config.state_projector_pump_n20x24_s100_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/N20x24_state_projector_pump_s100_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-frozen_record_fixed_flux_quench_N16x20_s10_v1}"
LOG="${LOG:-${OUTPUT_ROOT}/logs/tmux_state_projector_pump_s100.log}"
PYTHON_BIN="${PYTHON:-/home/abhuiyan/anaconda3/bin/python}"

mkdir -p "$(dirname "${LOG}")"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

BASE=(taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_state_projector_pump_s100.py")
if command -v numactl >/dev/null 2>&1 && [[ "${CPU_LIST}" == "28-55" ]]; then
  BASE=(numactl --cpunodebind=1 --membind=1 "${BASE[@]}")
fi
COMMON=(--resume --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" --workers "${WORKERS}")

echo "[tmux] started $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] CPU_LIST=${CPU_LIST} WORKERS=${WORKERS} BLAS_THREADS=${BLAS_THREADS}" | tee -a "${LOG}"
echo "[tmux] CONFIG=${CONFIG}" | tee -a "${LOG}"
echo "[tmux] OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${LOG}"
if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    echo "[tmux] waiting for ${WAIT_FOR_SESSION} to release cores ${CPU_LIST}" | tee -a "${LOG}"
    sleep 30
  done
fi
echo "[tmux] phase=sample-0 burn-in and both static paths" | tee -a "${LOG}"
"${BASE[@]}" all "${COMMON[@]}" --sample-ids 0 2>&1 | tee -a "${LOG}"
echo "[tmux] phase=full S100 resume and analysis" | tee -a "${LOG}"
"${BASE[@]}" all "${COMMON[@]}" --analyze 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
