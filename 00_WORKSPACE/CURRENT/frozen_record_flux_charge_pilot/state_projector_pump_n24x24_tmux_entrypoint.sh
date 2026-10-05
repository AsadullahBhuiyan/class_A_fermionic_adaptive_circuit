#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
WAIT_SESSION="${WAIT_SESSION:-state_projector_pump_N20_Ny28_36_s100_v3}"
CONFIG="${PROJECT_ROOT}/campaign_config.state_projector_pump_n24x24_s100_v1.json"
OUTPUT="${PROJECT_ROOT}/results/N24x24_state_projector_pump_s100_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_state_projector_pump_n24x24_s100.log}"
PYTHON_BIN="${PYTHON:-/home/abhuiyan/anaconda3/bin/python}"

mkdir -p "$(dirname "${LOG}")"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

echo "[tmux] queued $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] waiting for ${WAIT_SESSION} to release cores ${CPU_LIST}" | tee -a "${LOG}"
while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 60
done

BASE=(taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_state_projector_pump_variants.py")
if command -v numactl >/dev/null 2>&1 && [[ "${CPU_LIST}" == "28-55" ]]; then
  BASE=(numactl --cpunodebind=1 --membind=1 "${BASE[@]}")
fi

echo "[tmux] started computation $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] Nx=24 Ny=24 S=100/wall nshell=1 walls=(6,18)" | tee -a "${LOG}"
echo "[tmux] phase=sample-0 soft/hard smoke" | tee -a "${LOG}"
"${BASE[@]}" all --resume --sample-ids 0 --config "${CONFIG}" --output-root "${OUTPUT}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"

echo "[tmux] phase=full S100 production" | tee -a "${LOG}"
"${BASE[@]}" all --resume --analyze --config "${CONFIG}" --output-root "${OUTPUT}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
