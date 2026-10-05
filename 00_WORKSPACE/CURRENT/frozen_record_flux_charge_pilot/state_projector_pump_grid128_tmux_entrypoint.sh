#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_LIST="${CPU_LIST:-0-27}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
WAIT_SESSION="${WAIT_SESSION:-state_projector_pump_N20x24_dense_s100_v1}"
CONFIG="${PROJECT_ROOT}/campaign_config.state_projector_pump_n20x24_grid128_s100_v1.json"
OUTPUT="${PROJECT_ROOT}/results/N20x24_state_projector_pump_grid128_s100_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_state_projector_pump_grid128_s100.log}"
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
if command -v numactl >/dev/null 2>&1 && [[ "${CPU_LIST}" == "0-27" ]]; then
  BASE=(numactl --cpunodebind=0 --membind=0 "${BASE[@]}")
fi

echo "[tmux] started computation $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] Nx=20 Ny=24 reused S100 endpoints; flux intervals=128" | tee -a "${LOG}"
echo "[tmux] phase=sample-0 smoke" | tee -a "${LOG}"
"${BASE[@]}" all --resume --sample-ids 0 --config "${CONFIG}" --output-root "${OUTPUT}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"

echo "[tmux] phase=full S100 projector continuation" | tee -a "${LOG}"
"${BASE[@]}" all --resume --analyze --config "${CONFIG}" --output-root "${OUTPUT}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
