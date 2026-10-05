#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
SERIES_ROOT="${SERIES_ROOT:-${PROJECT_ROOT}/results/N20_state_projector_pump_Ny24_36_s100_series_v3}"
LOG="${LOG:-${SERIES_ROOT}/logs/tmux_state_projector_pump_size_series.log}"
PYTHON_BIN="${PYTHON:-/home/abhuiyan/anaconda3/bin/python}"
NY_VALUES=(28 30 32 34 36)

mkdir -p "$(dirname "${LOG}")"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

BASE=(taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/run_state_projector_pump_size_series.py")
if command -v numactl >/dev/null 2>&1 && [[ "${CPU_LIST}" == "28-55" ]]; then
  BASE=(numactl --cpunodebind=1 --membind=1 "${BASE[@]}")
fi

echo "[tmux] started $(date --iso-8601=seconds)" | tee -a "${LOG}"
echo "[tmux] Ny_VALUES=${NY_VALUES[*]} CPU_LIST=${CPU_LIST} WORKERS=${WORKERS} BLAS_THREADS=${BLAS_THREADS}" | tee -a "${LOG}"

echo "[tmux] phase=sample-0 smoke for every new circumference" | tee -a "${LOG}"
for ny in "${NY_VALUES[@]}"; do
  config="${PROJECT_ROOT}/campaign_config.state_projector_pump_n20x${ny}_s100_v3.json"
  output="${PROJECT_ROOT}/results/N20x${ny}_state_projector_pump_s100_v3"
  echo "[tmux] smoke Ny=${ny} config=${config} output=${output}" | tee -a "${LOG}"
  "${BASE[@]}" all --resume --sample-ids 0 --config "${config}" --output-root "${output}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
done

echo "[tmux] phase=full S100 size series" | tee -a "${LOG}"
for ny in "${NY_VALUES[@]}"; do
  config="${PROJECT_ROOT}/campaign_config.state_projector_pump_n20x${ny}_s100_v3.json"
  output="${PROJECT_ROOT}/results/N20x${ny}_state_projector_pump_s100_v3"
  echo "[tmux] production Ny=${ny}" | tee -a "${LOG}"
  "${BASE[@]}" all --resume --analyze --config "${config}" --output-root "${output}" --workers "${WORKERS}" 2>&1 | tee -a "${LOG}"
done

echo "[tmux] phase=cross-size analysis including completed Ny=24" | tee -a "${LOG}"
taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${PROJECT_ROOT}/analyze_state_projector_pump_across_sizes.py" 2>&1 | tee -a "${LOG}"
echo "[tmux] completed $(date --iso-8601=seconds)" | tee -a "${LOG}"
