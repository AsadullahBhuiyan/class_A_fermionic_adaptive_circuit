#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"

{
  echo "[start] $(date --iso-8601=seconds)"
  echo "[contract] raster_y, full 40-cycle pure occupied/empty cocycle"
  taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${HERE}/run_pure_tangent_flux.py" \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --cpu-list "${CPU_LIST}" \
    --workers "${WORKERS}" \
    --blas-threads "${BLAS_THREADS}"
  taskset -c "${CPU_LIST}" "${PYTHON_BIN}" -u "${HERE}/analyze_pure_tangent_flux.py" \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}"
  echo "[complete] $(date --iso-8601=seconds)"
} 2>&1 | tee -a "${LOG_PATH}"
