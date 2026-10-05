#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_wall_pump_endpoint_cpu_fallback.py"
CONFIG="${SCRIPT_DIR}/campaign_config.wall_pump_endpoint_cpu_fallback_v1.json"
OUTPUT="${SCRIPT_DIR}/imported_endpoints/wall_pump_width_endpoints_s100_v1_cpu_fallback"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u "${RUNNER}" report --config "${CONFIG}" --output-root "${OUTPUT}"

taskset -c 0-27 python -u "${RUNNER}" run \
  --config "${CONFIG}" --output-root "${OUTPUT}" --lane 0 --workers 28 &
LANE0_PID=$!
taskset -c 28-55 python -u "${RUNNER}" run \
  --config "${CONFIG}" --output-root "${OUTPUT}" --lane 1 --workers 28 &
LANE1_PID=$!

STATUS=0
wait "${LANE0_PID}" || STATUS=1
wait "${LANE1_PID}" || STATUS=1
python -u "${RUNNER}" report --config "${CONFIG}" --output-root "${OUTPUT}"
exit "${STATUS}"
