#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_wall_diabatic_spectral_pump_s100.py"
CONTROL_RUNNER="${SCRIPT_DIR}/run_wall_diabatic_controls_and_sensitivities.py"
CONFIG="${SCRIPT_DIR}/campaign_config.wall_diabatic_spectral_pump_s100_v1.json"
OUTPUT="${SCRIPT_DIR}/results/N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
A100_ENDPOINTS="${SCRIPT_DIR}/imported_endpoints/wall_pump_width_endpoints_s100_v1"
CPU_ENDPOINTS="${SCRIPT_DIR}/imported_endpoints/wall_pump_width_endpoints_s100_v1_cpu_fallback"

if [[ -n "${WALL_PUMP_ENDPOINT_ROOT:-}" ]]; then
  NEW_ENDPOINTS="$(realpath -m "${WALL_PUMP_ENDPOINT_ROOT}")"
elif [[ -d "${A100_ENDPOINTS}" ]]; then
  NEW_ENDPOINTS="${A100_ENDPOINTS}"
elif [[ -d "${CPU_ENDPOINTS}" ]]; then
  NEW_ENDPOINTS="${CPU_ENDPOINTS}"
else
  NEW_ENDPOINTS="${A100_ENDPOINTS}"
fi

if [[ "${NEW_ENDPOINTS}" == "${CPU_ENDPOINTS}" ]]; then
  BRIDGE_ARGUMENT="--no-include-bridge"
  EXPECTED_BASE_PAIRS=1600
else
  BRIDGE_ARGUMENT="--include-bridge"
  EXPECTED_BASE_PAIRS=1750
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ ! -f "${RUNNER}" || ! -f "${CONTROL_RUNNER}" || ! -f "${CONFIG}" ]]; then
  echo "Core runner/config is not installed; refusing to launch the fallback." >&2
  exit 2
fi

echo "[endpoint source] ${NEW_ENDPOINTS}"
echo "[base contract] ${EXPECTED_BASE_PAIRS} pairs (${BRIDGE_ARGUMENT})"
python -u "${SCRIPT_DIR}/validate_wall_diabatic_controls.py"
taskset -c 0-55 python -u "${CONTROL_RUNNER}" controls --config "${CONFIG}" --output-root "${OUTPUT}" --new-endpoint-root "${NEW_ENDPOINTS}" --workers 56 --resume
taskset -c 0-55 python -u "${RUNNER}" report --config "${CONFIG}" --output-root "${OUTPUT}" --new-endpoint-root "${NEW_ENDPOINTS}" "${BRIDGE_ARGUMENT}"
taskset -c 0-55 python -u "${RUNNER}" run --config "${CONFIG}" --output-root "${OUTPUT}" --new-endpoint-root "${NEW_ENDPOINTS}" "${BRIDGE_ARGUMENT}" --workers 56 --resume
taskset -c 0-55 python -u "${RUNNER}" report --config "${CONFIG}" --output-root "${OUTPUT}" --new-endpoint-root "${NEW_ENDPOINTS}" "${BRIDGE_ARGUMENT}"
taskset -c 0-55 python -u "${CONTROL_RUNNER}" sensitivity --config "${CONFIG}" --output-root "${OUTPUT}" --new-endpoint-root "${NEW_ENDPOINTS}" --workers 56 --resume
python -u "${SCRIPT_DIR}/validate_wall_diabatic_controls.py" --output-root "${OUTPUT}"
python -u "${SCRIPT_DIR}/analyze_wall_diabatic_width_sweep.py" --output-root "${OUTPUT}" --expected-pairs 1600 --expected-base-pairs "${EXPECTED_BASE_PAIRS}"
