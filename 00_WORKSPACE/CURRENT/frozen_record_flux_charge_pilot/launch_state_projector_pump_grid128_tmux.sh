#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-state_projector_pump_N20x24_grid128_s100_v1}"
CPU_LIST="${CPU_LIST:-0-27}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
WAIT_SESSION="${WAIT_SESSION:-state_projector_pump_N20x24_dense_s100_v1}"
OUTPUT="${PROJECT_ROOT}/results/N20x24_state_projector_pump_grid128_s100_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_state_projector_pump_grid128_s100.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

printf '[launch] session=%s\n' "${SESSION}"
printf '[launch] Nx=20 Ny=24; reuse 200 verified endpoints; 128 flux intervals\n'
printf '[launch] cores=%s workers=%s; waits for=%s\n' "${CPU_LIST}" "${WORKERS}" "${WAIT_SESSION}"
printf '[launch] output=%s\n' "${OUTPUT}"
printf '[launch] log=%s\n' "${LOG}"
if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "CPU_LIST=$(printf %q "${CPU_LIST}") WORKERS=$(printf %q "${WORKERS}") BLAS_THREADS=$(printf %q "${BLAS_THREADS}") WAIT_SESSION=$(printf %q "${WAIT_SESSION}") LOG=$(printf %q "${LOG}") $(printf %q "${PROJECT_ROOT}/state_projector_pump_grid128_tmux_entrypoint.sh")"
echo "[launch] queued ${SESSION}"
