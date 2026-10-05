#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-state_projector_pump_N20_Ny28_36_s100_v3}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
SERIES_ROOT="${SERIES_ROOT:-${PROJECT_ROOT}/results/N20_state_projector_pump_Ny24_36_s100_series_v3}"
LOG="${LOG:-${SERIES_ROOT}/logs/tmux_state_projector_pump_size_series.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

printf '[launch] session=%s\n' "${SESSION}"
printf '[launch] Ny values=28,30,32,34,36; Nx=20; S=100 per wall and size; no central-charge calculation\n'
printf '[launch] cores=%s workers=%s BLAS_threads=%s\n' "${CPU_LIST}" "${WORKERS}" "${BLAS_THREADS}"
printf '[launch] series_root=%s\n' "${SERIES_ROOT}"
printf '[launch] log=%s\n' "${LOG}"
printf '[launch] attach: tmux attach -t %q\n' "${SESSION}"
printf '[launch] follow: tail -f %q\n' "${LOG}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "CPU_LIST=$(printf %q "${CPU_LIST}") WORKERS=$(printf %q "${WORKERS}") BLAS_THREADS=$(printf %q "${BLAS_THREADS}") SERIES_ROOT=$(printf %q "${SERIES_ROOT}") LOG=$(printf %q "${LOG}") $(printf %q "${PROJECT_ROOT}/state_projector_pump_size_series_tmux_entrypoint.sh")"

echo "[launch] started ${SESSION}"
