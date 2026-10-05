#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-state_projector_pump_N20x24_dense_s100_v1}"
CPU_LIST="${CPU_LIST:-0-27}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
OUTPUT="${PROJECT_ROOT}/results/N20x24_state_projector_pump_s100_dense_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_state_projector_pump_dense_s100.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

printf '[launch] session=%s\n' "${SESSION}"
printf '[launch] Nx=20 Ny=24; S=100 per wall; nshell=infinity (canonical None/dense)\n'
printf '[launch] cores=%s workers=%s BLAS_threads=%s\n' "${CPU_LIST}" "${WORKERS}" "${BLAS_THREADS}"
printf '[launch] output=%s\n' "${OUTPUT}"
printf '[launch] log=%s\n' "${LOG}"
printf '[launch] attach: tmux attach -t %q\n' "${SESSION}"
printf '[launch] follow: tail -f %q\n' "${LOG}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "CPU_LIST=$(printf %q "${CPU_LIST}") WORKERS=$(printf %q "${WORKERS}") BLAS_THREADS=$(printf %q "${BLAS_THREADS}") LOG=$(printf %q "${LOG}") $(printf %q "${PROJECT_ROOT}/state_projector_pump_dense_tmux_entrypoint.sh")"

echo "[launch] started ${SESSION}"
