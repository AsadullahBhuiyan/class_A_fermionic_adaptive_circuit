#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-parent_schrodinger_rk4_N20x24_s50_tau1e4_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
OUTPUT="${PROJECT_ROOT}/results/N20x24_parent_schrodinger_rk4_s50_tau1e4_v1"
LOG="${LOG:-${OUTPUT}/logs/tmux_parent_schrodinger_rk4_s50.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

printf '[launch] session=%s\n' "${SESSION}"
printf '[launch] 50 verified Ny=24 endpoints; T=10000 RK4; CW+CCW\n'
printf '[launch] cores=%s workers=%s BLAS threads=%s\n' "${CPU_LIST}" "${WORKERS}" "${BLAS_THREADS}"
printf '[launch] output=%s\n' "${OUTPUT}"
printf '[launch] log=%s\n' "${LOG}"
if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "CPU_LIST=$(printf %q "${CPU_LIST}") WORKERS=$(printf %q "${WORKERS}") BLAS_THREADS=$(printf %q "${BLAS_THREADS}") LOG=$(printf %q "${LOG}") $(printf %q "${PROJECT_ROOT}/parent_schrodinger_rk4_tmux_entrypoint.sh")"
echo "[launch] started ${SESSION}"
