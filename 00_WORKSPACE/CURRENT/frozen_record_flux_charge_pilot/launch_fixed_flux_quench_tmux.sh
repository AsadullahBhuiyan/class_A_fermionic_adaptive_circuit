#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-frozen_record_fixed_flux_quench_N16x20_s10_v1}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/campaign_config.fixed_flux_quench_s10_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/N16x20_frozen_record_fixed_flux_quench_s10_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
LOG="${LOG:-${OUTPUT_ROOT}/logs/tmux_fixed_flux_quench.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

printf '[launch] session=%s\n' "${SESSION}"
printf '[launch] cores=%s workers=%s BLAS_threads=%s\n' "${CPU_LIST}" "${WORKERS}" "${BLAS_THREADS}"
printf '[launch] config=%s\n' "${CONFIG}"
printf '[launch] output=%s\n' "${OUTPUT_ROOT}"
printf '[launch] log=%s\n' "${LOG}"
printf '[launch] attach: tmux attach -t %q\n' "${SESSION}"
printf '[launch] follow: tail -f %q\n' "${LOG}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "CONFIG=$(printf %q "${CONFIG}") OUTPUT_ROOT=$(printf %q "${OUTPUT_ROOT}") CPU_LIST=$(printf %q "${CPU_LIST}") WORKERS=$(printf %q "${WORKERS}") BLAS_THREADS=$(printf %q "${BLAS_THREADS}") LOG=$(printf %q "${LOG}") $(printf %q "${PROJECT_ROOT}/fixed_flux_quench_tmux_entrypoint.sh")"

echo "[launch] started ${SESSION}"
