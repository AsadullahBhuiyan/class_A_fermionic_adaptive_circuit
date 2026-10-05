#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
SESSION="${SESSION:-frozen_flux_pure_tangent_N16x20_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
BLAS_THREADS="${BLAS_THREADS:-1}"
CONFIG="${CONFIG:-${HERE}/campaign_config.pure_tangent_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/results/N16x20_frozen_flux_pure_tangent_v1}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/tmux.log"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'session=%s\ncpus=%s\nworkers=%s\nconfig=%s\noutput=%s\nlog=%s\n' \
    "${SESSION}" "${CPU_LIST}" "${WORKERS}" "${CONFIG}" "${OUTPUT_ROOT}" "${LOG_PATH}"
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "attach: tmux attach -t ${SESSION}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
COMMAND="cd $(printf '%q' "${REPO_ROOT}") && CPU_LIST=$(printf '%q' "${CPU_LIST}") WORKERS=$(printf '%q' "${WORKERS}") BLAS_THREADS=$(printf '%q' "${BLAS_THREADS}") CONFIG=$(printf '%q' "${CONFIG}") OUTPUT_ROOT=$(printf '%q' "${OUTPUT_ROOT}") LOG_PATH=$(printf '%q' "${LOG_PATH}") exec bash $(printf '%q' "${HERE}/pure_tangent_tmux_entrypoint.sh")"
tmux new-session -d -s "${SESSION}" "${COMMAND}"

echo "launched: ${SESSION}"
echo "cpus: ${CPU_LIST}; workers: ${WORKERS}; BLAS threads: ${BLAS_THREADS}"
echo "log: ${LOG_PATH}"
echo "attach: tmux attach -t ${SESSION}"
echo "follow: tail -f ${LOG_PATH}"

