#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-kato_overlap_continuation_N20x24_N24x24_s25_v1}"
CPU_LIST="${CPU_LIST:-28-55}"
WORKERS="${WORKERS:-28}"
OUTPUT_ROOT="${PROJECT_DIR}/results/N20x24_N24x24_kato_overlap_continuation_s25_v1"
LOG="${OUTPUT_ROOT}/tmux.log"
SUPERSEDED="parent_schrodinger_rk4_N24x24_s50_tau1e4_v1"

if tmux has-session -t "${SUPERSEDED}" 2>/dev/null; then
  echo "Refusing to launch while the superseded N24x24 physical-time session is active: ${SUPERSEDED}" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
printf '[launch] session=%s cores=%s workers=%s log=%s\n' "${SESSION}" "${CPU_LIST}" "${WORKERS}" "${LOG}"
tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_DIR}' && CPU_LIST='${CPU_LIST}' WORKERS='${WORKERS}' bash './kato_overlap_continuation_tmux_entrypoint.sh' 2>&1 | tee -a '${LOG}'"
tmux list-sessions | grep "^${SESSION}:"
