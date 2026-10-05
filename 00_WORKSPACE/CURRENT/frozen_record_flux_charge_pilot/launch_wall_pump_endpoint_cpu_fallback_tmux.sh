#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="wall_pump_endpoint_cpu_fallback_s100_v1"
OUTPUT="${SCRIPT_DIR}/imported_endpoints/wall_pump_width_endpoints_s100_v1_cpu_fallback"
LOG="${OUTPUT}/tmux.log"
ENTRYPOINT="${SCRIPT_DIR}/wall_pump_endpoint_cpu_fallback_tmux_entrypoint.sh"
A100_RECEIPT="${1:-${SCRIPT_DIR}/results/wall_pump_width_endpoints_s100_v1/benchmarks/a100_endpoint_benchmark.json}"

if [[ ! -f "${A100_RECEIPT}" && "${CONFIRM_A100_FALLBACK:-0}" != "1" ]]; then
  echo "No rejected A100 benchmark receipt was supplied." >&2
  echo "Set CONFIRM_A100_FALLBACK=1 only after explicitly deciding to use the CPU fallback." >&2
  exit 2
fi
if [[ -f "${A100_RECEIPT}" ]]; then
  python - "${A100_RECEIPT}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "rejected":
    raise SystemExit(
        "A100 benchmark is not rejected; refusing to start the 56-core fallback"
    )
PY
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"
tmux new-session -d -s "${SESSION}" "bash '${ENTRYPOINT}' 2>&1 | tee -a '${LOG}'"
echo "launched ${SESSION}; log=${LOG}"
