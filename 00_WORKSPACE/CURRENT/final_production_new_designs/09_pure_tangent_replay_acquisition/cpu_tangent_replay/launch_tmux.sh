#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SESSION=pure_tangent_cpu_replay_s100_v3
OUTPUT="$HERE/../cpu_data/pure_tangent_cpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v3"
LOG="$OUTPUT/tmux.log"

mkdir -p "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" \
  "bash '$HERE/run_adaptive_cpu_production.sh' 2>&1 | tee -a '$LOG'"
printf 'launched tmux session %s\n' "$SESSION"
printf 'follow with: tmux attach -t %s\n' "$SESSION"
printf 'or: tail -f %s\n' "$LOG"
