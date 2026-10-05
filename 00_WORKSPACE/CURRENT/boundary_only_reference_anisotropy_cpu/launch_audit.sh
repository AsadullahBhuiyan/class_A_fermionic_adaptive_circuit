#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REVISION="boundary_reference_anisotropy_uniform_qwz_nx16_ny16-32_v1"
OUTPUT="$ROOT/outputs/$REVISION"
SESSION="boundary-alpha-audit"
LOG="$OUTPUT/logs/audit.log"
mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && taskset -c 28-47 python -u run_campaign.py audit --workers 20 --cpu-list 28-47 2>&1 | tee '$LOG'"
echo "started $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: $LOG"
