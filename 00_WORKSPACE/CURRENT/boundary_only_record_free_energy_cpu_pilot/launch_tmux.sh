#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="boundary_record_nx16_v1"
LOG="$ROOT/outputs/boundary_only_flattened_ground_nx16_ny16-32_s100_4ny_v1/campaign.log"
WORKERS="${1:-8}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already running: $SESSION"
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec python -u run_campaign.py --workers '$WORKERS' >> '$LOG' 2>&1"
echo "started $SESSION"
echo "log: $LOG"
