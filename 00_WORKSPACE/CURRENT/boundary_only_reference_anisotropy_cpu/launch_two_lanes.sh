#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REVISION="boundary_reference_anisotropy_uniform_qwz_nx16_ny16-32_v1"
OUTPUT="$ROOT/outputs/$REVISION"
AUDIT="$OUTPUT/analysis/audit_decision.json"
if [[ ! -f "$AUDIT" ]] || [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$AUDIT")" != "passed" ]]; then
  echo "production requires a passed audit: $AUDIT"
  exit 1
fi
mkdir -p "$OUTPUT/logs"
for session in boundary-alpha-lane-a boundary-alpha-lane-b; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session"
    exit 1
  fi
done
tmux new-session -d -s boundary-alpha-lane-a \
  "cd '$ROOT' && taskset -c 0-19 python -u run_campaign.py production --sizes 32,24,16 --workers 20 --cpu-list 0-19 2>&1 | tee '$OUTPUT/logs/lane_a.log'"
tmux new-session -d -s boundary-alpha-lane-b \
  "cd '$ROOT' && taskset -c 28-47 python -u run_campaign.py production --sizes 28,20 --workers 20 --cpu-list 28-47 2>&1 | tee '$OUTPUT/logs/lane_b.log'"
echo "started boundary-alpha-lane-a and boundary-alpha-lane-b"
echo "attach: tmux attach -t boundary-alpha-lane-a"
echo "attach: tmux attach -t boundary-alpha-lane-b"
