#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
LOG_ROOT="$HERE/outputs/logs"
mkdir -p "$LOG_ROOT"

HARD_SESSION="postselect_n20x40_hard"
SOFT_SESSION="postselect_n20x40_soft"

if tmux has-session -t "$HARD_SESSION" 2>/dev/null || tmux has-session -t "$SOFT_SESSION" 2>/dev/null; then
  echo "One of the requested tmux sessions already exists; refusing a duplicate launch." >&2
  exit 1
fi

tmux new-session -d -s "$HARD_SESSION" \
  "cd '$REPO_ROOT' && export OMP_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28 MKL_NUM_THREADS=28 NUMEXPR_NUM_THREADS=28; numactl --cpunodebind=0 --membind=0 taskset -c 0-27 nice -n 5 python -u '$HERE/run_hard_wall.py' 2>&1 | tee '$LOG_ROOT/hard.log'"

tmux new-session -d -s "$SOFT_SESSION" \
  "cd '$REPO_ROOT' && export OMP_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28 MKL_NUM_THREADS=28 NUMEXPR_NUM_THREADS=28; numactl --cpunodebind=1 --membind=1 taskset -c 28-55 nice -n 5 python -u '$HERE/run_soft_wall.py' 2>&1 | tee '$LOG_ROOT/soft.log'"

echo "launched $HARD_SESSION and $SOFT_SESSION"

