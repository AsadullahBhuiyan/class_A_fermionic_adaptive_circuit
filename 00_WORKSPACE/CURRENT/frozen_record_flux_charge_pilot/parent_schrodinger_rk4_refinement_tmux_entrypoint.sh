#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
primary_root="${project_root}/results/N20x24_parent_schrodinger_rk4_s50_tau1e4_v1"
refinement_root="${project_root}/results/N20x24_parent_schrodinger_rk4_s2_tau1e4_dt_half_v1"
primary_session="parent_schrodinger_rk4_N20x24_s50_tau1e4_v1"
analysis_marker="${primary_root}/analysis/analysis_summary.json"

echo "[wait] primary analysis marker: ${analysis_marker}"
while [[ ! -f "${analysis_marker}" ]]; do
  if ! tmux has-session -t "${primary_session}" 2>/dev/null; then
    echo "[failure] primary session ended before verified analysis was published" >&2
    exit 1
  fi
  sleep 60
done

echo "[launch] four-path RK4 step-halving refinement"
exec taskset -c 28-31 \
  "${HOME}/anaconda3/bin/python" -u \
  "${project_root}/run_parent_schrodinger_rk4_refinement.py" run \
  --workers 4 \
  --output-root "${refinement_root}"
