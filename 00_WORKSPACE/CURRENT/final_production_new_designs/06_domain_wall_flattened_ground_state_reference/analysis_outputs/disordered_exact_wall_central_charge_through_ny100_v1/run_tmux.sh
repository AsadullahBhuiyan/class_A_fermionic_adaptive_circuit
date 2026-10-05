#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec > >(tee -a run.log) 2>&1
trap 'result=$?; echo "$result" > exit_code.txt; if [ "$result" -eq 0 ]; then echo "PIPELINE COMPLETE"; else echo "PIPELINE FAILED: exit $result"; fi' EXIT
rm -f exit_code.txt
for ny in 32 36 40 44 48 80; do
 if [ ! -f "Ny$(printf '%03d' "$ny")/acquisition_complete.json" ]; then
  python -u run_analysis.py --sizes "$ny" --workers 16
 fi
 python -u build_notebook.py
 if [ "$ny" -eq 32 ]; then
  python -u '/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/entanglement_level_statistics_exact_wall_strength_scan_n20x32_L099_v2/build_notebook.py'
 fi
done
