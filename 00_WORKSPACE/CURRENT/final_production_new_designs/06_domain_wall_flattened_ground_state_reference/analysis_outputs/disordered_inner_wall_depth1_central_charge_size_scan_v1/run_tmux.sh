#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec > >(tee -a run.log) 2>&1
trap 'result=$?; echo "$result" > exit_code.txt; if [ "$result" -eq 0 ]; then echo "PIPELINE COMPLETE"; else echo "PIPELINE FAILED: exit $result"; fi' EXIT
rm -f exit_code.txt
echo "STARTED $(date -Is)"
python -u run_analysis.py --pilot --sizes 36 48
python -u run_analysis.py
python -u build_notebook.py
