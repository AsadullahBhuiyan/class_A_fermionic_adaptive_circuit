#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec > >(tee -a run.log) 2>&1
trap 'result=$?; echo "$result" > exit_code.txt; if [ "$result" -eq 0 ]; then echo "PIPELINE COMPLETE"; else echo "PIPELINE FAILED: exit $result"; fi' EXIT
rm -f exit_code.txt
python -u run_analysis.py --sizes 20 12 --pilot
for nx in 12 16 24 32 40; do
 python -u run_analysis.py --sizes "$nx" --workers 16
 python -u build_notebook.py
done
