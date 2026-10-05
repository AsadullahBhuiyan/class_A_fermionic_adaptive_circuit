#!/bin/bash
# Run this ON THE OSG ACCESS POINT after transferring the OSG/ directory.
# Usage: bash submit_jobs.sh
set -euo pipefail

OSG_USER="asad.bhuiyan"
ACCESS_POINT="ap20.uw.osg-htc.org"

echo "=== Step 1: Generate params.json ==="
python3 generate_params.py

echo ""
echo "=== Step 2: Create logs directory ==="
mkdir -p logs

echo ""
echo "=== Step 3: Make run.sh executable ==="
chmod +x run.sh

echo ""
echo "=== Step 4: Submit ==="
condor_submit batch_gpu.submit

echo ""
echo "=== Monitoring ==="
echo "  condor_watch_q          (live)"
echo "  condor_q -hold          (held jobs)"
echo "  condor_rm \$USER          (cancel all)"
