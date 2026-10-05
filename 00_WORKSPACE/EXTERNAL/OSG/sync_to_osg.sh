#!/bin/bash
# Run this on YOUR LOCAL MACHINE to push the OSG/ folder to the access point.
# Usage: bash sync_to_osg.sh
set -euo pipefail

OSG_USER="asad.bhuiyan"
ACCESS_POINT="ap20.uw.osg-htc.org"
REMOTE_DIR="~/osg_jobs"
LOCAL_SRC_DIR="../src/fgtn"   # classA_U1FGTN_gpu.py lives here

echo "Syncing OSG job files to ${OSG_USER}@${ACCESS_POINT}:${REMOTE_DIR} ..."

ssh "${OSG_USER}@${ACCESS_POINT}" "mkdir -p ${REMOTE_DIR}/src ${REMOTE_DIR}/logs"

# Sync job scripts
rsync -avz --progress \
    batch_gpu.submit \
    run.sh \
    generate_params.py \
    "${OSG_USER}@${ACCESS_POINT}:${REMOTE_DIR}/"

# Sync the GPU source module
rsync -avz --progress \
    "${LOCAL_SRC_DIR}/classA_U1FGTN_gpu.py" \
    "${LOCAL_SRC_DIR}/__init__.py" \
    "${OSG_USER}@${ACCESS_POINT}:${REMOTE_DIR}/src/"

echo ""
echo "Done. Now SSH in and run:"
echo "  ssh ${OSG_USER}@${ACCESS_POINT}"
echo "  cd ${REMOTE_DIR}"
echo "  bash submit_jobs.sh"
