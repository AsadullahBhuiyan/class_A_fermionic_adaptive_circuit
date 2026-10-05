#!/bin/bash
# Run this on your LOCAL MACHINE (not the OSG access point).
# Docker must be installed and you must be logged in: docker login
set -euo pipefail

DOCKERHUB_USER="${1:-<your-dockerhub-username>}"
IMAGE="${DOCKERHUB_USER}/fermionic-env:latest"

echo "Building ${IMAGE} ..."
docker build -t "${IMAGE}" .

echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

echo ""
echo "Done. Update container_image in batch_gpu.submit to:"
echo "  container_image = docker://${IMAGE}"
