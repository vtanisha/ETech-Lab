#!/bin/bash
# ===============================================================
#  cleanup.sh — Cleanup of project Docker environment
# ---------------------------------------------------------------
#  Removes unused images, containers, and build cache
#  to free up space after experiments.
#
# ===============================================================

set -e

# Image name and tag to delete
IMAGE_NAME="llm_predictor"
IMAGE_TAG="latest"

echo "#####################################################"
echo "### Removing Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "#####################################################"

# Remove image, ignoring error if it doesn't exist
docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null || true

echo "### Cleaning Docker builder cache..."
# Force clean build cache
docker builder prune -f

echo "### Cleanup completed."