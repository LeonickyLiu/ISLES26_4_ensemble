#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26_c09_hybrid_final_20260819"

docker build \
  --platform=linux/amd64 \
  --provenance=false \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG"  \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
