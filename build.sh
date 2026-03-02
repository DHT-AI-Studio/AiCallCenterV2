#!/bin/bash
set -e

git pull

IMAGE_NAME="sip-server-v2"
IMAGE_TAG="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"

echo "> Stopping old container..."
docker stop ${IMAGE_NAME} 2>/dev/null || true
docker rm ${IMAGE_NAME} 2>/dev/null || true
echo "> Container deleted"

echo "> Starting new container..."
docker compose up -d

echo "> Deployment complete!"