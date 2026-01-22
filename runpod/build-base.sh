#!/bin/bash
# Build and push the base image with PyTorch 2.6.0
# Only needs to be run once (or when PyTorch version changes)

set -e

echo "Building base image with PyTorch 2.6.0..."
docker build --platform linux/amd64 -f Dockerfile.base -t neilellis/pytorch-base:2.6.0-cu124 .

echo "Pushing base image to Docker Hub..."
docker push neilellis/pytorch-base:2.6.0-cu124

echo "Base image ready: neilellis/pytorch-base:2.6.0-cu124"