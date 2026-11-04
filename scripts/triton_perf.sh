#!/usr/bin/env bash
# Load tests the diffusion U-Net on Triton with NVIDIA perf_analyzer.
#
# 1. Start Triton:    docker compose -f docker-compose-triton.yml up -d
# 2. Run this script: bash scripts/triton_perf.sh
#
# It sweeps concurrency from 1 to 16 so dynamic batching can group requests,
# then writes throughput and p50/p95/p99 latency to outputs/triton_perf.csv.
# Paste the numbers into the README results table.
set -euo pipefail

TRITON_VERSION="${TRITON_VERSION:-24.08}"
URL="${TRITON_URL:-localhost:8002}"
mkdir -p outputs

docker run --rm --net=host -v "$PWD/outputs:/outputs" \
  "nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3-sdk" \
  perf_analyzer \
    -m diffusion_unet \
    -u "$URL" -i grpc \
    --shape t:1 \
    --concurrency-range 1:16:3 \
    --percentile=95 \
    --measurement-interval 5000 \
    -f /outputs/triton_perf.csv

echo "Results saved to outputs/triton_perf.csv"
column -s, -t < outputs/triton_perf.csv | cut -c1-140
