#!/usr/bin/env bash
# Run WM-H synthetic data generation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/WM-H"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export UV_PROJECT="$ROOT"

PROFILE="${PROFILE:-default}"
TOTAL_INSTRUCTIONS="${TOTAL_INSTRUCTIONS:-8}"

case "$PROFILE" in
  default)
    CONFIG="${CONFIG:-configs/pipeline.yaml}"
    VIDEO_CONFIG="${VIDEO_CONFIG:-configs/video.yaml}"
    exec env CONFIG="$CONFIG" VIDEO_CONFIG="$VIDEO_CONFIG" bash scripts/run_wmh.sh
    ;;
  single_gpu)
    CONFIG="${CONFIG:-configs/pipeline.single_gpu.yaml}"
    VIDEO_CONFIG="${VIDEO_CONFIG:-configs/video.single_gpu.yaml}"
    exec env CONFIG="$CONFIG" VIDEO_CONFIG="$VIDEO_CONFIG" bash scripts/run_wmh_single_gpu.sh
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (use: default | single_gpu)"
    exit 1
    ;;
esac
