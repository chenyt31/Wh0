#!/usr/bin/env bash
# Single-GPU or multi-GPU WM-H streaming in one process per GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p database/wm-h
ln -sfn instr_first database/wm-h/wmh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/pipeline.yaml}"
VIDEO_CONFIG="${VIDEO_CONFIG:-configs/video.yaml}"
TOTAL_INSTRUCTIONS="${TOTAL_INSTRUCTIONS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-0}"
VERBOSE="${VERBOSE:-0}"
IMAGE_DIR="${IMAGE_DIR:-}"
read -r -a PYTHON_CMD <<< "${WMH_PYTHON:-uv run python}"

cmd=("${PYTHON_CMD[@]}" wm_h/run_streaming_video.py
  --config "$CONFIG"
  --video-config "$VIDEO_CONFIG"
  --total-instructions "$TOTAL_INSTRUCTIONS"
  --batch-size "$BATCH_SIZE"
  --num-gpus "$NUM_GPUS")
if [[ -n "$IMAGE_DIR" ]]; then
  cmd+=(--image-dir "$IMAGE_DIR")
fi
if [[ "$VERBOSE" == "1" ]]; then
  cmd+=(--verbose)
fi

exec "${cmd[@]}"
