#!/usr/bin/env bash
# Single 80GB GPU: WM-H generation with sequential batches and lower VL memory limits.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p database/wm-h
ln -sfn instr_first database/wm-h/wmh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/pipeline.single_gpu.yaml}"
VIDEO_CONFIG="${VIDEO_CONFIG:-configs/video.single_gpu.yaml}"
TOTAL_INSTRUCTIONS="${TOTAL_INSTRUCTIONS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VERBOSE="${VERBOSE:-0}"
IMAGE_DIR="${IMAGE_DIR:-}"
read -r -a PYTHON_CMD <<< "${WMH_PYTHON:-uv run python}"

echo "[single-gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[single-gpu] config=$CONFIG video_config=$VIDEO_CONFIG"
echo "[single-gpu] batch_size=$BATCH_SIZE (use 1 on tight 80GB; increase only if stable)"
echo "[single-gpu] python=${WMH_PYTHON:-uv run python}"

cmd=("${PYTHON_CMD[@]}" wm_h/run_streaming_video.py
  --config "$CONFIG"
  --video-config "$VIDEO_CONFIG"
  --total-instructions "$TOTAL_INSTRUCTIONS"
  --batch-size "$BATCH_SIZE"
  --num-gpus 1)
if [[ -n "$IMAGE_DIR" ]]; then
  cmd+=(--image-dir "$IMAGE_DIR")
fi
if [[ "$VERBOSE" == "1" ]]; then
  cmd+=(--verbose)
fi

exec "${cmd[@]}"
