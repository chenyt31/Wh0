#!/usr/bin/env bash
# Multi-GPU producer/consumer: slot assembly + image edit on producer GPU(s),
# resident Wan I2V on consumer GPU(s). Recommended for large runs (e.g. 50k).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p database/wm-h
ln -sfn instr_first database/wm-h/wmh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

TOTAL_INSTRUCTIONS="${TOTAL_INSTRUCTIONS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRODUCER_GPUS="${PRODUCER_GPUS:-0}"
export PRODUCER_GPUS
CONFIG="${CONFIG:-configs/pipeline.yaml}"
VIDEO_CONFIG="${VIDEO_CONFIG:-configs/video.yaml}"
RUN_LABEL="${RUN_LABEL:-wmh_8gpu_50k_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-database/wm-h/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_LABEL}.log}"
MAX_VIDEO_FAILURES="${MAX_VIDEO_FAILURES:-100}"
THROUGHPUT_LOG_EVERY="${THROUGHPUT_LOG_EVERY:-10}"

mkdir -p "$LOG_DIR"

visible_count="$(uv run python - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
items = [x.strip() for x in visible.split(",") if x.strip()]
print(len(items) if items else 0)
PY
)"

if [[ "$visible_count" -lt 2 ]]; then
  echo "ERROR: expected at least 2 visible GPUs, got ${visible_count}" >&2
  exit 1
fi

if [[ -z "${CONSUMER_GPUS:-}" ]]; then
  CONSUMER_GPUS="$(uv run python - <<'PY'
import os
visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
producers = {x.strip() for x in os.environ.get("PRODUCER_GPUS", "0").split(",") if x.strip()}
consumers = [gpu for i, gpu in enumerate(visible) if gpu not in producers and str(i) not in producers]
print(",".join(consumers))
PY
)"
fi
export CONSUMER_GPUS

consumer_count="$(uv run python - <<'PY'
import os
items = [x.strip() for x in os.environ.get("CONSUMER_GPUS", "").split(",") if x.strip()]
print(len(items))
PY
)"
if [[ "$consumer_count" -lt 1 ]]; then
  echo "ERROR: no consumer GPUs left after producer assignment" >&2
  exit 1
fi

QUEUE_SIZE="${QUEUE_SIZE:-$((consumer_count * 2))}"

echo "[$(date '+%F %T')] Starting WM-H producer/consumer run"
echo "  repo:               $ROOT_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  producer GPUs:      $PRODUCER_GPUS"
echo "  consumer GPUs:      $CONSUMER_GPUS"
echo "  total instructions: $TOTAL_INSTRUCTIONS"
echo "  batch size:         $BATCH_SIZE"
echo "  queue size:         $QUEUE_SIZE"
echo "  log file:           $LOG_FILE"

uv run python scripts/run_local_producer_consumer.py \
  --config "$CONFIG" \
  --video-config "$VIDEO_CONFIG" \
  --producer-gpus "$PRODUCER_GPUS" \
  --consumer-gpus "$CONSUMER_GPUS" \
  --total-instructions "$TOTAL_INSTRUCTIONS" \
  --batch-size "$BATCH_SIZE" \
  --queue-size "$QUEUE_SIZE" \
  --max-video-failures "$MAX_VIDEO_FAILURES" \
  --throughput-log-every "$THROUGHPUT_LOG_EVERY" \
  --run-label "$RUN_LABEL" \
  --verbose 2>&1 | tee "$LOG_FILE"
