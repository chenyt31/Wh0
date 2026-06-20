#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vitra-wh0"

IMAGE_PATH="${IMAGE_PATH:-}"
INSTRUCTION="${INSTRUCTION:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-../output_videos/human_inference.mp4}"
SAMPLE_TIMES="${SAMPLE_TIMES:-4}"
FPS="${FPS:-8}"
MANO_PATH="${MANO_PATH:-../weights/mano}"
MODEL_PATH="${MODEL_PATH:-}"
HAND="${HAND:-right}"

if [[ -z "$IMAGE_PATH" || -z "$INSTRUCTION" || -z "$CONFIG_PATH" ]]; then
  cat >&2 <<'EOF'
IMAGE_PATH, INSTRUCTION, and CONFIG_PATH are required.

Example:
  IMAGE_PATH=/path/to/image.jpg \
  INSTRUCTION='Left hand: None. Right hand: pick up the cup.' \
  CONFIG_PATH=vla_checkpoint/<run>/config.json \
  bash scripts/run_human_inference.sh
EOF
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_VIDEO")"

args=(
  env UV_PROJECT="$ROOT" uv run python scripts/inference_human_prediction.py
  --config_path "$CONFIG_PATH"
  --image_path "$IMAGE_PATH"
  --instruction "$INSTRUCTION"
  --video_path "$OUTPUT_VIDEO"
  --sample_times "$SAMPLE_TIMES"
  --fps "$FPS"
  --mano_path "$MANO_PATH"
)

if [[ -n "$MODEL_PATH" ]]; then
  args+=(--model_path "$MODEL_PATH")
fi

case "$HAND" in
  left) args+=(--use_left) ;;
  right) args+=(--use_right) ;;
  both) args+=(--use_left --use_right) ;;
  *) echo "HAND must be left, right, or both" >&2; exit 1 ;;
esac

exec "${args[@]}"
