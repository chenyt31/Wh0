#!/usr/bin/env bash
# Annotate WM-H videos with HaWoR hand poses → VITRA-format .npy files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_PATH="${INPUT_PATH:-}"
PARALLEL_K="${PARALLEL_K:-1}"
DATASET_NAME="${WH0_DATASET_NAME:-WM-H}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: INPUT_PATH=WM-H/database/wm-h/.../run_xxx bash scripts/run_annotate.sh"
  echo ""
  echo "Environment variables:"
  echo "  INPUT_PATH     Root folder containing generated videos (required)"
  echo "  PARALLEL_K     Models per GPU (0 = serial, default 1)"
  echo "  WH0_DATASET_NAME  Output npy prefix (default: WM-H)"
  exit 1
fi

INPUT_PATH="$(realpath "$INPUT_PATH")"

cd vitra-wh0

exec env UV_PROJECT="$ROOT" uv run python ../tools/wmh/annotate_wmh_videos.py \
  --input_path "$INPUT_PATH" \
  --parallel_k "$PARALLEL_K" \
  --dataset-name "$DATASET_NAME" \
  "$@"
