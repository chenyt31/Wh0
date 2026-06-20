#!/usr/bin/env bash
# Unified Wh0 pipeline entrypoint (human- and agent-friendly).
#
# Examples:
#   bash scripts/run_all.sh --stage setup
#   bash scripts/run_all.sh --stage wmh --profile single_gpu
#   bash scripts/run_all.sh --stage annotate --input-path WM-H/database/wm-h/instr_first/streaming_runs/run_xxx
#   bash scripts/run_all.sh --stage hand_edit --input-path WM-H/database/wm-h/instr_first/streaming_runs/run_xxx/videos
#   bash scripts/run_all.sh --stage prepare_data --input-path WM-H/database/wm-h/instr_first/streaming_runs/run_xxx
#   bash scripts/run_all.sh --stage visualize --input-path WM-H/database/wm-h/instr_first/streaming_runs/run_xxx/vitra_training_data
#   bash scripts/run_all.sh --stage train --task finetune
#   bash scripts/run_all.sh --stage all --input-path ... --profile default
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="help"
PROFILE="default"
TASK="finetune"
INPUT_PATH=""
OUTPUT_PATH=""
PARALLEL_K="1"
HAND_EDIT_EVERY_N="0"
ROBOT_PROB="0.2"
TOTAL_INSTRUCTIONS="${TOTAL_INSTRUCTIONS:-10}"

usage() {
  cat <<'EOF'
Wh0 pipeline

Stages:
  setup      Install deps + sync third-party repos (scripts/setup.sh)
  weights    Download or link configured weights
  wmh        Generate synthetic manipulation videos (WM-H)
  annotate   HaWoR hand annotation → VITRA-format .npy
  hand_edit  Replace generated human hands with robot hands in videos
  prepare_data
             Build VITRA WM-H symlink tree with 20% robot-hand videos
  visualize  Render training/test data visualization videos
  train      Policy training / eval (vitra-wh0)
  all        wmh → annotate (train is separate; needs dataset prep)

Options:
  --stage STAGE
  --profile default|single_gpu
                                   WM-H GPU profile
  --input-path PATH                Required for annotate / all
  --output-path PATH               Optional output root for prepare_data
                                   or visualization output directory
  --parallel-k N                   Annotation parallelism (default 1)
  --hand-edit-every-n N            Edit every N frames for hand_edit (default from WM-H config: 4)
  --robot-prob FLOAT               Robot-hand video probability (default 0.2)
  --task finetune|pretrain|eval    Training task
  --total-instructions N           WM-H batch size hint
  -h, --help

Environment:
  WH0_ROOT, WH0_DATASET_NAME, WH0_DROID_WEIGHTS, WH0_MOGE_MODEL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --input-path) INPUT_PATH="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --parallel-k) PARALLEL_K="$2"; shift 2 ;;
    --hand-edit-every-n) HAND_EDIT_EVERY_N="$2"; shift 2 ;;
    --robot-prob) ROBOT_PROB="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --total-instructions) TOTAL_INSTRUCTIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

case "$STAGE" in
  help)
    usage
    ;;
  setup)
    bash scripts/setup.sh
    ;;
  weights)
    bash scripts/download_weights.sh
    ;;
  wmh)
    PROFILE="$PROFILE" TOTAL_INSTRUCTIONS="$TOTAL_INSTRUCTIONS" bash scripts/run_wmh.sh
    ;;
  annotate)
    INPUT_PATH="$INPUT_PATH" PARALLEL_K="$PARALLEL_K" bash scripts/run_annotate.sh
    ;;
  hand_edit)
    if [[ -z "$INPUT_PATH" ]]; then
      echo "Usage: bash scripts/run_all.sh --stage hand_edit --input-path <run_dir>/videos"
      exit 1
    fi
    args=(--input-dir "$(realpath "$INPUT_PATH")")
    if [[ "$HAND_EDIT_EVERY_N" != "0" ]]; then
      args+=(--every-n "$HAND_EDIT_EVERY_N")
    fi
    (cd WM-H && bash scripts/run_video_hand_edit.sh "${args[@]}")
    ;;
  prepare_data)
    if [[ -z "$INPUT_PATH" ]]; then
      echo "Usage: bash scripts/run_all.sh --stage prepare_data --input-path <run_dir>"
      exit 1
    fi
    args=("$(realpath "$INPUT_PATH")" --robot-prob "$ROBOT_PROB")
    if [[ -n "$OUTPUT_PATH" ]]; then
      args+=(--output-root "$OUTPUT_PATH")
    fi
    uv run python tools/wmh/prepare_wmh_training_data.py "${args[@]}"
    ;;
  visualize)
    if [[ -z "$INPUT_PATH" ]]; then
      echo "Usage: bash scripts/run_all.sh --stage visualize --input-path <training_data_or_run_dir>"
      exit 1
    fi
    env_args=(DATA_ROOT="$(realpath "$INPUT_PATH")")
    if [[ -n "$OUTPUT_PATH" ]]; then
      env_args+=(OUTPUT_PATH="$OUTPUT_PATH")
    fi
    env "${env_args[@]}" bash scripts/run_visualize_training_data.sh
    ;;
  train)
    TASK="$TASK" bash scripts/run_train.sh
    ;;
  all)
    PROFILE="$PROFILE" TOTAL_INSTRUCTIONS="$TOTAL_INSTRUCTIONS" bash scripts/run_wmh.sh
    if [[ -z "$INPUT_PATH" ]]; then
      echo "After WM-H finishes, re-run with:"
      echo "  bash scripts/run_all.sh --stage annotate --input-path WM-H/database/wm-h/instr_first/streaming_runs/<run_id>"
      exit 0
    fi
    INPUT_PATH="$INPUT_PATH" PARALLEL_K="$PARALLEL_K" bash scripts/run_annotate.sh
    ;;
  *)
    echo "Unknown stage: $STAGE"
    usage
    exit 1
    ;;
esac
