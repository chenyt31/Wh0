#!/usr/bin/env bash
# Visualize Wh0 training/test data for quick sanity checks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-${1:-}}"
MODE="${MODE:-auto}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
MANO_PATH="${MANO_PATH:-$ROOT/weights/mano}"
MAX_EPISODES="${MAX_EPISODES:-4}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-16}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-15}"
RENDER_HAND="${RENDER_HAND:-0}"
COMPARE_ACTION="${COMPARE_ACTION:-0}"
WHOLE_EPISODE="${WHOLE_EPISODE:-0}"

usage() {
  cat <<'EOF'
Usage:
  DATA_ROOT=<path> bash scripts/run_visualize_training_data.sh
  bash scripts/run_visualize_training_data.sh <path>

Modes:
  auto  Detect WM-H/VITRA tree or G1 robot training_index.npz
  wmh   Visualize WM-H/VITRA episodic annotations over videos
  g1    Visualize G1 robot training samples and action rollout

Environment:
  MODE=auto|wmh|g1
  OUTPUT_PATH=<dir>
  MANO_PATH=<dir>            default: weights/mano
  MAX_EPISODES=4             WM-H episodes to render, 0 = all
  SAMPLE_IDX=0               G1 first sample index
  NUM_SAMPLES=4              G1 samples to render
  SAMPLE_STRIDE=16           G1 stride between samples
  NUM_FRAMES=81              G1 sequence length
  RENDER_HAND=1|0            Enable MANO/PyTorch3D mesh render with 1
  COMPARE_ACTION=1|0         G1 action rollout comparison (default 0)
  WHOLE_EPISODE=1|0          G1 render the whole selected episode
EOF
}

if [[ -z "$DATA_ROOT" || "$DATA_ROOT" == "-h" || "$DATA_ROOT" == "--help" ]]; then
  usage
  exit 0
fi

DATA_ROOT="$(realpath "$DATA_ROOT")"
if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$DATA_ROOT/visualization"
fi
OUTPUT_PATH="$(mkdir -p "$(dirname "$OUTPUT_PATH")" && realpath -m "$OUTPUT_PATH")"

detect_mode() {
  local root="$1"
  if [[ -d "$root/Annotation/WM-H/episodic_annotations" ]]; then
    echo "wmh"
  elif [[ -d "$root/vitra_training_data/Annotation/WM-H/episodic_annotations" ]]; then
    echo "wmh"
  elif [[ -f "$root/training_index.npz" ]]; then
    echo "g1"
  elif [[ -d "$root/episodic_annotations" && -d "$root/videos" ]]; then
    echo "wmh"
  elif [[ -d "$root/annotations" && -d "$root/videos" ]]; then
    echo "wmh"
  else
    echo "unknown"
  fi
}

if [[ "$MODE" == "auto" ]]; then
  MODE="$(detect_mode "$DATA_ROOT")"
fi

common_env=(env UV_PROJECT="$ROOT")

case "$MODE" in
  wmh)
    if [[ -d "$DATA_ROOT/vitra_training_data/Annotation/WM-H/episodic_annotations" ]]; then
      DATA_ROOT="$DATA_ROOT/vitra_training_data"
    fi
    if [[ -d "$DATA_ROOT/Annotation/WM-H/episodic_annotations" ]]; then
      VIDEO_ROOT="$DATA_ROOT/Video/WM-H_root"
      LABEL_ROOT="$DATA_ROOT/Annotation/WM-H/episodic_annotations"
    elif [[ -d "$DATA_ROOT/episodic_annotations" && -d "$DATA_ROOT/videos" ]]; then
      VIDEO_ROOT="$DATA_ROOT/videos"
      LABEL_ROOT="$DATA_ROOT/episodic_annotations"
    elif [[ -d "$DATA_ROOT/annotations" && -d "$DATA_ROOT/videos" ]]; then
      VIDEO_ROOT="$DATA_ROOT/videos"
      LABEL_ROOT="$DATA_ROOT/annotations"
    else
      echo "Cannot find WM-H video/annotation folders under: $DATA_ROOT" >&2
      exit 1
    fi
    args=(
      --video_root "$VIDEO_ROOT"
      --label_root "$LABEL_ROOT"
      --save_path "$OUTPUT_PATH"
      --mano_model_path "$MANO_PATH"
      --max_episodes "$MAX_EPISODES"
    )
    if [[ "$RENDER_HAND" == "0" ]]; then
      args+=(--no_render)
    fi
    "${common_env[@]}" uv run python vitra-wh0/data/demo_visualization_epi.py "${args[@]}"
    ;;
  g1)
    args=(
      --root-dir "$DATA_ROOT"
      --sample-idx "$SAMPLE_IDX"
      --num-frames "$NUM_FRAMES"
      --num-samples "$NUM_SAMPLES"
      --sample-stride "$SAMPLE_STRIDE"
      --output-dir "$OUTPUT_PATH"
      --mano-path "$MANO_PATH"
      --fps "$FPS"
    )
    if [[ "$RENDER_HAND" == "0" ]]; then
      args+=(--no-render)
    fi
    if [[ "$COMPARE_ACTION" == "0" ]]; then
      args+=(--no-action-compare)
    fi
    if [[ "$WHOLE_EPISODE" == "1" ]]; then
      args+=(--whole-episode)
    fi
    (cd vitra-wh0 && "${common_env[@]}" uv run python -m vitra.tools.render_hand_dataset "${args[@]}")
    ;;
  *)
    echo "Could not auto-detect data type for: $DATA_ROOT" >&2
    echo "Set MODE=wmh or MODE=g1 explicitly." >&2
    exit 1
    ;;
esac

echo "Visualization outputs: $OUTPUT_PATH"
