#!/usr/bin/env bash
# Train / fine-tune the Wh0 policy (VITRA-based).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vitra-wh0"
export UV_PROJECT="$ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"

TASK="${TASK:-finetune}"

case "$TASK" in
  finetune)
    exec env CONFIG="${CONFIG:-vitra/configs/robot_finetune_wmh.json}" bash scripts/run_robot_wmh_finetune.sh "$@"
    ;;
  pretrain)
    exec env CONFIG="${CONFIG:-vitra/configs/human_pretrain.json}" bash scripts/run_human_pretrain.sh "$@"
    ;;
  eval)
    exec bash ../scripts/run_eval_pipeline.sh
    ;;
  *)
    echo "Unknown TASK=$TASK (use: finetune | pretrain | eval)"
    exit 1
    ;;
esac
