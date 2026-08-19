#!/usr/bin/env bash
set -euo pipefail

source /share/yangtao/init.sh
cd /share/yangtao/EgoS2
source .venv/bin/activate
cd third_party/Wh0/vitra-wh0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

test -f /share/yangtao/Wh0/weights/checkpoints/vitra-vla-3b.pt
test -f /share/yangtao/Wh0/weights/statistics/dataset_statistics.json
export WANDB_MODE=disabled
torchrun --nproc_per_node=1 --standalone scripts/train.py \
  --config vitra/configs/adamu_single_episode_overfit.json "$@"
