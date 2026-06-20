
export HF_TOKEN=""
export WANDB_API_KEY=""

: "${CONFIG:=vitra/configs/human_pretrain.json}"
: "${NPROC_PER_NODE:=8}"

uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --standalone \
    scripts/train.py \
    --config "$CONFIG" \
