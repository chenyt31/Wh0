
export HF_TOKEN="${HF_TOKEN:-}"

: "${CONFIG:=vitra/configs/robot_finetune_wmh.json}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    else
        NPROC_PER_NODE=1
    fi
fi

uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --standalone \
    scripts/train.py \
    --config "$CONFIG" \
    "$@"
