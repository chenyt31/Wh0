#!/usr/bin/env bash
# Download or link model weights used by Wh0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

exec uv run --no-project --with huggingface_hub python tools/weights/manage_weights.py sync --hf-downloader "${WH0_HF_DOWNLOADER:-hfd}" "$@"
