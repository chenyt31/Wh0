#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-summary}"
CONFIG_PATH="${CONFIG_PATH:-configs/project_request.yaml}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

exec uv run --with pyyaml --with huggingface_hub \
  python tools/orchestrator/project_orchestrator.py "$ACTION" --config "$CONFIG_PATH"
