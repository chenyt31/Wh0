#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LEVEL="${1:-quick}"
CONFIG_PATH="${CONFIG_PATH:-configs/validation.yaml}"

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL uv is required. Install uv before validation." >&2
  exit 1
fi

exec uv run --no-project --with pyyaml --with numpy --with pillow --with scipy \
  python tools/validation/validate_environment.py "$LEVEL" --config "$CONFIG_PATH"
