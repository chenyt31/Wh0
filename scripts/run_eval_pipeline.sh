#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vitra-wh0"

PRESET="${PRESET:-annotation_default}"
if [[ $# -gt 0 ]]; then
  PRESET="$1"
  shift
fi

exec env UV_PROJECT="$ROOT" uv run python -m vitra.tools.human_prediction_pipeline \
  --preset-file ../configs/eval_pipeline_presets.json \
  --preset "$PRESET" \
  "$@"
