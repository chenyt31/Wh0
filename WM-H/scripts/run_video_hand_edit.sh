#!/usr/bin/env bash
# Stage 3: replace human hands with robot hands in generated videos.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UV_PROJECT="${UV_PROJECT:-$(cd "$ROOT/.." && pwd)}"
uv run python wm_h/run_video_hand_edit.py --config configs/video_hand_edit.yaml "$@"
