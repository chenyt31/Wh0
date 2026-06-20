#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  local ref="${3:-}"

  if [[ ! -d "$target_dir/.git" ]]; then
    rm -rf "$target_dir"
    git clone "$repo_url" "$target_dir"
  fi

  if ! git -C "$target_dir" fetch --all --tags; then
    echo "WARN: git fetch failed for $target_dir; using existing checkout." >&2
  fi
  if [[ -n "$ref" ]]; then
    git -C "$target_dir" checkout "$ref"
  else
    local default_branch
    default_branch="$(git -C "$target_dir" symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@')"
    git -C "$target_dir" checkout "$default_branch"
    if ! git -C "$target_dir" pull --ff-only; then
      echo "WARN: git pull failed for $target_dir; using existing checkout." >&2
    fi
  fi
}

clone_or_update "https://github.com/chenyt31/DiffSynth-Studio.git" "WM-H/third_party/DiffSynth-Studio" "a84251704224d6189f695ce72c3d834b7f84557c"
clone_or_update "https://github.com/ThunderVVV/HaWoR.git" "WM-H/third_party/HaWoR"

if [[ -d WM-H/third_party/HaWoR/thirdparty/DROID-SLAM ]]; then
  echo "==> Syncing DROID-SLAM submodules"
  if ! git -C WM-H/third_party/HaWoR/thirdparty/DROID-SLAM \
    -c http.version=HTTP/1.1 \
    -c http.postBuffer=524288000 \
    submodule update --init thirdparty/lietorch thirdparty/eigen; then
    echo "WARN: DROID-SLAM submodule sync failed; setup will use available local/system fallbacks." >&2
  fi
fi
