#!/usr/bin/env bash
# One-time environment setup for Wh0 (data + policy).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Wh0 setup (repo root: $ROOT)"
CONFIG_PATH="${CONFIG_PATH:-configs/project_request.yaml}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
SYNC_THIRD_PARTY="${SYNC_THIRD_PARTY:-}"
CUDA_EXTRA="${CUDA_EXTRA:-}"
WH0_EXTRAS="${WH0_EXTRAS:-}"
INSTALL_VLLM="${INSTALL_VLLM:-}"
INSTALL_VISUALIZATION="${INSTALL_VISUALIZATION:-}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-}"
PYPI_EXTRA_INDEX_URLS="${PYPI_EXTRA_INDEX_URLS:-}"
PYPI_FIND_LINKS="${PYPI_FIND_LINKS:-}"
UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
WH0_HFD_URL="${WH0_HFD_URL:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [[ -f "$CONFIG_PATH" ]]; then
  eval "$(
    uv run --no-project --with pyyaml python - "$CONFIG_PATH" <<'PY'
import shlex
import sys

import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

environment = config.get("environment", {})
sources = config.get("package_sources", {})

def emit(name, value):
    if value is None:
        value = ""
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, list):
        value = ",".join(str(item) for item in value)
    else:
        value = str(value)
    print(f": ${{{name}:={shlex.quote(value)}}}")

emit("PYTHON_VERSION", environment.get("python_version", "3.10"))
emit("SYNC_THIRD_PARTY", environment.get("sync_third_party", True))
emit("CUDA_EXTRA", environment.get("cuda_extra", "auto"))
emit("WH0_EXTRAS", environment.get("dependency_extras", ["policy", "wmh", "annotation", "dev"]))
emit("INSTALL_VLLM", environment.get("install_vllm", False))
emit("INSTALL_VISUALIZATION", environment.get("install_visualization", False))
emit("PYPI_INDEX_URL", sources.get("pypi_index_url", ""))
emit("PYPI_EXTRA_INDEX_URLS", sources.get("pypi_extra_index_urls", []))
emit("PYPI_FIND_LINKS", sources.get("pypi_find_links", []))
emit("UV_INDEX_STRATEGY", sources.get("uv_index_strategy", ""))
emit("PYTORCH_INDEX_URL", sources.get("pytorch_index_url", ""))
emit("WH0_HFD_URL", sources.get("hfd_script_url", "https://hf-mirror.com/hfd/hfd.sh"))
PY
  )"
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SYNC_THIRD_PARTY="${SYNC_THIRD_PARTY:-1}"
CUDA_EXTRA="${CUDA_EXTRA:-auto}"
WH0_EXTRAS="${WH0_EXTRAS:-policy,wmh,annotation,dev}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
INSTALL_VISUALIZATION="${INSTALL_VISUALIZATION:-0}"
WH0_HFD_URL="${WH0_HFD_URL:-https://hf-mirror.com/hfd/hfd.sh}"
export WH0_HFD_URL

if [[ "$SYNC_THIRD_PARTY" == "1" ]]; then
  echo "==> Syncing third-party repositories"
  bash tools/setup/sync_third_party.sh
fi

if [[ "${CLEAR_PACKAGE_PROXY:-0}" == "1" ]]; then
  echo "==> Clearing proxy variables for package installation"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

detect_cuda_extra() {
  if [[ "$CUDA_EXTRA" != "auto" ]]; then
    echo "$CUDA_EXTRA"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "none"
    return
  fi
  local cuda_version
  cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9]\).*/\1/p' | head -1)"
  case "$cuda_version" in
    12.8*|12.9*) echo "cu128" ;;
    12.4*|12.5*|12.6*|12.7*) echo "cu124" ;;
    12.1*|12.2*|12.3*) echo "cu121" ;;
    *) echo "none" ;;
  esac
}

SELECTED_CUDA_EXTRA="$(detect_cuda_extra)"
echo "==> Python environment (uv, CUDA extra: $SELECTED_CUDA_EXTRA)"
if [[ -d .venv ]]; then
  echo "==> Reusing existing .venv"
else
  uv venv --python "$PYTHON_VERSION"
fi

sync_cmd=(uv sync --inexact)
if [[ -n "$PYPI_INDEX_URL" ]]; then
  sync_cmd+=(--index-url "$PYPI_INDEX_URL")
fi
IFS=',' read -ra extra_index_urls <<< "$PYPI_EXTRA_INDEX_URLS"
for index_url in "${extra_index_urls[@]}"; do
  index_url="${index_url// /}"
  if [[ -n "$index_url" ]]; then
    sync_cmd+=(--extra-index-url "$index_url")
  fi
done
if [[ -n "$PYTORCH_INDEX_URL" ]]; then
  sync_cmd+=(--extra-index-url "$PYTORCH_INDEX_URL")
fi
IFS=',' read -ra find_links <<< "$PYPI_FIND_LINKS"
for find_link in "${find_links[@]}"; do
  find_link="${find_link// /}"
  if [[ -n "$find_link" ]]; then
    sync_cmd+=(--find-links "$find_link")
  fi
done
if [[ -n "$UV_INDEX_STRATEGY" ]]; then
  sync_cmd+=(--index-strategy "$UV_INDEX_STRATEGY")
fi
if [[ "$SELECTED_CUDA_EXTRA" != "none" && -n "$SELECTED_CUDA_EXTRA" ]]; then
  sync_cmd+=(--extra "$SELECTED_CUDA_EXTRA")
fi
IFS=',' read -ra extras <<< "$WH0_EXTRAS"
for extra in "${extras[@]}"; do
  extra="${extra// /}"
  if [[ -n "$extra" ]]; then
    sync_cmd+=(--extra "$extra")
  fi
done
if [[ "$INSTALL_VLLM" == "1" ]]; then
  sync_cmd+=(--extra wmh-vllm)
fi
if [[ "$INSTALL_VISUALIZATION" == "1" ]]; then
  sync_cmd+=(--extra visualization)
fi

echo "==> ${sync_cmd[*]}"
"${sync_cmd[@]}"

echo "==> Registering local editable packages without resolving duplicate dependencies"
uv pip install --no-deps -e vitra-wh0
if [[ -d WM-H/third_party/DiffSynth-Studio ]]; then
  uv pip install --no-deps -e WM-H/third_party/DiffSynth-Studio
fi

patch_droid_slam_for_torch2() {
  local droid_dir="$1"

  if [[ ! -e "$droid_dir/thirdparty/eigen/Eigen" && -d /usr/include/eigen3/Eigen ]]; then
    echo "==> Using system Eigen headers for DROID-SLAM"
    mkdir -p "$droid_dir/thirdparty/eigen"
    ln -sfn /usr/include/eigen3/Eigen "$droid_dir/thirdparty/eigen/Eigen"
  fi

  if [[ ! -d "$droid_dir/thirdparty/lietorch/lietorch" ]]; then
    echo "WARN: DROID-SLAM lietorch submodule is missing; build may fail." >&2
  fi

  if [[ -f "$droid_dir/src/correlation_kernels.cu" ]] && grep -qF "volume.type()" "$droid_dir/src/correlation_kernels.cu"; then
    perl -0pi -e 's/volume\.type\(\)/volume.scalar_type()/g' "$droid_dir/src/correlation_kernels.cu"
  fi
  if [[ -f "$droid_dir/src/altcorr_kernel.cu" ]] && grep -qF "fmap1.type()" "$droid_dir/src/altcorr_kernel.cu"; then
    perl -0pi -e 's/fmap1\.type\(\)/fmap1.scalar_type()/g' "$droid_dir/src/altcorr_kernel.cu"
  fi
  if [[ -f "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu" ]] && grep -Eq '\b(a|X)\.type\(\)' "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu"; then
    perl -0pi -e 's/\ba\.type\(\)/a.scalar_type()/g; s/\bX\.type\(\)/X.scalar_type()/g' \
      "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu"
  fi
  if [[ -f "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp" ]] && grep -Eq '\b(a|X)\.type\(\)' "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp"; then
    perl -0pi -e 's/\ba\.type\(\)/a.scalar_type()/g; s/\bX\.type\(\)/X.scalar_type()/g' \
      "$droid_dir/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp"
  fi
  if [[ -f "$droid_dir/thirdparty/lietorch/lietorch/include/dispatch.h" ]] && grep -qF "at::ScalarType _st = ::detail::scalar_type(the_type);" "$droid_dir/thirdparty/lietorch/lietorch/include/dispatch.h"; then
    perl -0pi -e 's/at::ScalarType _st = ::detail::scalar_type\(the_type\);/at::ScalarType _st = the_type;/g' \
      "$droid_dir/thirdparty/lietorch/lietorch/include/dispatch.h"
  fi
}

echo "==> HaWoR / DROID-SLAM (optional, needed for annotation)"
if [[ -d WM-H/third_party/HaWoR/thirdparty/DROID-SLAM ]]; then
  (
    cd WM-H/third_party/HaWoR/thirdparty/DROID-SLAM
    patch_droid_slam_for_torch2 "$PWD"
    python_bin="$ROOT/.venv/bin/python"
    if [[ ! -x "$python_bin" ]]; then
      python_bin="python"
    fi
    MAX_JOBS="${MAX_JOBS:-2}" UV_PROJECT="$ROOT" "$python_bin" setup.py install || echo "WARN: DROID-SLAM build failed — annotation SLAM stage may not work."
  )
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Validate with bash tools/validation/validate_environment.sh quick"
echo "  2. Fill configs/project_request.yaml and configs/validation.yaml"
echo "  3. Run bash scripts/run_from_config.sh all"
