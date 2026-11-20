#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
xiaomi_runtime=${XIAOMI_ENV:-../TDA/.venv}
nvidia_libs=$(find "$(realpath "$xiaomi_runtime/lib/python3.12/site-packages/nvidia")" -maxdepth 3 -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$nvidia_libs:${LD_LIBRARY_PATH:-}"
export FLAGS_allocator_strategy=auto_growth
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
exec .cache/layout-gpu/bin/python "$@"
