#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
paddlex_runtime=${PADDLEX_ENV:-../TenderDoc-AI/.cache/paddle-env}
xiaomi_runtime=${XIAOMI_ENV:-../TDA/.venv}
for runtime in "$paddlex_runtime" "$xiaomi_runtime"; do
    test -x "$runtime/bin/python" || { echo "找不到推理运行环境 $runtime" >&2; exit 1; }
done
wheel_name=paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl
wheel=".cache/wheels/$wheel_name"
expected=4f7eb73b4a1cfa7c4ddd9fcce092c644cf3d10eb321418891e4cda7e401f8a7b
mkdir -p .cache/wheels
if [[ ! -f "$wheel" ]]; then
    curl -fL --retry 3 --continue-at - -o "$wheel.part" "https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/$wheel_name"
    echo "$expected  $wheel.part" | sha256sum -c -
    mv "$wheel.part" "$wheel"
else
    echo "$expected  $wheel" | sha256sum -c -
fi
if [[ ! -x .cache/layout-gpu/bin/python ]]; then uv venv --python "$xiaomi_runtime/bin/python" .cache/layout-gpu; fi
uv pip install --no-deps --python .cache/layout-gpu/bin/python "$wheel"
dest=$(.cache/layout-gpu/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
for runtime in "$paddlex_runtime" "$xiaomi_runtime"; do
    "$runtime/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
done >"$dest/reuse-local-runtime.pth.part"
mv "$dest/reuse-local-runtime.pth.part" "$dest/reuse-local-runtime.pth"
