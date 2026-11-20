#!/usr/bin/env bash
# Run the post-pilot measurements serially on the local GPU.
set -eu
cd "$(dirname "$0")/.."
exec 8>logs/local/benchmark.lock
flock -n 8 || { echo '另一个本机测试已在运行'; exit 1; }
if [[ ${1:-} == --after-pilot ]]; then
    while ! .venv/bin/python - <<'PY'
from pathlib import Path
import sys
root=Path.cwd()
for f,n in [("data/tables/predictions",1000),("data/layout/predictions",13247)]:
    if sum(1 for p in Path(f).glob("*.json"))<n:sys.exit(1)
for p in Path("/proc").glob("[0-9]*"):
    try:
        if (p/"cwd").resolve()!=root:continue
        args=(p/"cmdline").read_bytes().split(b"\0")
        if any(n in args for n in (b"scripts/cloud_ocr.py",b"scripts/layout.py")):sys.exit(1)
    except OSError:pass
PY
    do sleep 10; done
fi
py=${XIAOMI_PYTHON:-../TDA/.venv/bin/python}
failures=0
scripts/local_paddle.sh scripts/bench_layout_precision.py >logs/local/precision_isolated.log 2>&1 || failures=1
"$py" -u -m scripts.bench_deep_decode --inputs reports/performance/xiaomi_holdout_inputs.json \
    --output reports/performance/xiaomi_isolated_final.json \
    --modes original8 original8_graph original8_stack original12 original12_stack deployed_graph norm_graph original8 \
    >logs/local/isolated_final.log 2>&1 || failures=1
"$py" -u -m scripts.bench_deep_decode --inputs reports/performance/xiaomi_holdout_inputs.json \
    --output reports/performance/xiaomi_hybrid.json --modes original8 original8_hybrid original8_hybrid original8 \
    >logs/local/hybrid.log 2>&1 || failures=1
"$py" -u -m scripts.bench_deep_decode --inputs reports/performance/xiaomi_tail_inputs.json \
    --output reports/performance/xiaomi_tail.json --modes original8 original8_hybrid \
    >logs/local/long_tables.log 2>&1 || failures=1
"$py" -u -m scripts.bench_length_groups >logs/local/length_groups.log 2>&1 || failures=1
"$py" -u -m scripts.profile_decode >logs/local/profile_decode.log 2>&1 || failures=1
exit "$failures"
