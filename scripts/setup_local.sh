#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/{raw/pdf,metadata,rendered,layout/{predictions,previews},tables/{crops,predictions},otsl/raw,annotation/{packets,raw}} logs/{local,cloud,annotation} reports database
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.13; fi
uv pip install --python .venv/bin/python -r configs/local-requirements.lock
