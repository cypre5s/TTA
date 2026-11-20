#!/usr/bin/env bash
set -eu
uname -sr
lscpu | grep 'Model name' | head -1
.venv/bin/python -V
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h .
