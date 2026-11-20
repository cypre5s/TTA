#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
exec scripts/run_pilot.sh annotate
