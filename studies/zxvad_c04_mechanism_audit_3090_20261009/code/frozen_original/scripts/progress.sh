#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${SCREEN_PYTHON:-/home/xjy/zxvad-c04-combinations-v1/.venv/bin/python}
if [[ -f "$ROOT/runtime_python.txt" ]]; then PY=$(cat "$ROOT/runtime_python.txt"); fi
"$PY" "$ROOT/scripts/progress.py"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
