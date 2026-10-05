#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${REPRO_PYTHON:-/home/xjy/.conda/envs/aris-torch/bin/python}
SOURCE=${REPRO_SOURCE:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}
TARGET=${REPRO_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}
OUT=${REPRO_OUTPUT:-$ROOT/outputs/legacy_v1_seed17}
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=${REPRO_GPU:-0}
mkdir -p -- "$OUT"
exec >>"$OUT/execution.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$OUT/last_exit.txt"' EXIT
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This output already has an active run.'; exit 1; }
printf 'Waiting for GPU lock for visible device %s\n' "$CUDA_VISIBLE_DEVICES"
exec 8>"${TMPDIR:-/tmp}/zxvad-repro-gpu-${CUDA_VISIBLE_DEVICES//[^a-zA-Z0-9_-]/_}.lock"
flock 8
"$PY" "$ROOT/scripts/verify_repository.py"
"$PY" "$ROOT/scripts/collect_environment.py" --output "$OUT/environment.json"
"$PY" "$ROOT/src/verify_package.py"
"$PY" "$ROOT/src/preflight.py" --source "$SOURCE" --target-root "$TARGET" --output "$OUT" --device cuda:0
"$PY" "$ROOT/src/train.py" --source "$SOURCE" --output "$OUT" --variant baseline --device cuda:0
"$PY" "$ROOT/src/evaluate.py" --target-root "$TARGET" --output "$OUT" --device cuda:0
"$PY" "$ROOT/src/report.py" --output "$OUT" --private-bootstrap-repeats 2000
echo "Completed one legacy_v1 baseline fit and three AUROC rows: $OUT/summary.json"
