#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${SCREEN_PYTHON:-/home/xjy/zxvad-c04-combinations-v1/.venv/bin/python}
[[ -x "$PY" ]] || { echo "Existing pinned interpreter missing: $PY" >&2; exit 1; }
mkdir -p -- "$ROOT/outputs"
exec 9>"$ROOT/outputs/run.lock"
flock -n 9 || { echo 'An existing job owns this workspace lock.'; exit 0; }
printf '%s\n' "$BASHPID" > "$ROOT/outputs/runner.pid"
printf '%s\n' "$PY" > "$ROOT/runtime_python.txt"
phase=VERIFYING
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$ROOT/outputs/last_exit.txt"
  if ((rc!=0)); then "$PY" "$ROOT/scripts/state.py" FAILED "phase=$phase exit=$rc"; fi
}
trap finish EXIT
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Coordinator touches no CUDA context; each child sets its own single physical GPU.
export CUDA_VISIBLE_DEVICES=''
command -v nvidia-smi >/dev/null
"$PY" "$ROOT/scripts/verify_release.py"
"$PY" -m pip freeze > "$ROOT/outputs/pip_freeze.txt"
"$PY" - "$ROOT" <<'DUAL_DISK_CHECK'
import shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
assert free>15*1024**3,f'Need at least15GiB free for22 checkpoints; available {free/1024**3:.2f}GiB'
DUAL_DISK_CHECK
phase=PREPARING_DATA
"$PY" "$ROOT/scripts/state.py" "$phase"
"$PY" "$ROOT/scripts/prepare_lab_view.py" \
  --source "${SCREEN_SOURCE_FRAMES:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}" \
  --target-root "${SCREEN_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}"
phase=DUAL_GPU_EXPERIMENT
"$PY" "$ROOT/scripts/coordinator.py"
echo "Completed22 fits and66 AUROC rows: $ROOT/outputs/results.csv"
echo "Review bundle: $ROOT/outputs/review_bundle.zip"
