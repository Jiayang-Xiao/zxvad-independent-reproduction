#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PY=${COMBO_BASE_PYTHON:-/home/xjy/.conda/envs/aris-torch/bin/python}
[[ -x "$BASE_PY" ]] || BASE_PY=$(command -v python3)
mkdir -p -- "$ROOT/outputs"
exec 9>"$ROOT/outputs/run.lock"
flock -n 9 || { echo 'An existing job owns this workspace lock.'; exit 0; }
printf '%s\n' "$BASHPID" > "$ROOT/outputs/runner.pid"
phase=VERIFYING
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$ROOT/outputs/last_exit.txt"
  if ((rc!=0)); then "$BASE_PY" "$ROOT/scripts/state.py" FAILED "phase=$phase exit=$rc"; fi
}
trap finish EXIT
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${COMBO_GPU:-1}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo 'COMBO_GPU must be one physical numeric GPU index.' >&2; exit 1; }
"$BASE_PY" "$ROOT/scripts/verify_release.py"
phase=INSTALLING_ENVIRONMENT
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then "$BASE_PY" -m venv "$ROOT/.venv"; fi
if [[ ! -f "$ROOT/.venv/environment_ready.json" ]]; then
  "$PY" -m pip install --upgrade pip --index-url https://pypi.org/simple
  "$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install -r "$ROOT/requirements.txt" --index-url https://pypi.org/simple
fi
"$PY" -m pip freeze > "$ROOT/outputs/pip_freeze.txt"
phase=PREPARING_DATA
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
"$PY" "$ROOT/scripts/prepare_lab_view.py" \
  --source "${COMBO_SOURCE_FRAMES:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}" \
  --target-root "${COMBO_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}"
phase=WAITING_FOR_GPU
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
"$BASE_PY" "$ROOT/scripts/wait_gpu.py" --gpu "$CUDA_VISIBLE_DEVICES"
phase=GPU_COMPATIBILITY
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
"$PY" "$ROOT/scripts/check_gpu.py"
cp -- "$ROOT/outputs/gpu_compatibility.json" "$ROOT/.venv/environment_ready.json"
phase=SOURCE_PREFLIGHT_AND_FACTORIAL
"$PY" "$ROOT/src/pipeline.py" --workers 4 --evaluation-batch 16
phase=EXPORTING_RESULTS
"$PY" "$ROOT/scripts/export_results.py"
"$BASE_PY" "$ROOT/scripts/state.py" COMPLETED 'Eight paired fits and twenty-four AUROC rows exported.'
echo "Completed. Results: $ROOT/outputs/results.csv"
