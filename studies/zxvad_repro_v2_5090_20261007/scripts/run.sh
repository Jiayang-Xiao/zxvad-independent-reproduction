#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PY=/environment/miniconda3/bin/python
[[ -x "$BASE_PY" ]] || BASE_PY=$(command -v python3)
mkdir -p -- "$ROOT/outputs"
exec 9>"$ROOT/outputs/run.lock"
if ! flock -n 9; then echo 'An existing job owns this workspace lock.'; exit 0; fi
printf '%s\n' "$BASHPID" > "$ROOT/outputs/runner.pid"
phase=VERIFYING
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$ROOT/outputs/last_exit.txt"
  if ((rc!=0 && rc!=20)); then "$BASE_PY" "$ROOT/scripts/state.py" FAILED "phase=$phase exit=$rc"; fi
}
trap finish EXIT
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$BASE_PY" "$ROOT/scripts/verify_release.py"
"$BASE_PY" "$ROOT/scripts/state.py" INVENTORY
"$BASE_PY" "$ROOT/scripts/inventory.py" --dataset-root "${V2_DATA_ROOT:-/home/featurize/datasets}" > "$ROOT/outputs/data_inventory.json"
if [[ ! -f "$ROOT/data_config.json" || -n "${V2_SOURCE_FRAMES:-}" || -n "${V2_TARGET_ROOT:-}" ]]; then
  config_args=(--data-root "${V2_TARGET_ROOT:-${V2_DATA_ROOT:-/home/featurize/datasets}}")
  [[ -z "${V2_SOURCE_FRAMES:-}" ]] || config_args+=(--source "$V2_SOURCE_FRAMES")
  "$BASE_PY" "$ROOT/scripts/configure_data.py" "${config_args[@]}"
fi
phase=INSTALLING_ENVIRONMENT
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then "$BASE_PY" -m venv "$ROOT/.venv"; fi
if [[ ! -f "$ROOT/.venv/environment_ready.json" ]]; then
  "$PY" -m pip install --upgrade pip --index-url https://pypi.org/simple
  "$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install -r "$ROOT/requirements.txt" --index-url https://pypi.org/simple
  "$PY" -m pip freeze > "$ROOT/outputs/pip_freeze.txt"
fi
phase=GPU_COMPATIBILITY
"$BASE_PY" "$ROOT/scripts/state.py" "$phase"
"$PY" "$ROOT/scripts/check_gpu.py"
cp -- "$ROOT/outputs/gpu_compatibility.json" "$ROOT/.venv/environment_ready.json"
phase=DATA_CHECK_AND_PIPELINE
"$PY" "$ROOT/src/pipeline.py" --workers 4 --evaluation-batch 16
phase=EXPORTING_RESULTS
"$PY" "$ROOT/scripts/export_results.py"
"$BASE_PY" "$ROOT/scripts/state.py" COMPLETED 'Four fits and twelve target AUROC rows exported.'
echo "Completed. Results: $ROOT/outputs/results.csv"
