#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARENT=/home/xjy/zxvad-c04-screen-dualgpu-v1
PY=$(cat "$PARENT/runtime_python.txt")
[[ -x "$PY" ]] || { echo "Frozen interpreter missing: $PY" >&2; exit 1; }
mkdir -p -- "$ROOT/outputs"
exec 9>"$ROOT/outputs/run.lock"
flock -n 9 || { echo 'Another runner owns this workspace; no duplicate started.'; exit 0; }
printf '%s\n' "$BASHPID" > "$ROOT/outputs/runner.pid"
phase=VERIFYING
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$ROOT/outputs/last_exit.txt"
  if ((rc!=0)); then "$PY" "$ROOT/state.py" FAILED "phase=$phase exit=$rc" || true; fi
}
trap finish EXIT
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
command -v nvidia-smi >/dev/null
"$PY" "$ROOT/state.py" VERIFYING
"$PY" "$ROOT/verify_release.py"
"$PY" "$ROOT/regression.py"
"$PY" - "$ROOT" <<'MECHANISM_DISK'
import shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
assert free>25*1024**3,f'Need25GiB free for32 fresh final/resumable checkpoints and evidence; available {free/1024**3:.2f}GiB'
MECHANISM_DISK
phase=MECHANISM_AUDIT
"$PY" "$ROOT/coordinate.py" --workers 4 --evaluation-batch 16
phase=REPORTING
"$PY" "$ROOT/state.py" REPORTING
"$PY" "$ROOT/report.py"
phase=EXPORTING
"$PY" "$ROOT/state.py" EXPORTING
"$PY" "$ROOT/export.py"
"$PY" "$ROOT/state.py" COMPLETED '32 new5000-step fits,32 heldout source probes,96 primary AUROC rows and672 fixed readout AUROC rows.'
echo "Completed: $ROOT/outputs/results.csv"
echo "Review bundle: $ROOT/outputs/review_bundle.zip"
