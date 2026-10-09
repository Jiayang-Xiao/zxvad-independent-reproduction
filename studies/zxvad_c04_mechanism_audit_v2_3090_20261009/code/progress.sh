#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="$ROOT/outputs"
pid=$(cat "$OUT/runner.pid" 2>/dev/null || true)
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$ROOT/run.sh" >/dev/null; then
  echo "Runner active: PID=$pid"
else
  echo 'Runner not active'
fi
cat "$OUT/state.json" 2>/dev/null || true
PARENT=/home/xjy/zxvad-c04-screen-dualgpu-v1
PY=$(cat "$PARENT/runtime_python.txt")
"$PY" - "$ROOT" <<'MECHANISM_PROGRESS'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from spec import QUEUES
out=root/'outputs'
for lane,queue in QUEUES.items():
  print(f'\n=== GPU{lane}: {len(queue)} planned fits ===')
  gate=out/'workers'/('gpu'+lane)/'optimizer_regression.json'
  if gate.is_file():
    value=json.loads(gate.read_text()); print(f"Three-step optimizer gate: {value.get('status')} ({value.get('steps')} disposable source updates)")
  else: print('Three-step optimizer gate: pending (required before fits)')
  print('arm                              step/5000  train phase       source probe  target arrays')
  for arm in queue:
    dst=out/arm
    try:state=json.loads((dst/'state.json').read_text())
    except (OSError,ValueError):state={}
    probes=all((dst/name).is_file() for name in ('source_probe.json','source_probe.npz','calibration.json'))
    targets=sum((dst/(name+'.npz')).is_file() and (dst/(name+'.json')).is_file() for name in ('ped1','ped2','avenue'))
    print(f"{arm:32} {state.get('step',0):>4}/5000  {state.get('phase','NOT_STARTED'):17} {'written' if probes else 'pending':12} {targets}/3")
selection=out/'source_selection.json'
if selection.is_file():
  value=json.loads(selection.read_text());print('\nSource-only selection:')
  print(json.dumps(value.get('chosen_arms',{}),ensure_ascii=False,indent=2))
MECHANISM_PROGRESS
for lane in 0 1; do
  printf '\n=== GPU%s worker ===\n' "$lane"
  cat "$OUT/workers/gpu$lane/state.json" 2>/dev/null || true
  tail -n 4 "$OUT/workers/gpu$lane/execution.log" 2>/dev/null || true
done
printf '\n=== Coordinator log ===\n'
tail -n 12 "$OUT/execution.log" 2>/dev/null || true
printf '\n=== Last exit (historical on resume) ===\n'
cat "$OUT/last_exit.txt" 2>/dev/null || true
printf '\n=== Primary PSNR AUROC rows (available after reporting) ===\n'
cat "$OUT/primary_results.csv" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
