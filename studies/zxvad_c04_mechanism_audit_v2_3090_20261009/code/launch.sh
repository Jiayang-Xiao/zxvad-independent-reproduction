#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v flock >/dev/null
mkdir -p -- "$ROOT/outputs"
if [[ -f "$ROOT/outputs/runner.pid" ]]; then
  pid=$(cat "$ROOT/outputs/runner.pid")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$ROOT/run.sh" >/dev/null; then
    echo "Already running: PID=$pid"; exit 0
  fi
fi
nohup bash "$ROOT/run.sh" >> "$ROOT/outputs/execution.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/outputs/launched.pid"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  if [[ $(cat "$ROOT/outputs/last_exit.txt" 2>/dev/null || true) == 0 ]] && grep -Eq '"phase"[[:space:]]*:[[:space:]]*"COMPLETED"' "$ROOT/outputs/state.json"; then
    echo 'Already completed; frozen results validated without new GPU work.'; exit 0
  fi
  tail -n 40 "$ROOT/outputs/execution.log"; exit 1
fi
echo "Background job dispatched: PID=$pid"
echo "Progress: bash $ROOT/progress.sh"
echo '32 fresh fits run as16 sequential fits per GPU on GPU0 and GPU1; no time cutoff.'
echo 'You may disconnect SSH/local computer after this dispatch. Source-only diagnostics precede all target evaluation.'
echo 'An occupied card waits without touching other jobs. Check increasing steps in both worker logs.'
