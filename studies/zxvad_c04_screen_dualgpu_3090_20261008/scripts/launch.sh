#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command -v flock >/dev/null || { echo 'Linux flock is required.' >&2; exit 1; }
mkdir -p -- "$ROOT/outputs"
if [[ -f "$ROOT/outputs/runner.pid" ]]; then
  pid=$(cat "$ROOT/outputs/runner.pid")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$ROOT/scripts/run.sh" >/dev/null; then
    echo "Already running: PID=$pid"; exit 0
  fi
fi
nohup bash "$ROOT/scripts/run.sh" >> "$ROOT/outputs/execution.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/outputs/launched.pid"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then tail -n 35 "$ROOT/outputs/execution.log"; exit 1; fi
echo "Background job dispatched: PID=$pid"
echo "Log: $ROOT/outputs/execution.log"
echo "Progress: bash $ROOT/scripts/progress.sh"
echo 'Two fixed queues run on physical GPU0 and GPU1 without a wall-time cutoff. SSH/local computer may disconnect after dispatch. Inspect both worker logs for increasing training steps.'
