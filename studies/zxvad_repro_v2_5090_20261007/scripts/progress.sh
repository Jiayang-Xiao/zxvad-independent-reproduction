#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/outputs"
if [[ -f "$OUT/runner.pid" ]]; then
  pid=$(cat "$OUT/runner.pid")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$ROOT/scripts/run.sh" >/dev/null; then echo "Runner alive: PID=$pid"; else echo 'Runner not active'; fi
fi
[[ ! -f "$OUT/state.json" ]] || cat "$OUT/state.json"
for candidate in c01 c02 c03 c04; do
  if [[ -f "$OUT/$candidate/state.json" ]]; then echo "$candidate"; cat "$OUT/$candidate/state.json"; fi
done
[[ ! -f "$OUT/last_exit.txt" ]] || { echo 'Last exit:'; cat "$OUT/last_exit.txt"; }
[[ ! -f "$OUT/execution.log" ]] || tail -n 15 "$OUT/execution.log"
[[ ! -f "$OUT/results.csv" ]] || cat "$OUT/results.csv"
