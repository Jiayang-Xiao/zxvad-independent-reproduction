#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pid=$(cat "$ROOT/outputs/runner.pid" 2>/dev/null || true)
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -- "$ROOT/scripts/run.sh" >/dev/null; then
  echo "Runner active: PID=$pid"
else echo 'Runner not active'; fi
cat "$ROOT/outputs/state.json" 2>/dev/null || true
for arm in B T N Q TN TQ NQ TNQ; do
  echo "=== $arm ==="
  cat "$ROOT/outputs/$arm/state.json" 2>/dev/null || echo 'Not started'
done
echo '=== Last exit (historical on resume) ==='
cat "$ROOT/outputs/last_exit.txt" 2>/dev/null || true
echo '=== Latest log ==='
tail -n 12 "$ROOT/outputs/execution.log" 2>/dev/null || true
echo '=== Completed AUROC rows ==='
cat "$ROOT/outputs/results.csv" 2>/dev/null || true
