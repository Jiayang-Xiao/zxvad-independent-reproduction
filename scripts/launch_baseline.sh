#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${REPRO_PYTHON:-/home/xjy/.conda/envs/aris-torch/bin/python}
OUT=${REPRO_OUTPUT:-$ROOT/outputs/legacy_v1_seed17}
SESSION=${REPRO_SESSION:-zxvad-repro-legacy-v1}
command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required.' >&2; exit 1; }
[[ -x "$PY" ]] || { echo "Python missing: $PY" >&2; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"; exit 0
fi
mkdir -p -- "$OUT"
# Encode an explicit environment rather than relying on tmux's old server environment.
DRIVER="$OUT/tmux_driver.sh"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'export REPRO_PYTHON=%q\n' "$PY"
  printf 'export REPRO_OUTPUT=%q\n' "$OUT"
  printf 'export REPRO_SOURCE=%q\n' "${REPRO_SOURCE:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}"
  printf 'export REPRO_TARGET_ROOT=%q\n' "${REPRO_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}"
  printf 'export REPRO_GPU=%q\n' "${REPRO_GPU:-0}"
  printf 'exec bash %q\n' "$ROOT/scripts/run_baseline.sh"
} > "$DRIVER"
printf -v CMD 'bash %q' "$DRIVER"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$CMD"
tmux set-option -t "=$SESSION" remain-on-exit on
echo "Detached tmux session created: $SESSION"
echo "Log: $OUT/execution.log"
echo 'The SSH connection may now close while the server remains running.'
