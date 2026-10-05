#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${REPRO_PYTHON:-/home/xjy/.conda/envs/aris-torch/bin/python}
OUT=${REPRO_OUTPUT:-$ROOT/outputs/legacy_v1_seed17}
cd -- "$ROOT"
[[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || { echo 'Unexpected repository root.' >&2; exit 1; }
[[ $(git config --get remote.origin.url) == git@github.com:Jiayang-Xiao/zxvad-independent-reproduction.git ]] || { echo 'Unexpected origin.' >&2; exit 1; }
[[ $(git symbolic-ref -q HEAD) == refs/heads/main ]] || { echo 'Expected main.' >&2; exit 1; }
if ! git diff --cached --quiet; then echo 'Other staged changes exist.' >&2; exit 1; fi
"$PY" scripts/verify_repository.py
"$PY" scripts/export_run.py --output "$OUT"
git add -- evidence/fresh_legacy_v1_run
if ! git diff --cached --quiet; then git commit -m 'Add completed fresh run of independently implemented zxVAD legacy baseline'; fi
git push origin main
printf 'Branch: main\nCommit: %s\n' "$(git rev-parse HEAD)"
"$PY" scripts/render_reply.py --repo Jiayang-Xiao/zxvad-independent-reproduction
