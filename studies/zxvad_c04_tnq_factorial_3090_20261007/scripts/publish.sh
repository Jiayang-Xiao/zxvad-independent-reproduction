#!/usr/bin/env bash
# Run interactively only after outputs/summary.json says COMPLETED.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
exec 9>"$ROOT/outputs/run.lock"
flock -n 9 || { echo 'Runner still active; publish after completion.' >&2; exit 1; }
"$PY" "$ROOT/scripts/verify_release.py"
"$PY" "$ROOT/scripts/export_results.py"
command -v git >/dev/null
exec 8>"$ROOT/outputs/publish.lock"
flock -n 8 || { echo 'Another publication process owns this workspace.' >&2; exit 1; }

if ! command -v gh >/dev/null; then
  for binary in /home/xjy/zxvad-independent-reproduction/.private/gh-2.102.0/gh_2.102.0_linux_amd64/bin/gh /home/xjy/zxvad-independent-reproduction/.private/gh-deb-2.102.0/bin/gh; do
    if [[ -x "$binary" ]]; then export PATH="$(dirname -- "$binary"):$PATH"; break; fi
  done
fi
command -v gh >/dev/null || { echo 'GitHub CLI missing. Install gh from cli.github.com, then rerun this publication command; completed experiments remain saved.' >&2; exit 1; }
if ! gh auth status --hostname github.com >/dev/null 2>&1; then gh auth login --hostname github.com --git-protocol https --web; fi
[[ $(gh api user --jq .login) == Jiayang-Xiao ]] || { echo 'Expected GitHub account Jiayang-Xiao.' >&2; exit 1; }
CHECKOUT="$ROOT/.publish/repository"
BASE=3eb129ec5362ebd0575f9531d6ab178f4149e879
BRANCH=c04-tnq-factorial-3090-20261007
URL=https://github.com/Jiayang-Xiao/zxvad-independent-reproduction.git
STUDY=studies/zxvad_c04_tnq_factorial_3090_20261007
mkdir -p -- "$ROOT/.publish"
if [[ ! -d "$CHECKOUT/.git" ]]; then git -c 'credential.helper=!gh auth git-credential' clone "$URL" "$CHECKOUT"; fi
cd -- "$CHECKOUT"
[[ $(git config --get remote.origin.url) == "$URL" ]]
[[ -z $(git status --porcelain) ]] || { echo 'Publication checkout has local changes; inspect before retry.' >&2; exit 1; }
git -c 'credential.helper=!gh auth git-credential' fetch origin
if git rev-parse --verify "refs/heads/$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
  if git rev-parse --verify "refs/remotes/origin/$BRANCH" >/dev/null 2>&1; then git merge --ff-only "origin/$BRANCH"; fi
elif git rev-parse --verify "refs/remotes/origin/$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH" "$BASE"
fi
git merge-base --is-ancestor "$BASE" HEAD
"$PY" "$ROOT/scripts/materialize_public.py" --destination "$CHECKOUT/$STUDY"
git add -- "$STUDY"
if ! git diff --cached --quiet; then
  git -c user.name=Jiayang-Xiao -c user.email=Jiayang-Xiao@users.noreply.github.com commit -m 'Add paired c04 T-N-Q factorial and all final AUROC evidence on RTX3090'
fi
git -c 'credential.helper=!gh auth git-credential' push -u origin "HEAD:refs/heads/$BRANCH"
echo "Branch: $BRANCH"
echo "Commit: $(git rev-parse HEAD)"
echo "https://github.com/Jiayang-Xiao/zxvad-independent-reproduction/tree/$BRANCH/$STUDY"
