#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARENT=/home/xjy/zxvad-c04-screen-dualgpu-v1
STUDY=studies/zxvad_c04_mechanism_audit_3090_20261009
BRANCH=c04-mechanism-audit-3090-20261009
BASE=b57560884471b9807e2f475de9c38e757cfdefc0
HTTPS_REPO=https://github.com/Jiayang-Xiao/zxvad-independent-reproduction.git
CHECKOUT="$ROOT/.publish/repository"
STAGE=LOCAL_COMPLETION
trap 'rc=$?; printf "\nFAILED stage=%s line=%s exit=%s\n" "$STAGE" "$LINENO" "$rc" >&2' ERR
PY=$(cat "$ROOT/runtime_python.txt")
[[ -x "$PY" ]] || { echo 'Pinned interpreter missing.' >&2; exit 1; }
exec 9>"$ROOT/outputs/run.lock"
flock -n 9 || { echo 'Experiment runner is still active; upload after completion.' >&2; exit 1; }
exec 8>"$ROOT/outputs/publish.lock"
flock -n 8 || { echo 'Another publication process is active.' >&2; exit 1; }
"$PY" - "$ROOT" <<'CROSSCHECK_COMPLETE'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);o=r/'outputs'
assert json.loads((o/'state.json').read_text())['phase']=='COMPLETED'
assert (o/'last_exit.txt').read_text().strip()=='0'
CROSSCHECK_COMPLETE
"$PY" "$ROOT/verify_release.py"
"$PY" "$ROOT/export.py"
STAGE=PROCESS_PROXY
PROXY=http://127.0.0.1:18090
unset all_proxy ALL_PROXY
export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
if ! command -v gh >/dev/null; then
  for binary in /home/xjy/zxvad-independent-reproduction/.private/gh-2.102.0/gh_2.102.0_linux_amd64/bin/gh /home/xjy/zxvad-independent-reproduction/.private/gh-deb-2.102.0/bin/gh /home/xjy/zxvad-independent-reproduction/.private/gh-deb-2.102.0/usr/bin/gh; do
    if [[ -x "$binary" ]]; then export PATH="$(dirname -- "$binary"):$PATH"; break; fi
  done
fi
command -v gh >/dev/null || { echo 'Existing gh not found; use the same installation as the successful b575608 upload.' >&2; exit 1; }
STAGE=TUNNEL_CONNECTIVITY
"$PY" - "$PROXY" <<'CROSSCHECK_CONNECT'
import urllib.request,sys
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':sys.argv[1],'https':sys.argv[1]}))
with opener.open('https://github.com/login/device',timeout=30) as r:
    assert r.status==200,r.status
print('Forwarded proxy reaches GitHub: PASS')
CROSSCHECK_CONNECT
STAGE=GITHUB_AUTH
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo 'Complete GitHub device login in your browser; keep the local SSH forwarding window open.'
  gh auth login --hostname github.com --git-protocol https --web </dev/tty
fi
ACCOUNT=$(gh api user --jq .login)
[[ "$ACCOUNT" == Jiayang-Xiao ]] || { echo "Unexpected GitHub account: $ACCOUNT" >&2; exit 1; }
gitnet() { git -c "http.proxy=$PROXY" -c "http.${HTTPS_REPO}.proxy=$PROXY" -c credential.helper= -c 'credential.helper=!gh auth git-credential' "$@"; }
STAGE=PUBLICATION_CHECKOUT
mkdir -p -- "$ROOT/.publish"
INIT_MARKER="$ROOT/.publish/initializing.txt"
INIT_ID="$BASE $BRANCH $HTTPS_REPO"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  [[ ! -e "$CHECKOUT" ]] || { echo 'Non-Git publication destination exists; refusing overwrite.' >&2; exit 1; }
  printf '%s\n' "$INIT_ID" > "$INIT_MARKER"
  if [[ -d "$PARENT/.publish/repository/.git" ]]; then
    git clone --no-hardlinks --no-checkout -- "$PARENT/.publish/repository" "$CHECKOUT"
  else
    gitnet clone --no-checkout -- "$HTTPS_REPO" "$CHECKOUT"
  fi
fi
if [[ -f "$INIT_MARKER" ]]; then
  [[ $(cat "$INIT_MARKER") == "$INIT_ID" ]] || { echo 'Different publication initialization marker.' >&2; exit 1; }
  git -C "$CHECKOUT" remote set-url origin "$HTTPS_REPO"
  if ! git -C "$CHECKOUT" cat-file -e "$BASE^{commit}" 2>/dev/null; then
    gitnet -C "$CHECKOUT" fetch "$HTTPS_REPO" refs/heads/c04-screen-dualgpu-3090-20261008
  fi
  if [[ $(git -C "$CHECKOUT" symbolic-ref -q HEAD) != "refs/heads/$BRANCH" ]]; then
    # Resume only our untouched no-checkout clone, never switch another worktree.
    "$PY" - "$CHECKOUT" <<'CROSSCHECK_UNINITIALIZED'
import sys
from pathlib import Path
r=Path(sys.argv[1])
assert not any(p.is_file() and '.git' not in p.relative_to(r).parts for p in r.rglob('*')),'Initialization checkout contains working files; inspect it before retry.'
CROSSCHECK_UNINITIALIZED
    if git -C "$CHECKOUT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
      [[ $(git -C "$CHECKOUT" rev-parse "refs/heads/$BRANCH") == "$BASE" ]] || { echo 'Unexpected initial branch commit.' >&2; exit 1; }
      git -C "$CHECKOUT" checkout "$BRANCH"
    else
      git -C "$CHECKOUT" checkout -b "$BRANCH" "$BASE"
    fi
  fi
  rm -f -- "$INIT_MARKER"
fi
cd -- "$CHECKOUT"
[[ $(git rev-parse --show-toplevel) == "$CHECKOUT" ]] || { echo 'Unexpected Git root.' >&2; exit 1; }
[[ $(git symbolic-ref -q HEAD) == "refs/heads/$BRANCH" ]] || { echo 'Unexpected checkout branch; refusing automatic switch.' >&2; exit 1; }
git merge-base --is-ancestor "$BASE" HEAD
scope_check() {
"$PY" - "$STUDY" "$BASE" <<'CROSSCHECK_SCOPE'
import subprocess,sys
prefix=sys.argv[1]+'/'
for args in (['diff','--name-only',sys.argv[2],'HEAD'],['diff','--name-only'],['diff','--cached','--name-only'],['ls-files','--others','--exclude-standard']):
    names=subprocess.check_output(['git',*args],text=True).splitlines()
    assert all(n.startswith(prefix) for n in names),f'Unrelated checkout changes: {names}'
CROSSCHECK_SCOPE
}
scope_check
STAGE=REMOTE_BRANCH
remote=$(gitnet ls-remote --heads "$HTTPS_REPO" "refs/heads/$BRANCH" | awk '{print $1}')
if [[ -n "$remote" ]]; then
  gitnet fetch "$HTTPS_REPO" "refs/heads/$BRANCH"
  if [[ $(git rev-parse HEAD) != "$remote" ]] && git merge-base --is-ancestor HEAD "$remote"; then
    [[ -z $(git status --porcelain) ]] || { echo 'Remote is ahead and checkout has pending files; inspect before retry.' >&2; exit 1; }
    git merge --ff-only "$remote"
  else
    git merge-base --is-ancestor "$remote" HEAD || { echo 'Remote branch diverged; no force-push performed.' >&2; exit 1; }
  fi
fi
STAGE=FINAL_SCOPE
scope_check
STAGE=MATERIALIZING
"$PY" "$ROOT/export.py" --destination "$CHECKOUT/$STUDY"
git add -f -- "$STUDY"
if ! git diff --cached --quiet; then
  if ! git config user.name >/dev/null; then git config user.name Jiayang-Xiao; fi
  if ! git config user.email >/dev/null; then
    ACCOUNT_ID=$(gh api user --jq .id)
    git config user.email "${ACCOUNT_ID}+Jiayang-Xiao@users.noreply.github.com"
  fi
  git commit -m 'Add32 c04 source-heldout mechanism fits and all frozen AUROC evidence'
fi
STAGE=PUSHING
gitnet push -u "$HTTPS_REPO" "HEAD:refs/heads/$BRANCH"
LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(gitnet ls-remote --heads "$HTTPS_REPO" "refs/heads/$BRANCH" | awk '{print $1}')
[[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]] || { echo 'Remote HEAD differs; upload not confirmed.' >&2; exit 1; }
printf '\nPUBLICATION_CONFIRMED\nBranch: %s\nCommit: %s\n' "$BRANCH" "$LOCAL_HEAD"
printf 'https://github.com/Jiayang-Xiao/zxvad-independent-reproduction/tree/%s/%s\n' "$BRANCH" "$STUDY"
