#!/usr/bin/env bash
# Run interactively only after outputs/summary.json says COMPLETED.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/scripts/verify_release.py"
"$PY" "$ROOT/scripts/export_results.py"
command -v git >/dev/null
if ! command -v gh >/dev/null; then
  CACHE="$ROOT/.private/gh-2.102.0"
  mkdir -p -- "$CACHE"
  if [[ ! -x "$CACHE/gh_2.102.0_linux_amd64/bin/gh" ]]; then
    [[ $(uname -m) == x86_64 ]]
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 \
      https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz -o "$CACHE/gh.tar.gz"
    printf '%s  %s\n' bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386 "$CACHE/gh.tar.gz" | sha256sum -c -
    tar -xzf "$CACHE/gh.tar.gz" -C "$CACHE"
  fi
  export PATH="$CACHE/gh_2.102.0_linux_amd64/bin:$PATH"
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then gh auth login --hostname github.com --git-protocol https --web; fi
[[ $(gh api user --jq .login) == Jiayang-Xiao ]] || { echo 'Expected GitHub account Jiayang-Xiao.' >&2; exit 1; }
CHECKOUT="$ROOT/.publish/repository"
BASE=6eab454931182614639b17b68faf11dc42895428
BRANCH=repro-v2-5090-20261007
URL=https://github.com/Jiayang-Xiao/zxvad-independent-reproduction.git
STUDY=studies/zxvad_repro_v2_5090_20261007
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
  git -c user.name=Jiayang-Xiao -c user.email=Jiayang-Xiao@users.noreply.github.com commit -m 'Add paper-guided zxVAD V2 candidates and final AUROC evidence on RTX5090'
fi
git -c 'credential.helper=!gh auth git-credential' push -u origin "HEAD:refs/heads/$BRANCH"
echo "Branch: $BRANCH"
echo "Commit: $(git rev-parse HEAD)"
echo "https://github.com/Jiayang-Xiao/zxvad-independent-reproduction/tree/$BRANCH/$STUDY"
