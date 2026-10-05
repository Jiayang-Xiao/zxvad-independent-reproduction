#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${REPRO_PYTHON:-/home/xjy/.conda/envs/aris-torch/bin/python}
REPO=Jiayang-Xiao/zxvad-independent-reproduction
cd -- "$ROOT"
[[ -x "$PY" ]] || { echo "Python missing: $PY" >&2; exit 1; }
"$PY" scripts/verify_repository.py
"$PY" scripts/audit_legacy_scores.py
# Download the official CLI to a user-owned private cache if the server lacks gh.
if ! command -v gh >/dev/null 2>&1; then
  command -v curl >/dev/null || { echo 'curl or an installed gh is required.' >&2; exit 1; }
  [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'This portable gh bootstrap requires Linux x86_64.' >&2; exit 1; }
  CACHE="$ROOT/.private/gh-2.102.0"
  mkdir -p -- "$CACHE"
  ARCHIVE="$CACHE/gh.tar.gz"
  if [[ ! -x "$CACHE/gh_2.102.0_linux_amd64/bin/gh" ]]; then
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 240 \
      'https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz' -o "$ARCHIVE"
    printf '%s  %s\n' 'bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386' "$ARCHIVE" | sha256sum -c -
    tar -xzf "$ARCHIVE" -C "$CACHE"
  fi
  export PATH="$CACHE/gh_2.102.0_linux_amd64/bin:$PATH"
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  echo 'Authenticate using the browser/device code shown below; the email is not sent by this script.'
  gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key < /dev/tty
fi
LOGIN=$(gh api user --jq .login)
[[ "$LOGIN" == Jiayang-Xiao ]] || { echo "Authenticated account is $LOGIN; expected Jiayang-Xiao." >&2; exit 1; }
if [[ ! -d .git ]]; then git init; fi
[[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || { echo 'Refusing to publish from a parent repository.' >&2; exit 1; }
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then git symbolic-ref HEAD refs/heads/main; fi
if ! git config user.name >/dev/null; then git config user.name Jiayang-Xiao; fi
if ! git config user.email >/dev/null; then git config user.email '92557165+Jiayang-Xiao@users.noreply.github.com'; fi
[[ $(git symbolic-ref -q HEAD) == refs/heads/main ]] || { echo 'Initial publication expects the main branch.' >&2; exit 1; }
if ! git diff --cached --quiet; then echo 'Unexpected staged changes; inspect them before rerunning.' >&2; exit 1; fi
"$PY" scripts/collect_environment.py --output evidence/publication_host_environment.json
git add -- README.md .gitignore src scripts docs provenance evidence release_manifest.json
if ! git diff --cached --quiet; then git commit -m 'Publish independently implemented zxVAD legacy baseline for reproduction review'; fi
if gh api "repos/$REPO" >/dev/null 2>&1; then
  VISIBILITY=$(gh api "repos/$REPO" --jq .visibility)
  [[ "$VISIBILITY" == public ]] || { echo 'The named repository exists but is not public; refusing to change its visibility.' >&2; exit 1; }
else
  gh repo create "$REPO" --public --description 'Independent zxVAD implementation, baseline evidence, and author-assisted reproduction review'
fi
REMOTE="git@github.com:$REPO.git"
if git config --get remote.origin.url >/dev/null 2>&1; then
  [[ $(git config --get remote.origin.url) == "$REMOTE" ]] || { echo 'origin differs from the intended new repository.' >&2; exit 1; }
else git remote add origin "$REMOTE"; fi
git push -u origin main
COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git ls-remote origin refs/heads/main | awk '{print $1}')
[[ "$COMMIT" == "$REMOTE_COMMIT" ]] || { echo 'Remote commit mismatch.' >&2; exit 1; }
"$PY" scripts/render_reply.py --repo "$REPO"
printf 'Repository: https://github.com/%s\nBranch: main\nCommit: %s\n' "$REPO" "$COMMIT"
echo "Reply file: $ROOT/.private/author_reply_ready.txt"
cat -- "$ROOT/.private/author_reply_ready.txt"
