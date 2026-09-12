#!/bin/bash
# Mirror the operating tree into the git repo.
#
#   ~/jettank      operating tree - what runs, what gets deployed to the robot
#   ~/git/jettank  git repo       - what gets published
#
# The split is deliberate: the operating tree accumulates things that must
# never be published (credentials, the face database, 61MB voice models, a
# venv), and keeping the repo a separate directory means a stray file cannot
# be swept in by an over-broad `git add`.
#
# This script refuses to sync if it finds anything that looks like a secret.
set -euo pipefail

SRC="${JETTANK_SRC:-$HOME/jettank}"
DST="${JETTANK_GIT:-$HOME/git/jettank}"

[ -d "$SRC" ] || { echo "operating tree not found: $SRC" >&2; exit 1; }
[ -d "$DST/.git" ] || { echo "git repo not found: $DST" >&2; exit 1; }

# Things that exist to run the robot, not to be published.
EXCLUDES=(
  --exclude '.git'  --exclude '__pycache__' --exclude '*.pyc'
  --exclude '.venv' --exclude 'voices'      --exclude '*.env'
  --exclude '*_faces.json' --exclude '*.log' --exclude 'vendor'
)

# --- secret guard -----------------------------------------------------------
# Checked on the *source*, before anything is copied. Patterns match the shape
# of real credentials, not the words "key" or "password", so that config field
# names and documentation do not trip it.
PATTERN='sk-ant-[a-zA-Z0-9_-]{10}|ghp_[a-zA-Z0-9]{20}|github_pat_[a-zA-Z0-9_]{20}|glpat-[a-zA-Z0-9_-]{15}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|AKIA[0-9A-Z]{16}'

hits=$(grep -rlnE "$PATTERN" "$SRC" \
         --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.venv \
         --exclude-dir=voices --exclude-dir=vendor 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "REFUSING TO SYNC - credential-shaped content found in:" >&2
  echo "$hits" | sed 's/^/  /' >&2
  echo >&2
  echo "Move it to ~/.jettank.env (mode 600, outside every repo) first." >&2
  exit 1
fi

rsync -a --delete "${EXCLUDES[@]}" "$SRC/" "$DST/"

cd "$DST"
echo "synced $SRC -> $DST"
# --porcelain, not `git diff`: diff does not see untracked files, so a brand
# new module would be reported as "no changes".
if [ -z "$(git status --porcelain)" ]; then
  echo "no changes"
else
  git status --short
fi
