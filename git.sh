#!/usr/bin/env bash
set -euo pipefail

BRANCH="main"
REMOTE="origin"

# ── Commit message ────────────────────────────────────────────────────────────
if [ $# -ge 1 ] && [ -n "$1" ]; then
  MSG="$1"
else
  printf "Commit message: "
  read -r MSG
  if [ -z "$MSG" ]; then
    echo "Error: commit message cannot be empty."
    exit 1
  fi
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "Current directory: $(pwd)"
echo

if [ ! -d ".git" ]; then
  echo "Error: not a git repository. Run this from your project root."
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: remote '$REMOTE' is not configured."
  exit 1
fi

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Error: unresolved merge conflicts detected. Resolve them before pushing."
  exit 1
fi

git branch -M "$BRANCH"

# ── Status preview ────────────────────────────────────────────────────────────
echo "Status before sync:"
git status --short
echo

# ── Stash local work before pull ──────────────────────────────────────────────
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "Stashing local changes before pull..."
  git stash push -u -m "auto-stash before git.sh sync"
  STASHED=1
fi

# ── Sync with remote ──────────────────────────────────────────────────────────
echo "Fetching remote..."
git fetch "$REMOTE"

echo "Pulling with rebase..."
git pull --rebase "$REMOTE" "$BRANCH"

# ── Restore stash ─────────────────────────────────────────────────────────────
if [ "$STASHED" -eq 1 ]; then
  echo "Restoring stashed changes..."
  git stash pop || {
    echo
    echo "Stash pop produced conflicts. Resolve them manually, then run:"
    echo "  git add . && git rebase --continue"
    exit 1
  }
fi

# ── Stage, commit, push ───────────────────────────────────────────────────────
echo
echo "Staging all changes..."
git add .

echo
echo "Staged changes:"
git diff --cached --name-status
echo

if git diff --cached --quiet; then
  echo "Nothing to commit — working tree is clean."
  exit 0
fi

git commit -m "$MSG"
git push "$REMOTE" "$BRANCH"

echo
echo "Done. Pushed to $REMOTE/$BRANCH."
