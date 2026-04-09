#!/usr/bin/env bash
set -euo pipefail

BRANCH="main"
REMOTE="origin"
MAX_FILE_MB=50  # reject any single file over this size

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

# ── Large file check ──────────────────────────────────────────────────────────
echo "Checking for large files..."
LARGE_FILES=$(git ls-files --others --exclude-standard -z | \
  xargs -0 -I{} sh -c \
  'size=$(du -sm "$1" 2>/dev/null | cut -f1); [ "$size" -gt '"$MAX_FILE_MB"' ] && echo "$1 (${size}MB)"' _ {})

if [ -n "$LARGE_FILES" ]; then
  echo
  echo "Error: the following untracked files exceed ${MAX_FILE_MB}MB and would be staged:"
  echo "$LARGE_FILES"
  echo
  echo "Add them to .gitignore before continuing."
  exit 1
fi

# ── Status preview ────────────────────────────────────────────────────────────
echo "Status before sync:"
git status --short
echo

# ── Check if remote has commits (initial push case) ───────────────────────────
REMOTE_EMPTY=0
if ! git ls-remote --exit-code "$REMOTE" "refs/heads/$BRANCH" >/dev/null 2>&1; then
  REMOTE_EMPTY=1
fi

# ── Stash local work before pull (only if remote has history) ────────────────
STASHED=0
if [ "$REMOTE_EMPTY" -eq 0 ]; then
  if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "Stashing local changes before pull..."
    git stash push -u -m "auto-stash before git.sh sync"
    STASHED=1
  fi

  echo "Fetching remote..."
  git fetch "$REMOTE"

  echo "Pulling with rebase..."
  git pull --rebase "$REMOTE" "$BRANCH"

  if [ "$STASHED" -eq 1 ]; then
    echo "Restoring stashed changes..."
    git stash pop || {
      echo
      echo "Stash pop produced conflicts. Resolve them manually, then run:"
      echo "  git add . && git rebase --continue"
      exit 1
    }
  fi
else
  echo "Remote branch not found — skipping pull (initial push)."
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

if [ "$REMOTE_EMPTY" -eq 1 ]; then
  git push -u "$REMOTE" "$BRANCH"
else
  git push "$REMOTE" "$BRANCH"
fi

echo
echo "Done. Pushed to $REMOTE/$BRANCH."