#!/usr/bin/env bash
set -euo pipefail

BRANCH="main"
REMOTE="origin"
MAX_FILE_MB=50

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

echo "Current directory: $(pwd)"
echo

if [ ! -d ".git" ]; then
  echo "Error: not a git repository. Run this from the project root."
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: remote '$REMOTE' is not configured."
  exit 1
fi

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Error: unresolved merge conflicts detected."
  git status --short
  exit 1
fi

git branch -M "$BRANCH"

echo "Checking for large untracked files..."
LARGE_FILES=$(
  git ls-files --others --exclude-standard -z |
  while IFS= read -r -d '' file; do
    if [ -f "$file" ]; then
      size_mb=$(du -m "$file" 2>/dev/null | cut -f1)
      if [ -n "$size_mb" ] && [ "$size_mb" -gt "$MAX_FILE_MB" ]; then
        echo "$file (${size_mb}MB)"
      fi
    fi
  done
)

if [ -n "$LARGE_FILES" ]; then
  echo
  echo "Error: these untracked files exceed ${MAX_FILE_MB}MB:"
  echo "$LARGE_FILES"
  echo
  echo "Add them to .gitignore or remove them before running this script."
  exit 1
fi

echo
echo "Status before sync:"
git status --short
echo

REMOTE_EMPTY=0
if ! git ls-remote --exit-code "$REMOTE" "refs/heads/$BRANCH" >/dev/null 2>&1; then
  REMOTE_EMPTY=1
fi

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
    if ! git stash pop; then
      echo
      echo "Error: stash pop produced conflicts."
      echo "Resolve conflicts manually, then run:"
      echo "  git status"
      echo "  git add ."
      echo "  git commit -m \"your message\""
      echo "  git push"
      exit 1
    fi
  fi
else
  echo "Remote branch not found. Skipping pull."
fi

echo
echo "Staging all non-ignored changes..."
git add .

echo
echo "Staged changes:"
git diff --cached --name-status
echo

if git diff --cached --quiet; then
  echo "Nothing to commit."
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