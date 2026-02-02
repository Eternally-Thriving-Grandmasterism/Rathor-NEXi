#!/usr/bin/env bash
# repo-hygiene.sh – Keep lattice clean

echo "Mercy hygiene check..."

# Remove ignored files if any leaked
git clean -fdx --dry-run

# Check for untracked files
untracked=$(git ls-files --others --exclude-standard)
if [[ -n "$untracked" ]]; then
  echo "Mercy note: Untracked files present:"
  echo "$untracked"
fi

# Check branch status
git status --short

echo "Mercy hygiene complete – lattice clean"
