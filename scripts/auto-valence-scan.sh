#!/usr/bin/env bash
# auto-valence-scan.sh – Mercy pre-commit scanner

set -euo pipefail

MESSAGE="\( {1:- \)(git log -1 --pretty=%B)}"
FILES=$(git diff --cached --name-only)

BAD_PATTERNS=(
  "harm" "coercion" "weapon" "destroy" "enslave" "control"
  "entropy" "bleed" "drift" "corrupt" "exploit"
)

for pattern in "${BAD_PATTERNS[@]}"; do
  if echo "$MESSAGE" | grep -iq "$pattern"; then
    echo "Mercy shield: Valence drop detected in commit message — pattern: $pattern"
    exit 1
  fi
done

for file in $FILES; do
  if [[ "\( file" =~ \.rs \)|\.metta\( |\.md \) ]]; then
    if grep -iqE "\( (IFS=\| ; echo " \){BAD_PATTERNS[*]}")" "$file"; then
      echo "Mercy shield: Valence drop detected in file: $file"
      exit 1
    fi
  fi
done

echo "Mercy-approved: Valence scan passed"
exit 0
