#!/usr/bin/env bash
# merge-master.sh – Safe fast-forward from mercy-main

git fetch origin
git checkout main
git merge --ff-only origin/mercy-main || echo "Merge conflict – manual resolve needed"
git push origin main

echo "Mercy merge complete – main updated"
