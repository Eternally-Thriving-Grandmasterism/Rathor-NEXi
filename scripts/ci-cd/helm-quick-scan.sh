#!/usr/bin/env bash
# helm-quick-scan.sh – Local mercy Helm chart check

set -euo pipefail

echo "Mercy Helm quick scan..."

# Check for secrets in Helm files
if grep -r -E '(password|secret|key|token|api_key|private_key|secret_key)=[^ ]+' **/Chart.yaml **/values.yaml **/templates/*.yaml; then
  echo "Mercy shield: Potential secret leaked in Helm chart"
  exit 1
fi

# Basic Helm lint (if helm installed)
if command -v helm >/dev/null; then
  find . -name Chart.yaml -exec dirname {} \; | sort -u | while read chart_dir; do
    helm lint "$chart_dir" --strict || exit 1
  done
else
  echo "helm not installed – skipping lint (install via brew or script)"
fi

echo "Mercy Helm quick scan passed"
