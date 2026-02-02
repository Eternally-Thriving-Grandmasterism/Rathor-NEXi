#!/usr/bin/env bash
# k8s-quick-scan.sh – Local Kubernetes mercy scan

set -euo pipefail

echo "Mercy Kubernetes quick scan..."

# Check manifests for secrets
if grep -r -E '(password|secret|key|token|api_key|private_key|secret_key)=[^ ]+' **/*.yaml **/*.yml; then
  echo "Mercy shield: Potential secret leaked in Kubernetes manifest"
  exit 1
fi

# Basic lint (kube-linter if installed)
if command -v kube-linter >/dev/null; then
  kube-linter lint **/*.yaml **/*.yml || exit 1
else
  echo "kube-linter not installed – skipping (install via brew/cargo)"
fi

echo "Mercy K8s quick scan passed"
