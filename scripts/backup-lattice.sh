#!/usr/bin/env bash
# backup-lattice.sh – Mercy-encrypted local backup

set -euo pipefail

BACKUP_DIR="\( HOME/.nexi-backups/ \)(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Mercy backup initiated..."

git archive --format=tar main | gzip > "$BACKUP_DIR/nexi-main.tar.gz"

# If gpg installed, encrypt
if command -v gpg >/dev/null; then
  gpg --symmetric --cipher-algo AES256 "$BACKUP_DIR/nexi-main.tar.gz"
  rm "$BACKUP_DIR/nexi-main.tar.gz"
  echo "Encrypted backup created: $BACKUP_DIR/nexi-main.tar.gz.gpg"
else
  echo "Backup created (unencrypted): $BACKUP_DIR/nexi-main.tar.gz"
  echo "Install gpg for mercy-encrypted backups"
fi

echo "Mercy backup complete"
