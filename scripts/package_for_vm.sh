#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

default_name="biomni_deploy_$(date +%Y%m%d_%H%M%S).tar.gz"
archive_name="$default_name"
include_data_lake=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-data-lake)
      include_data_lake=true
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/package_for_vm.sh [archive_name.tar.gz] [--with-data-lake]"
      echo ""
      echo "Examples:"
      echo "  ./scripts/package_for_vm.sh"
      echo "  ./scripts/package_for_vm.sh biomni_release.tar.gz"
      echo "  ./scripts/package_for_vm.sh --with-data-lake"
      echo "  ./scripts/package_for_vm.sh biomni_full.tar.gz --with-data-lake"
      exit 0
      ;;
    *)
      archive_name="$1"
      shift
      ;;
  esac
done

if [[ "$archive_name" != *.tar.gz ]]; then
    archive_name="${archive_name}.tar.gz"
fi

if [[ "$include_data_lake" == true ]]; then
  echo "Creating deployment archive WITH data lake: $archive_name"
  tar -czf "$archive_name" \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.tar.gz' \
    --exclude='runs' \
    --exclude='.env' \
    .
else
  echo "Creating deployment archive (code-only, without data lake): $archive_name"
  tar -czf "$archive_name" \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.tar.gz' \
    --exclude='data' \
    --exclude='runs' \
    --exclude='.env' \
    .
fi

echo "Done: $archive_name"
ls -lh "$archive_name"

echo ""
echo "Next:"
echo "  scp $archive_name <VM_USER>@<VM_PUBLIC_IP>:~"