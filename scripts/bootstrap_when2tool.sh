#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="\${ROOT_DIR}/.cache/when2tool"
UPSTREAM_URL="https://github.com/Trustworthy-ML-Lab/when2tool.git"
PINNED_COMMIT="ef0cf1ea72eb8acdfd3fb1251c4b6a9a0131a23e"

mkdir -p "\${ROOT_DIR}/.cache"
if [[ ! -d "\${UPSTREAM_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "\${UPSTREAM_URL}" "\${UPSTREAM_DIR}"
fi

git -C "\${UPSTREAM_DIR}" fetch --quiet origin "\${PINNED_COMMIT}"
git -C "\${UPSTREAM_DIR}" checkout --quiet --detach "\${PINNED_COMMIT}"

ACTUAL_COMMIT="$(git -C "\${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${PINNED_COMMIT}" ]]; then
  echo "ERROR: upstream checkout mismatch: \${ACTUAL_COMMIT}" >&2
  exit 2
fi

echo "Pinned When2Tool source ready:"
echo "  \${UPSTREAM_URL}"
echo "  \${ACTUAL_COMMIT}"
echo "  \${UPSTREAM_DIR}"
