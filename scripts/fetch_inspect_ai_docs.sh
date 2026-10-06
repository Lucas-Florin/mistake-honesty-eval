#!/usr/bin/env bash
# Downloads the Inspect AI llms.txt-style docs used as coding-agent context.
# Overwrites any existing copies under claude_context/library_docs/inspect_ai/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/claude_context/library_docs/inspect_ai"
BASE_URL="https://inspect.aisi.org.uk"

mkdir -p "${DEST_DIR}"

declare -A FILES=(
  ["llms.txt"]="inspect-index.md"
  ["llms-guide.txt"]="inspect-guide.md"
  ["llms-full.txt"]="inspect-full.md"
)

for remote in "${!FILES[@]}"; do
  local_name="${FILES[$remote]}"
  dest_path="${DEST_DIR}/${local_name}"
  url="${BASE_URL}/${remote}"
  echo "Downloading ${url} -> ${dest_path}"
  curl -fsSL "${url}" -o "${dest_path}"
done

echo "Done. Files written to ${DEST_DIR}"
