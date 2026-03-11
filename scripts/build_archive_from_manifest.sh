#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_archive_from_manifest.sh [--check-only] <manifest.lst> <output.tgz>

The manifest may contain blank lines and lines starting with '#'.
Paths are interpreted relative to the repository root.
EOF
}

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

MANIFEST_INPUT=$1
OUTPUT_INPUT=$2

if [[ "${MANIFEST_INPUT}" = /* ]]; then
  MANIFEST_PATH="${MANIFEST_INPUT}"
else
  MANIFEST_PATH="${REPO_ROOT}/${MANIFEST_INPUT}"
fi

if [[ "${OUTPUT_INPUT}" = /* ]]; then
  OUTPUT_PATH="${OUTPUT_INPUT}"
else
  OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_INPUT}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

TMP_MANIFEST=$(mktemp)
trap 'rm -f "${TMP_MANIFEST}"' EXIT

MISSING_COUNT=0
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  line=${raw_line%$'\r'}
  if [[ -z "${line}" ]]; then
    continue
  fi
  case "${line}" in
    \#*) continue ;;
  esac

  if [[ ! -e "${REPO_ROOT}/${line}" ]]; then
    echo "Missing path: ${line}" >&2
    MISSING_COUNT=$((MISSING_COUNT + 1))
    continue
  fi

  printf '%s\n' "${line}" >> "${TMP_MANIFEST}"
done < "${MANIFEST_PATH}"

if [[ ${MISSING_COUNT} -ne 0 ]]; then
  echo "Manifest validation failed with ${MISSING_COUNT} missing path(s)." >&2
  exit 1
fi

ENTRY_COUNT=$(wc -l < "${TMP_MANIFEST}")
echo "Validated ${ENTRY_COUNT} manifest entries from ${MANIFEST_PATH}"

if [[ ${CHECK_ONLY} -eq 1 ]]; then
  exit 0
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
tar -C "${REPO_ROOT}" -czf "${OUTPUT_PATH}" -T "${TMP_MANIFEST}"
sha256sum "${OUTPUT_PATH}"
