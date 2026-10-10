#!/usr/bin/env bash
# Build the .difypkg from a clean copy of the repo (tracked + untracked-but-not-ignored files),
# so local junk such as .env, venvs or old scripts can never end up in the package.
#
# Usage: DIFY_PLUGIN_CLI=/path/to/dify-plugin scripts/package.sh [out_dir]
# CLI: https://github.com/langgenius/dify-plugin-daemon/releases (verify the sha256 digest).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${DIFY_PLUGIN_CLI:-dify-plugin}"
OUT="${1:-$ROOT/dist}"
VERSION="$(sed -n 's/^version: //p' "$ROOT/manifest.yaml" | head -1)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT"
git ls-files -co --exclude-standard -z \
  | while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done \
  | tar --null -T - -cf - | tar -xf - -C "$STAGE"
mkdir -p "$OUT"
"$CLI" plugin package "$STAGE" -o "$OUT/zenmux-dify-plugin_${VERSION}.difypkg"

echo "package contents (non-YAML):"
unzip -Z1 "$OUT/zenmux-dify-plugin_${VERSION}.difypkg" | grep -vE '^models/.*/[^_][^/]*\.yaml$'
