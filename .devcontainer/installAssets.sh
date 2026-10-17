#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

MARKER="$ROOT/assets/.extract_complete"
if [ -f "$MARKER" ]; then
  echo "[installAssets] Assets already extracted ($MARKER exists); skipping."
  exit 0
fi

echo "[installAssets] Downloading RoboTwin assets (~11 GB) and extracting them ..."
bash ./scripts/_download_assets.sh
touch "$MARKER"
echo "[installAssets] All assets are ready."
