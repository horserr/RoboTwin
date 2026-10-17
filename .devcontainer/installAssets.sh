#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "[installAssets] Downloading RoboTwin assets (~11 GB) and extracting them ..."
bash ./scripts/_download_assets.sh
echo "[installAssets] All assets are ready."
