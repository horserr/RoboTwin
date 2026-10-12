#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The workspace is bind-mounted from the WSL host, so its owner differs from
# the container user (root). Mark it (and submodules) as safe for git.
git config --global --add safe.directory "$ROOT"
git config --global --add safe.directory '*'

# Idempotent: safe to re-run on every container creation / rebuild.
echo "[initialize] Updating git submodules..."
git submodule update --init --recursive

# update_xpolicylab.sh already guards against clobbering an existing checkout.
echo "[initialize] Updating XPolicyLab..."
# bash "$ROOT/scripts/update_xpolicylab.sh"

# `git clone` fails if the target already exists, so guard it.
CUROBO_DIR="$ROOT/env/curobo"
if [[ -d "$CUROBO_DIR/.git" ]]; then
    echo "[initialize] curobo already checked out at $CUROBO_DIR, skipping clone."
else
    echo "[initialize] Cloning curobo v0.7.8..."
    git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
