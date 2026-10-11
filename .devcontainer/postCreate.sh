#!/usr/bin/env bash
set -euo pipefail

sh -c "$(curl -fsLS https://get.chezmoi.io)" -- -b /bin

chezmoi init --source=~/.config/.dotfiles --apply $GITHUB_USERNAME
