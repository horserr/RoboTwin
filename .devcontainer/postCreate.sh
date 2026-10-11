set -o euo pipefail

sh -c \"$(curl -fsLS https://get.chezmoi.io)\" -- -b /usr/bin

chezmoi init --source=~/.config/.dotfiles --apply $GITHUB_USERNAME
