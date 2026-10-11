set -o euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git submodule update --init --recursive

bash $ROOT/scripts/update_xpolicylab.sh

git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git $ROOT/env/curobo
