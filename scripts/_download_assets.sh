set -euo pipefail

cd assets
# Archives stay in the Hugging Face cache and are extracted from there, so a
# re-run (or a container rebuild) never downloads them again.
python -u _download.py

cd ..
echo "Configuring Path ..."
python ./scripts/update_embodiment_config_path.py
