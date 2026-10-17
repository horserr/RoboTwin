"""Download and extract RoboTwin assets through the Hugging Face cache.

Files are downloaded once into the shared Hugging Face cache (in the dev
container this is the ``huggingface-cache-volume`` mounted at
``/root/.cache/huggingface/hub``) and extracted directly from there. Because
the archives never live inside ``assets/``, re-running this script after a
container rebuild or a fresh checkout reuses the cached blobs instead of
downloading ~14.5 GB again.

Set ``HF_HUB_OFFLINE=1`` to extract from an existing cache without any network
access, or ``HF_REVISION`` to override the pinned commit.
"""

import os
from pathlib import Path
from zipfile import ZipFile

# 强制开启 huggingface_hub 的进度条（即使输出被重定向到非 TTY）
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import enable_progress_bars

REPO_ID = "TianxingChen/RoboTwin2.0"
# 固定 commit，使缓存命中时无需访问网络
REVISION = os.environ.get("HF_REVISION", "3dc3b798668feb99ac61cc9086d84cbcc3d79186")
FILES = ["background_texture.zip", "embodiments.zip", "objects.zip"]

enable_progress_bars()

for idx, filename in enumerate(FILES, 1):
    print(f"[{idx}/{len(FILES)}] Resolving {filename} from HF cache ...", flush=True)
    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            revision=REVISION,
        )
    )
    print(f"[{idx}/{len(FILES)}] Extracting {archive} ...", flush=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(".")
    print(f"[{idx}/{len(FILES)}] Finished {filename}", flush=True)
