# ==========================================
# 阶段 1：基础环境
# ==========================================
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  git \
  ffmpeg \
  libvulkan1 \
  mesa-vulkan-drivers \
  vulkan-tools \
  libegl1 \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 && \
  rm -rf /var/lib/apt/lists/*

# uv 缓存配置
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy

# 项目虚拟环境
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ==========================================
# 阶段 2：获取 RoboTwin 源码
# ==========================================
FROM base AS target

# ARG ROBOTWIN_REPO=https://github.com/RoboTwin-Platform/RoboTwin.git

# RUN git clone --recurse-submodules "${ROBOTWIN_REPO}" . && \
#   git submodule update --init --recursive

# # ==========================================
# # 阶段 3：创建 Python 虚拟环境
# # ==========================================
# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv venv \
#   --python 3.10 \
#   "${VIRTUAL_ENV}" && \
#   python --version && \
#   uv --version

# # ==========================================
# # 阶段 4：安装 CUDA 12.1 版本的 PyTorch
# # ==========================================
# # PyTorch 2.4.1 对应 torchvision 0.19.1。
# # 这里单独使用 cu121 index，避免普通依赖解析到错误的 PyTorch wheel。
# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   --index-url https://download.pytorch.org/whl/cu121 \
#   torch==2.4.1 \
#   torchvision==0.19.1

# # ==========================================
# # 阶段 5：安装普通 requirements
# # ==========================================
# # requirements.txt 中的 torch/torchvision 已经单独安装，
# # 因此过滤掉这两项，避免再次从默认 PyPI index 解析。
# RUN set -euxo pipefail; \
#   sed \
#   -E \
#   -e '/^[[:space:]]*torch([<=>!~]|[[:space:]]|$)/d' \
#   -e '/^[[:space:]]*torchvision([<=>!~]|[[:space:]]|$)/d' \
#   scripts/requirements.txt \
#   > /tmp/requirements-without-torch.txt

# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   --requirements /tmp/requirements-without-torch.txt

# # ==========================================
# # 阶段 6：安装 PyTorch3D
# # ==========================================
# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   --no-build-isolation \
#   "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# # ==========================================
# # 阶段 7：准备并安装 XPolicyLab
# # ==========================================
# RUN bash scripts/update_xpolicylab.sh

# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   --editable ./XPolicyLab

# # ==========================================
# # 阶段 8：修改 sapien/wrapper/urdf_loader.py
# # ==========================================
# RUN SAPIEN_LOCATION="$$(python -m pip show sapien | \
#   awk '/^Location:/{print $$2}')/sapien" && \
#   test -f "$${SAPIEN_LOCATION}/wrapper/urdf_loader.py" && \
#   sed -i -E \
#   's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
#   "$${SAPIEN_LOCATION}/wrapper/urdf_loader.py"

# # ==========================================
# # 阶段 9：修改 mplib/planner.py
# # ==========================================
# RUN MPLIB_LOCATION="$$(python -m pip show mplib | \
#   awk '/^Location:/{print $$2}')/mplib" && \
#   test -f "$${MPLIB_LOCATION}/planner.py" && \
#   sed -i -E \
#   's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
#   "$${MPLIB_LOCATION}/planner.py"

# # ==========================================
# # 阶段 10：安装 Curobo
# # ==========================================
# RUN mkdir -p envs && \
#   git clone \
#   --branch v0.7.8 \
#   --depth 1 \
#   https://github.com/NVlabs/curobo.git \
#   envs/curobo

# # 先安装 Curobo 的构建依赖。
# # --no-build-isolation 要求构建依赖已经存在于当前环境。
# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   setuptools==69.5.1 \
#   wheel \
#   ninja

# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   --no-build-isolation \
#   --editable ./envs/curobo

# RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
#   uv pip install \
#   --python "${VIRTUAL_ENV}/bin/python" \
#   warp-lang==1.12.0

# # ==========================================
# # 阶段 11：验证安装
# # ==========================================
# RUN python - <<'PY'
# import torch
# import torchvision

# print("Python:", __import__("sys").version)
# print("Torch:", torch.__version__)
# print("Torch CUDA:", torch.version.cuda)
# print("Torchvision:", torchvision.__version__)
# print("CUDA available during build:", torch.cuda.is_available())
# PY

# WORKDIR /workspace/RoboTwin

# # Dev Container 中不建议设置 ENTRYPOINT ["pixi", "run"]。
# # 直接使用 PATH 中的 /opt/venv/bin/python。
# CMD ["python", "--version"]
