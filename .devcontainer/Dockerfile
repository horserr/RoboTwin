FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  wget \
  unzip \
  ffmpeg \
  libvulkan1 \
  mesa-vulkan-drivers \
  vulkan-tools \
  tzdata \
  openssh-server \
  # needed to build C/CUDA extensions (pytorch3d, curobo)
  python3-dev \
  ninja-build \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# 创建 sshd 运行目录及 root 的 .ssh 目录，并赋予正确的权限（700）
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh

# 宿主机为 CST (Asia/Shanghai, UTC+8)，设置时区避免容器 date 显示 UTC
ENV TZ=Asia/Shanghai

# uv 缓存配置
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy

# 项目虚拟环境
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

FROM base AS target

ENV HF_HUB_CACHE=/root/.cache/huggingface/hub

RUN --mount=type=cache,target=/root/.cache/uv \
  uv tool install huggingface_hub

RUN echo 'root:root' | chpasswd

# sshd config
# due to host network, 22 port cannot bind again
RUN echo "PermitRootLogin yes" > /etc/ssh/sshd_config.d/root_login.conf && \
  echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config.d/root_login.conf && \
  echo "Port 2222" >> /etc/ssh/sshd_config.d/root_login.conf

EXPOSE 2222
