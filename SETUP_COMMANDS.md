# 环境搭建与测试命令记录

> 本文件记录本次会话中用于搭建 π0.5 评测环境、下载 checkpoint、以及冒烟/批量测试所执行的命令，便于重建容器后复现或查阅。
> 更新日期：2026-10-17

## 0. 关键文件改动（非命令，但属于环境搭建的一部分）

| 文件                                                                 | 改动                                                                                             |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `.devcontainer/Dockerfile`                                           | 基础镜像 `12.1.1-cudnn8-devel` → `12.8.1-cudnn-devel`；apt 增加 `tzdata`；`ENV TZ=Asia/Shanghai` |
| `.devcontainer/onCreate.sh`                                          | bootstrap torch `2.4.1`(cu121) → `2.7.1`(cu128)                                                  |
| `pyproject.toml`                                                     | torch→`2.7.1`、torchvision→`0.22.1`、index cu121→cu128；依赖增加 `setuptools<82.0.0`             |
| `XPolicyLab/policy/Pi_05/install.sh`                                 | 开头 `unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT`                                                  |
| `XPolicyLab/policy/Pi_05/deploy.yml`                                 | `checkpoint_num`→`29999`；`repo_id`→`RoboTwin_lerobot_v30`                                       |
| `XPolicyLab/policy/Pi_05/setup_eval_policy_server.sh`                | `command -v conda` 守卫，YAML_PYTHON 回退 ambient python                                         |
| `XPolicyLab/utils/run_sim_env_client.sh` / `run_debug_env_client.sh` | conda 守卫；无 conda 时强制 `/opt/venv`                                                          |
| `.gitmodules`                                                        | XPolicyLab URL → `https://github.com/horserr/XPolicyLab.git`                                     |
| `env_cfg/eval/smoke_tasks.yml`、`batch_tasks.yml`                    | 新建；`all_tasks.yml` 的 `gpu_ids` 0-7→0-1                                                       |

## 1. Phase 0 — 硬件/环境自检与升级

```bash
# 诊断
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); torch.zeros(1,device='cuda')"

# 升级 torch（改完 pyproject.toml 后）
cd /root/workspace/RoboTwin
uv lock
uv sync

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.zeros(1, device='cuda'))"
python -c "import sapien, warp, pytorch3d, curobo; print('ALL IMPORTS OK')"

# 修复 sapien 的 pkg_resources（setuptools>=82 移除了它）
uv add "setuptools<82"
uv lock
uv sync
python -c "import sapien, warp, pytorch3d, curobo; print('ALL IMPORTS OK')"

# 渲染验证
python scripts/test_render.py
```

## 2. Phase 1 — π0.5 策略环境（OpenPI）

```bash
cd /root/workspace/RoboTwin/XPolicyLab/policy/Pi_05
bash install.sh

source openpi/.venv/bin/activate
python -c "import openpi, XPolicyLab; print('openpi + XPolicyLab ok')"
python -c "import jax, jax.numpy as jnp; print('jax', jax.__version__); g=jax.devices('gpu'); print('gpus:', len(g)); a=jax.device_put(jnp.ones((1024,1024)), g[0]); print('matmul sum:', float((a @ a).block_until_ready().sum()))"
```

## 3. Phase 2 — checkpoint 下载与解压

```bash
cd /root/workspace/RoboTwin/XPolicyLab/policy/Pi_05
mkdir -p checkpoints
hf download TianxingChen/RoboTwin2.0 cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz \
  --repo-type dataset \
  --local-dir checkpoints

tar -xzf checkpoints/cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz -C checkpoints/
ls checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999
# 期望：_CHECKPOINT_METADATA  assets  params
```

## 4. Phase 3 — 单任务冒烟

```bash
cd /root/workspace/RoboTwin
# dry-run 验证调度
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/smoke_tasks.yml \
  --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex \
  --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint \
  --dry-run

# 正式跑 5 episode
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/smoke_tasks.yml \
  --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex \
  --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint \
  --test-num 5
```

## 5. Phase 4 — 小批量 / 全量

```bash
cd /root/workspace/RoboTwin
# 小批量 5 任务（20 ep/任务）
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/batch_tasks.yml \
  --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex \
  --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint \
  --test-num 20

# 全量 50 任务（100 ep/任务，后台跑）
nohup bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex \
  --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint \
  > eval_full.log 2>&1 &
```

## 6. Git 操作（fork XPolicyLab + submodule 重定向）

```bash
# 在 XPolicyLab 内：加 fork 远程并提交
cd /root/workspace/RoboTwin/XPolicyLab
git remote add mine git@github.com:horserr/XPolicyLab.git
git add policy/Pi_05/install.sh policy/Pi_05/openpi/uv.lock
git commit -m "chore: unset VIRTUAL_ENV in Pi_05 install.sh; refresh openpi uv.lock"
git push mine main

# 若 push 报 non-fast-forward（fork 落后/分叉），对齐后 cherry-pick
git fetch mine
git checkout -B main mine/main
git cherry-pick <你的commit>
git push mine main

# 根仓库：submodule URL 指向 fork 并提交
cd /root/workspace/RoboTwin
git submodule sync
git -C XPolicyLab remote set-url origin https://github.com/horserr/XPolicyLab.git
git -C XPolicyLab remote remove mine
git -C XPolicyLab remote add upstream https://github.com/XPolicyLab/XPolicyLab.git
git -C XPolicyLab fetch origin
git -C XPolicyLab branch --set-upstream-to=origin/main main
git add .gitmodules XPolicyLab
git commit -m "point XPolicyLab submodule to my fork"
git push origin main
```

## 7. 容器时区修复

```bash
apt-get update && apt-get install -y tzdata
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
date
# Dockerfile 已加 tzdata + ENV TZ=Asia/Shanghai，重建后永久生效
```

## 8. 排障：所有 job 秒退 failed（ModuleNotFoundError: No module named 'jax'）

现象：`eval_policy.sh multitask` 每个 job 约 2s 内 return_code=1，日志里出现

```
[SERVER] Activating uv environment: .../Pi_05/openpi/.venv
[SERVER] Using python: /opt/venv/bin/python
ModuleNotFoundError: No module named 'jax'
```

原因：`openpi/.venv/bin/python` 指向 uv 管理的解释器
`/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11`，
该目录在容器 home 下、**重建容器后丢失**（venv 在 `/root/workspace` 卷上仍在），
软链悬挂 → `command -v python` 静默回退到 `/opt/venv/bin/python`（无 jax）。

修复（无需重建 venv，site-packages 完好）：

```bash
uv python install 3.11
/root/workspace/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
```

自检：`openpi/.venv/bin/python -V` 必须能输出版本（如 `Python 3.11.17`）。
`Pi_05/setup_eval_policy_server.sh` 已改为显式引用该解释器并做存在性校验。

## 9. 排障：GPU 1 上 job 大面积失败 / 假成功（vulkan DeviceLost）

现象（run `2026-10-18_03-17-38_653172`，50 任务全量）：

- 只有 `adjust_bottle`(99%)、`beat_block_hammer`(90%) 真正跑完，其余 48 个 job 12.3s 就结束且被记为 `success`；
- 这些假成功日志只有 `Render Error` + `[MAIN] eval finished`；
- 后续复跑又变成 `RuntimeError: cannot create buffer`（`envs/camera/camera.py` 取样时）
  和 `vk::Device::waitForFences: ErrorDeviceLost`。

根因：**GPU 1 显存被容器外的进程占用约 19GB**（`nvidia-smi` 无 process 但显存不释放，
`/proc/*/fd` 也找不到持有者），单 job 峰值需 ~12GB，总量撞到 32.6GB 上限后
sapien 的光追渲染器直接 device lost。

实测（同一批任务）：

| GPU | 空闲显存 | 峰值总占用 | 结果                   |
| --- | -------- | ---------- | ---------------------- |
| 0   | ~27GB    | ~17GB      | 3/3 成功               |
| 1   | ~13GB    | ~31.3GB    | 3/3 失败（DeviceLost） |

处理：

- `env_cfg/eval/{all,batch,smoke,remaining}_tasks.yml` 全部固定 `gpu_ids: "0"`；
- `XPolicyLab/policy/Pi_05/{eval.sh,setup_eval_policy_server.sh}` 增加
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免 jax 预留 30% 显存把渲染色吃满；
- `scripts/test_render.py` 改为 `traceback.print_exc()` + `sys.exit(1)`，
  渲染自检失败不再被记成 `success`；
- 等 GPU 1 空闲（`nvidia-smi` 显示 ~0MiB）后再改回 `gpu_ids: "0-1"` 可翻倍吞吐。

## 10. 已知非阻塞警告

- `missing pytorch3d`（`envs/camera/camera.py`）：不需要 3D 数据时可忽略。
- `OIDN Error: unsupported device type: CUDA`：降噪器在 CUDA 上不可用，渲染正常，每帧刷错。

## 11. 排障：job 全部 failed（RuntimeError: failed to find a rendering device）

现象（run `2026-10-18_18-30-33_060222`，5 任务批量、每 job 约 12s 失败）：

```
[CLIENT] Connecting to server localhost:57603...
Render Error
  File ".../scripts/test_render.py", line 48, in setup_scene
    self.renderer = sapien.SapienRenderer()
RuntimeError: failed to find a rendering device
[MAIN] kill server 594031
```

**不是第 8 节的 jax 问题**：日志里 `[SERVER] Using python: .../openpi/.venv/bin/python`
和 `[CONNECTED] Policy server ready` 都正常，说明策略服务端（jax）起来了，
挂的是 **sim 客户端的 Vulkan 渲染**。

根因有两层：

1. `sapien/_vulkan_tricks.py` 只检查 `/usr/share/vulkan/icd.d/nvidia_icd.json`；
   重建容器后 nvidia-container-toolkit 只把 ICD 挂到了 `/etc/vulkan/icd.d/`，
   于是 sapien 打印 `Failed to find Vulkan ICD file ...` 并**强制**
   `VK_ICD_FILENAMES=<sapien 自带 ICD>`（api 仅 1.2.140，且把 Mesa llvmpipe
   回退 ICD 一起屏蔽掉）。此后渲染器初始化没有任何容错余量，
   GPU 侧稍有抖动就直接 `failed to find a rendering device`。

2. 当时的 GPU 0 上还有容器外进程残留显存（`nvidia-smi` 显示 5GB 占用但
   `No running processes found`），与第 9 节同源。

自检 / 修复：

```bash
# 检查 ICD 是否就位（sapien 只看这个路径）
ls -l /usr/share/vulkan/icd.d/nvidia_icd.json
# 检查 vulkan 能否枚举到 GPU（应能看到两块 RTX 5090）
vulkaninfo --summary | grep -E 'deviceName|driverName'
# 直接跑渲染自检，必须出现绿色 Render Well
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python scripts/test_render.py

# 补软链（Dockerfile 已加入，重建后自动生效）
mkdir -p /usr/share/vulkan/icd.d
ln -sfn /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
```

修复后 `import sapien` 不再打印 ICD 警告，`VK_ICD_FILENAMES` 保持 unset，
vulkan loader 会同时尝试 NVIDIA 与 Mesa，不会再因为单点失败而全批 failed。

遇到这类「登录后第一时间全批 failed」时，先 `bash scripts/test_render.py`
确认渲染链路，再怀疑策略侧。

## 12. rebuild 容器后的固定动作（checklist）

### 12.1 什么会留下、什么会丢

| 路径                                                                                    | 类型         | 重建后                                        |
| --------------------------------------------------------------------------------------- | ------------ | --------------------------------------------- |
| `/root/workspace/RoboTwin`（含 `assets/` 16G、`Pi_05/checkpoints` 24G、`eval_result/`） | bind mount   | 保留                                          |
| `/opt/venv`（RoboTwin 主环境，py3.10）                                                  | volume       | 保留，`home=/usr/bin` 安全                    |
| `/root/.cache/uv`、`/root/.cache/huggingface/hub`                                       | volume       | 保留（不用重下 wheel/权重）                   |
| `Pi_05/openpi/.venv`                                                                    | 在 workspace | **文件在，但解释器软链悬空**                  |
| `/root/.local/share/uv/python`                                                          | 容器内       | **丢失** → 需 `uv python install 3.11`        |
| `/root/.local/bin`（`hf` 等 uv tool）                                                   | 容器内       | 丢失，靠 Dockerfile 的 `uv tool install` 重建 |
| 手动 `apt install` / 手改系统文件                                                       | 容器内       | 丢失，必须写进 Dockerfile                     |

> 结论：凡是「手动在容器里装/改」的东西，一律写回 `.devcontainer/Dockerfile`，
> 否则下次 rebuild 必然复发（tzdata、Vulkan ICD 软链都属此类）。

### 12.2 每次 rebuild 后按顺序过一遍

```bash
cd /root/workspace/RoboTwin

# 1) GPU 是否干净：used 应接近 0；有 GB 级占用但 "No running processes found"
#    = 容器外/旧容器残留，等它释放，否则会 DeviceLost / 假成功（第 9 节）
nvidia-smi

# 2) 主环境（/opt/venv）
/opt/venv/bin/python -c "import torch,sapien,warp,pytorch3d,curobo; print(torch.__version__, torch.cuda.is_available())"

# 3) 渲染链路（第 11 节，必须绿色 Render Well）
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python scripts/test_render.py

# 4) 依赖锁一致性（onCreate 会跑 uv sync --frozen，锁不同步会直接失败）
uv lock --check

# 5) π0.5 的解释器（第 8 节，最容易漏）
uv python install 3.11
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -V      # 必须输出 3.11.x

# 6) 先冒烟，再上批量（别直接全量）
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/smoke_tasks.yml \
  --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex \
  --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint \
  --test-num 5
```

### 12.3 其它容易踩的点

- **`GH_TOKEN` / `HF_TOKEN` / `GITHUB_USERNAME`**：容器创建时从宿主 env 注入。
  缺 `GH_TOKEN` 会导致 `postCreate` 的 chezmoi 恢复 dotfiles（含 ssh key）失败；
  缺 `HF_TOKEN` 会导致 `hf download` 拉 private/大文件失败。
- **assets 不会重下**：`assets/.extract_complete` 在 workspace 里，`installAssets.sh`
  见到标记就跳过那 11G 下载；需要重解压时删标记再重建。
- **根目录 `./.venv` 别乱删**：它属于 WSL 宿主（owner 1005，指向 `/home/lyz/...`），
  在容器里是悬空链接。容器内一律用 `/opt/venv`（由 `UV_PROJECT_ENVIRONMENT` 与
  `run_sim_env_client.sh` 强制），不要 `source .venv/bin/activate`。
- **网络模式**：`runArgs` 用 `--network=host`，ssh 走 2222，端口冲突时先看宿主占用。
- **镜像本身变了吗**：只重启容器 ≠ 重建镜像。改了 `Dockerfile` 必须
  Dev Containers: Rebuild Container（不带 cache）才会生效，否则你以为修好了其实没生效。
