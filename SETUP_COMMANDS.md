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

## 8. 已知非阻塞警告

- `missing pytorch3d`（`envs/camera/camera.py`）：不需要 3D 数据时可忽略。
- `OIDN Error: unsupported device type: CUDA`：降噪器在 CUDA 上不可用，渲染正常，每帧刷错。
