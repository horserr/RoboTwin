# Plan: 用官方 π0.5 checkpoint 在 RoboTwin 上跑评测

## TL;DR

“跑通 pi0.5” = 在 RoboTwin 上使用官方联合训练好的 π0.5 checkpoint 跑**推理评测**，看到成功率与视频。**不需要训练、不需要下载数据集**（数据集仅用于微调）。核心路径：装 OpenPI(uv) 策略环境 → 下载并解压官方 checkpoint → 单任务冒烟评测 → 扩展多任务。

## 关键结论（已确认）

- 用户目标：官方 checkpoint 评测（非全流程微调）。
- 硬件：2× RTX 5090（Blackwell sm_120），驱动 580.178.04、CUDA runtime 13.0 —— ✅ 满足。
- ❌ torch 2.4.1+cu121 实测与 sm_120 不兼容：`no kernel image is available for execution on the device`。
- ❌ 容器 nvcc = 12.1（基础镜像 `nvidia/cuda:12.1.1-...`）：CUDA 12.1 不支持 sm_120（sm_120 需 CUDA ≥12.8），因此**只升级 torch 不够，必须同时升级 CUDA toolkit**。
- 环境：**无 conda**。uv 管理，虚拟环境在 Docker volume `/opt/venv`（`UV_PROJECT_ENVIRONMENT=/opt/venv`，`uv.lock` 冻结依赖）。
- 评测框架（`XPolicyLab/utils/run_*_env_client.sh`）硬依赖 conda —— Phase 3 需装 conda 或打补丁跳过（见 Phase 3 注记）。
- 官方 checkpoint：`cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`（12.4GB，HF 仓库 `TianxingChen/RoboTwin2.0`）。
- checkpoint 命名约定 = `{bench}-{ckpt}-{env_cfg_type}-{action_type}-{seed}` → bench=RoboTwin, ckpt=lerobot_v30, env_cfg_type=aloha_agilex, action=joint, seed=0。

## Steps

### Phase 0 — 硬件/环境自检（阻塞一切）

1. ✅ `nvidia-smi`：2×5090，驱动 580（≥570），CUDA runtime 13.0。
2. ❌ torch CUDA 冒烟：`torch 2.4.1+cu121` 报 `no kernel image is available`（不支持 sm_120）——已复现，确认需升级。
3. ⬜ 升级 CUDA toolkit 到 ≥12.8（让 nvcc 能编译 sm_120），两选一：
   - A（推荐，可复现）：改 `.devcontainer/Dockerfile` 基础镜像 `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` → 12.8+ devel（如 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`），重建容器。
   - B（不重建，临时）：在当前容器安装 nvcc 12.8（pip 包 `nvidia-cuda-nvcc-cu12`，或下载 CUDA 12.8 toolkit）。
4. ⬜ 升级 torch：`pyproject.toml` 中 `torch==2.4.1`→`torch==2.7.1+cu128`、`torchvision`→`==0.22.1+cu128`、index `pytorch-cu121`→`pytorch-cu128`；同步改 `.devcontainer/onCreate.sh` 里 bootstrap 的 torch。
5. ⬜ 重新生成锁并安装：`uv lock`（不能用 `--frozen`）→ `uv sync`，重编 `pytorch3d` / `nvidia-curobo`（慢、可能失败，见 Further Considerations）。
6. ⬜ 验证：`python -c "import torch; torch.zeros(1,device='cuda')"` 通过；`python -c "import sapien, warp, pytorch3d, curobo"` 全部成功。
7. ⬜ 仿真渲染验证：`python scripts/test_render.py`。

### Phase 1 — 安装 π0.5 策略环境（OpenPI/uv）

4. 确认 `uv` 存在；`cd XPolicyLab/policy/Pi_05 && bash install.sh`（uv sync 安装 openpi + XPolicyLab，生成 `openpi/.venv`）。
5. 若 jax 0.5.3 在 5090 报错 → 升级 jax 到支持 sm_120 的版本（≥0.5.4/cuda12.8）。

### Phase 2 — 下载并放置官方 checkpoint

6. 下载：`hf download TianxingChen/RoboTwin2.0 cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`。
7. 解压到 `XPolicyLab/policy/Pi_05/checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/`，确认含 `params/` 与 `assets/`。
8. 检查 `assets/` 子目录名是否与 `deploy.yml` 的 `repo_id: arx_x5_sim` 一致；不一致则改 `repo_id`。`train_config_name` 保持 `pi05_base_aloha_full_sim_arx-x5_seed_0`（adapter 内唯一 pi05 配置）。

### Phase 3 — 单任务冒烟评测

> ⚠️ 无 conda 问题：`XPolicyLab/utils/run_sim_env_client.sh` 会执行 `conda info --base` + `conda activate ${eval_env_conda_env}`。本容器无 conda，需先二选一：
>
> - 装 miniconda 并建 `RoboTwin` env（与 `/opt/venv` 重复，不推荐）；或
> - 给 `run_sim_env_client.sh` / `run_debug_env_client.sh` / `setup_eval_policy_server.sh` 加 `command -v conda` 守卫，无 conda 时直接用 `/opt/venv`（推荐）。

9. 裁剪 `env_cfg/eval/all_tasks.yml` 的 `tasks` 为单任务（如 `beat_block_hammer`）。
10. `--dry-run` 验证调度。
11. 运行：
    bash scripts/eval_policy.sh multitask \
     --config env_cfg/eval/all_tasks.yml \
     --policy-name Pi_05 \
     --ckpt-name lerobot_v30 \
     --env-cfg-type aloha_agilex \
     --policy-conda-env uv \
     --eval-env-conda-env RoboTwin \
     --action-type joint
    （bench-name 默认 RoboTwin；seed 默认 0；task-config 默认 demo_clean）
    注意：`--policy-conda-env uv` 是 Pi_05 的 uv 约定（其 README：arg9 传 `uv`，非 conda 名）；
    `--eval-env-conda-env` 无 conda 时仅作标识，实际用 `/opt/venv`（依赖上面的补丁）。
    确认 `eval_result/multitask/` 出视频与成功率。

### Phase 4 — 全量评测

12. 恢复 `all_tasks.yml` 为全部 50 任务，重跑；结果在 `eval_result/multitask/`。

## Relevant files

- `.devcontainer/Dockerfile` / `.devcontainer/devcontainer.json` / `.devcontainer/onCreate.sh` — CUDA 12.1 基础镜像 + uv volume venv 初始化（升级 torch/nvcc 的改动点）
- `XPolicyLab/policy/Pi_05/` — π0.5 adapter（`install.sh`/`deploy.yml`/`model.py`/`eval.sh`/`setup_eval_policy_server.sh`）
- `XPolicyLab/utils/run_sim_env_client.sh` / `setup_env_client.sh` — eval 客户端（硬依赖 conda，无 conda 需打补丁）
- `XPolicyLab/policy/Pi_05/deploy.yml` — `repo_id: arx_x5_sim`、`train_config_name: pi05_base_aloha_full_sim_arx-x5_seed_0`、`checkpoint_num: 59999`
- `XPolicyLab/utils/checkpoint_resolver.py` — checkpoint 目录解析（`{bench}-{ckpt}-{env}-{action}-{seed}` + `checkpoints/<ckpt_name>` fallback）
- `scripts/eval_policy.sh` / `scripts/eval_policy_multitask.py` — RoboTwin 评测入口
- `env_cfg/eval/all_tasks.yml` — 任务列表与 GPU 配置
- `env_cfg/aloha_agilex.yml` + `env_cfg/robot/_robot_info.json` — `aloha_agilex` 机器人动作维度 [6,6]+[1,1]

## Verification

1. nvidia-smi ✅；升级后 torch CUDA 冒烟通过（否则继续排查 nvcc/torch 版本）。
2. `import sapien, warp, pytorch3d, curobo` 全部成功（源码扩展重编成功）。
3. `scripts/test_render.py` 能渲染（仿真可用）。
4. `bash install.sh` 后 `openpi/.venv` 存在，`uv run python -c "import openpi, XPolicyLab"` 成功。
5. checkpoint 解压后 `params/`、`assets/<repo_id>/` 存在且 `repo_id` 与 deploy.yml 一致。
6. `--dry-run` 无调度错误；单任务评测出视频且成功率 > 0（参考 leaderboard π0.5 约 80%+）。

## Further Considerations

1. 5090 兼容性（已确认不兼容，见 Phase 0）：升级 torch 2.7.1+cu128 + nvcc 12.8 后需重编 curobo/pytorch3d；warp-lang 1.12.0 是否支持 sm_120 待验证（若失败尝试更新 warp-lang）。
2. 源码编译风险：pytorch3d/curobo 用 nvcc 12.8 编译可能仍有 ABI/头文件问题；若失败，尝试预编译 wheel 或升级到 torch 2.8+。
3. OpenPI 的 jax[cuda12]==0.5.3 可能不支持 sm_120，Phase 1 需验证（可能需升级 jax）。
4. 无 conda：评测框架硬依赖 conda，需装 conda 或打补丁（Phase 3）。
5. 若导师后续要"全流程"，再补：下载数据集 → transform_lerobot_v30 → finetune → eval（LoRA 需 >46GB，2×5090=64GB 勉强可 LoRA）。
