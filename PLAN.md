# Plan: 用官方 π0.5 checkpoint 在 RoboTwin 上跑评测

## TL;DR

“跑通 pi0.5” = 在 RoboTwin 上使用官方联合训练好的 π0.5 checkpoint 跑**推理评测**，看到成功率与视频。**不需要训练、不需要下载数据集**（数据集仅用于微调）。核心路径：装 OpenPI(uv) 策略环境 → 下载并解压官方 checkpoint → 单任务冒烟评测 → 扩展多任务。

## 关键结论（已确认）

- 用户目标：官方 checkpoint 评测（非全流程微调）。
- 硬件：2× RTX 5090（Blackwell sm_120）。⚠️ 与 RoboTwin 固定的 torch 2.4.1(cu121) 及 OpenPI 的 jax[cuda12]0.5.3 可能不兼容，必须先验证。
- 环境进度：RoboTwin 环境已装，但仿真未跑通过。
- 官方 checkpoint：`cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`（12.4GB，HF 仓库 `TianxingChen/RoboTwin2.0`）。
- checkpoint 命名约定 = `{bench}-{ckpt}-{env_cfg_type}-{action_type}-{seed}` → bench=RoboTwin, ckpt=lerobot_v30, env_cfg_type=aloha_agilex, action=joint, seed=0。

## Steps

### Phase 0 — 硬件/环境自检（阻塞一切）

1. `nvidia-smi` 确认 2×5090 与驱动版本（Blackwell 需 CUDA ≥12.8，驱动 ≥570）。
2. RoboTwin conda 环境冒烟：`python -c "import torch; print(torch.__version__, torch.cuda.is_available()); torch.zeros(1,device='cuda')"`。若 torch 2.4.1 在 5090 失败 → 升级 torch ≥2.7(cu128)，并重装依赖（sapien/warp-lang/pytorch3d/curobo）。
3. 仿真渲染验证：`python scripts/test_render.py`。

### Phase 1 — 安装 π0.5 策略环境（OpenPI/uv）

4. 确认 `uv` 存在；`cd XPolicyLab/policy/Pi_05 && bash install.sh`（uv sync 安装 openpi + XPolicyLab，生成 `openpi/.venv`）。
5. 若 jax 0.5.3 在 5090 报错 → 升级 jax 到支持 sm_120 的版本（≥0.5.4/cuda12.8）。

### Phase 2 — 下载并放置官方 checkpoint

6. 下载：`hf download TianxingChen/RoboTwin2.0 cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`。
7. 解压到 `XPolicyLab/policy/Pi_05/checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/`，确认含 `params/` 与 `assets/`。
8. 检查 `assets/` 子目录名是否与 `deploy.yml` 的 `repo_id: arx_x5_sim` 一致；不一致则改 `repo_id`。`train_config_name` 保持 `pi05_base_aloha_full_sim_arx-x5_seed_0`（adapter 内唯一 pi05 配置）。

### Phase 3 — 单任务冒烟评测

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
    确认 `eval_result/multitask/` 出视频与成功率。

### Phase 4 — 全量评测

12. 恢复 `all_tasks.yml` 为全部 50 任务，重跑；结果在 `eval_result/multitask/`。

## Relevant files

- `XPolicyLab/policy/Pi_05/` — π0.5 adapter（`install.sh`/`deploy.yml`/`model.py`/`eval.sh`/`setup_eval_policy_server.sh`）
- `XPolicyLab/policy/Pi_05/deploy.yml` — `repo_id: arx_x5_sim`、`train_config_name: pi05_base_aloha_full_sim_arx-x5_seed_0`、`checkpoint_num: 59999`
- `XPolicyLab/utils/checkpoint_resolver.py` — checkpoint 目录解析（`{bench}-{ckpt}-{env}-{action}-{seed}` + `checkpoints/<ckpt_name>` fallback）
- `scripts/eval_policy.sh` / `scripts/eval_policy_multitask.py` — RoboTwin 评测入口
- `env_cfg/eval/all_tasks.yml` — 任务列表与 GPU 配置
- `env_cfg/aloha_agilex.yml` + `env_cfg/robot/_robot_info.json` — `aloha_agilex` 机器人动作维度 [6,6]+[1,1]

## Verification

1. nvidia-smi / torch CUDA 冒烟通过（否则先解决 5090 兼容性）。
2. `scripts/test_render.py` 能渲染（仿真可用）。
3. `bash install.sh` 后 `openpi/.venv` 存在，`uv run python -c "import openpi, XPolicyLab"` 成功。
4. checkpoint 解压后 `params/`、`assets/<repo_id>/` 存在且 `repo_id` 与 deploy.yml 一致。
5. `--dry-run` 无调度错误；单任务评测出视频且成功率 > 0（参考 leaderboard π0.5 约 80%+）。

## Further Considerations

1. 5090 兼容性：RoboTwin 固定 torch 2.4.1(cu121) 可能无法在 sm_120 上运行；若失败需升级 torch 并重装 curobo/pytorch3d/warp-lang（影响最大，先验证）。
2. OpenPI 的 jax[cuda12]==0.5.3 可能不支持 sm_120，需升级。
3. 若导师后续要“全流程”，再补：下载数据集 → transform_lerobot_v30 → finetune → eval（LoRA 需 >46GB，2×5090=64GB 勉强可 LoRA）。
