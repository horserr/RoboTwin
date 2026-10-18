#!/usr/bin/env bash
# =============================================================================
# RoboTwin devcontainer 自检脚本
#
# 用途：容器 rebuild / 长时间闲置后，快速确认「环境是否还能直接跑评测」。
#       默认只读检查，不改动任何东西；加 --fix 才做两处安全修复：
#         1) 补 /usr/share/vulkan/icd.d/nvidia_icd.json 软链
#         2) uv python install 3.11（修复 openpi/.venv 解释器悬空）
#       不会重装依赖、不会动 /opt/venv、不会删任何数据。
#
# 用法：
#   bash .devcontainer/check.sh            # 全量检查
#   bash .devcontainer/check.sh --quick    # 跳过渲染自检（省 ~10s）
#   bash .devcontainer/check.sh --fix      # 先检查，再做上述安全修复
#   bash .devcontainer/check.sh --soft     # 即使有 FAIL 也返回 0（供 onCreateCommand 使用，
#                                         # 避免自检失败导致容器创建整体中断）
#
# 退出码：0 = 无 FAIL（允许 WARN）；1 = 存在 FAIL；2 = 参数错误
#        加了 --soft 时恒为 0
# 对应文档：SETUP_COMMANDS.md 第 8 / 9 / 11 / 12 节
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
PY="${VENV}/bin/python"
OPENPI_PY="${ROOT}/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
ICD_SRC="/etc/vulkan/icd.d/nvidia_icd.json"
ICD_LINK="/usr/share/vulkan/icd.d/nvidia_icd.json"

FIX=0
RUN_RENDER=1
SOFT=0
for arg in "$@"; do
  case "${arg}" in
    --fix) FIX=1 ;;
    --quick | --no-render) RUN_RENDER=0 ;;
    --soft) SOFT=1 ;;
    -h | --help)
      sed -n '2,24p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "未知参数: ${arg}（可用 --fix / --quick / --soft / --help）" >&2
      exit 2
      ;;
  esac
done

# ----------------------------- 输出工具 -------------------------------------
c() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
PASS=0
WARN=0
FAIL=0
pass() { printf '  %s %s\n' "$(c 32 PASS)" "$*"; PASS=$((PASS + 1)); }
warn() { printf '  %s %s\n' "$(c 33 WARN)" "$*"; WARN=$((WARN + 1)); }
fail() { printf '  %s %s\n' "$(c 31 FAIL)" "$*"; FAIL=$((FAIL + 1)); }
info() { printf '  %s %s\n' "$(c 2 ....)" "$*"; }
section() { printf '\n%s\n' "$(c '1;36' "== $* ==")"; }

printf '%s\n' "$(c 1 'RoboTwin 环境自检')  $(c 2 "root=${ROOT}  venv=${VENV}  fix=${FIX}")"

# ----------------------------- 1. GPU ---------------------------------------
section "1/9 GPU 状态"
gpu_rows=""
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "找不到 nvidia-smi（驱动未就绪，或容器未直通 GPU）"
else
  gpu_rows="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)"
  if [ -z "${gpu_rows}" ]; then
    fail "nvidia-smi 无法枚举 GPU"
  else
    n_gpu="$(printf '%s\n' "${gpu_rows}" | grep -c .)"
    pass "检测到 ${n_gpu} 块 GPU"
    while IFS=',' read -r idx name used total; do
      idx="$(echo "${idx}" | xargs)"
      name="$(echo "${name}" | xargs)"
      used="$(echo "${used}" | xargs)"
      total="$(echo "${total}" | xargs)"
      [ -z "${idx}" ] && continue
      pct=$((used * 100 / total))
      if [ "${pct}" -ge 50 ]; then
        warn "GPU${idx} ${name} 显存已用 ${used}/${total} MiB (${pct}%)，跑评测前先确认占用来源"
      else
        info "GPU${idx} ${name} 显存 ${used}/${total} MiB (${pct}%)"
      fi
    done <<<"${gpu_rows}"

    # 有显存占用但看不到进程 = 容器外/旧容器残留（第 9 节）
    apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')"
    used_total="$(printf '%s\n' "${gpu_rows}" | awk -F, '{gsub(/ /,"",$3); s+=$3} END{print s+0}')"
    if [ "${apps}" -eq 0 ] && [ "${used_total}" -gt 2048 ]; then
      warn "共 ${used_total} MiB 显存被占用但无对应进程（容器外残留），可能导致 DeviceLost / 假成功"
    fi
  fi
fi

# ----------------------------- 2. 主环境 ------------------------------------
section "2/9 RoboTwin 主环境 ${VENV}"
if [ ! -x "${PY}" ]; then
  fail "${PY} 不存在或不可执行 → 需重建依赖（见 .devcontainer/onCreate.sh 的 uv sync --frozen）"
else
  pyver="$("${PY}" -V 2>&1)"
  if [ $? -eq 0 ]; then
    pass "解释器可用：${pyver}"
  else
    fail "解释器不可用：${pyver}"
  fi

  imp="$("${PY}" -W ignore -c 'import torch,sapien,warp,toppra; print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"
  case "${imp}" in
    [0-9]*" True") pass "核心包可导入，torch ${imp%% *}，CUDA 可用" ;;
    [0-9]*" False") fail "torch ${imp%% *} 可导入但 CUDA 不可用（检查驱动/直通）" ;;
    *) fail "核心包导入失败：${imp}" ;;
  esac
fi

# ----------------------------- 3. Vulkan ICD --------------------------------
# section "3/9 Vulkan ICD（sapien 渲染依赖，第 11 节）"
# if [ -f "${ICD_LINK}" ]; then
#   pass "${ICD_LINK} 已就位"
# elif [ "${FIX}" -eq 1 ] && [ -f "${ICD_SRC}" ]; then
#   if mkdir -p "$(dirname "${ICD_LINK}")" && ln -sfn "${ICD_SRC}" "${ICD_LINK}"; then
#     pass "已补软链：${ICD_LINK} -> ${ICD_SRC}"
#   else
#     fail "补软链失败（权限？）：${ICD_LINK}"
#   fi
# else
#   if [ -f "${ICD_SRC}" ]; then
#     fail "${ICD_LINK} 缺失，sapien 会锁死 VK_ICD_FILENAMES（加 --fix 可修复）"
#   else
#     fail "NVIDIA ICD 完全缺失（${ICD_SRC} 不存在）：容器未挂载 graphics 驱动能力"
#   fi
# fi

# if [ -x "${PY}" ]; then
#   vk_raw="$("${PY}" -W ignore -c 'import os,sapien; print(os.environ.get("VK_ICD_FILENAMES",""))' 2>/dev/null)"
#   vk_rc=$?
#   if [ "${vk_rc}" -ne 0 ]; then
#     warn "无法导入 sapien，跳过 VK_ICD_FILENAMES 检查"
#   else
#     vk_env="$(printf '%s\n' "${vk_raw}" | tail -1)"
#     if [ -n "${vk_env}" ]; then
#       warn "sapien 强制指定了 VK_ICD_FILENAMES=${vk_env}（失去 Mesa 回退，渲染更易失败）"
#     else
#       pass "sapien 未覆盖 VK_ICD_FILENAMES（使用系统全部 ICD）"
#     fi
#   fi
# fi

# ----------------------------- 4. 渲染自检 ----------------------------------
section "4/9 渲染自检 scripts/test_render.py"
if [ "${RUN_RENDER}" -eq 0 ]; then
  info "--quick 模式已跳过"
elif [ ! -x "${PY}" ]; then
  fail "跳过：${PY} 不可用"
else
  render_out="$(
    cd "${ROOT}" 2>/dev/null || exit 1
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" timeout 300 "${PY}" scripts/test_render.py 2>&1
  )"
  if printf '%s' "${render_out}" | grep -q "Render Well"; then
    pass "渲染链路正常（Render Well）"
  else
    fail "渲染链路异常：$(printf '%s' "${render_out}" | tail -3 | tr '\n' ' ')"
  fi
fi

# ----------------------------- 5. 依赖锁 ------------------------------------
section "5/9 依赖锁一致性"
if ! command -v uv >/dev/null 2>&1; then
  warn "找不到 uv（镜像内应自带，检查 PATH）"
else
  lock_out="$(cd "${ROOT}" && uv lock --check 2>&1)"
  if [ $? -eq 0 ]; then
    pass "uv.lock 与 pyproject.toml 一致"
  else
    fail "uv.lock 不同步（需 uv lock），否则 onCreate 的 uv sync --frozen 会失败：$(printf '%s' "${lock_out}" | tail -1)"
  fi
fi

# ----------------------------- 6. openpi ------------------------------------
section "6/9 π0.5 策略侧解释器（openpi/.venv，第 8 节）"
if [ -x "${OPENPI_PY}" ] && openpi_ver="$("${OPENPI_PY}" -V 2>&1)"; then
  pass "解释器可用：${openpi_ver}"
else
  if [ "${FIX}" -eq 1 ] && command -v uv >/dev/null 2>&1; then
    info "执行 uv python install 3.11 ..."
    if uv python install 3.11 >/dev/null 2>&1 && [ -x "${OPENPI_PY}" ] && openpi_ver="$("${OPENPI_PY}" -V 2>&1)"; then
      pass "已修复解释器：${openpi_ver}"
    else
      fail "uv python install 3.11 后仍不可用 → 完整重装：bash XPolicyLab/policy/Pi_05/install.sh"
    fi
  else
    fail "openpi/.venv/bin/python 悬空（uv 管理的 Python 丢失），执行 uv python install 3.11 或加 --fix"
  fi
fi

if [ -x "${OPENPI_PY}" ]; then
  jax_out="$("${OPENPI_PY}" -W ignore -c 'import jax; print(jax.__version__)' 2>&1 | tail -1)"
  case "${jax_out}" in
    [0-9]*) pass "jax 可导入：${jax_out}" ;;
    *) warn "jax 导入异常：${jax_out}" ;;
  esac
fi

# ----------------------------- 7. 数据资产 ----------------------------------
section "7/9 数据资产（assets / checkpoint）"
if [ -f "${ROOT}/assets/.extract_complete" ]; then
  pass "assets 已解压（installAssets.sh 会跳过下载）"
else
  warn "缺少 assets/.extract_complete → 重建时会重新下载约 11G 资产"
fi
ckpt="${ROOT}/XPolicyLab/policy/Pi_05/checkpoints/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999"
if [ -d "${ckpt}" ]; then
  pass "checkpoint 29999 已就位"
else
  warn "缺少 checkpoint：${ckpt}（见第 3 节的 hf download）"
fi

# ----------------------------- 8. 注入的环境变量 ----------------------------
section "8/9 注入的环境变量"
for v in GITHUB_USERNAME GH_TOKEN HF_TOKEN; do
  if [ -n "${!v:-}" ]; then
    pass "${v} 已注入"
  else
    warn "${v} 为空（chezmoi 恢复 dotfiles / hf download 可能失败）"
  fi
done

# ----------------------------- 9. 其它 --------------------------------------
section "9/9 其它"
if [ -L "${ROOT}/.venv/bin/python" ] && [ ! -e "${ROOT}/.venv/bin/python" ]; then
  info "根 .venv 是宿主机遗留的悬空链接（容器内请勿使用，也不要删除）"
fi
if avail_kb="$(df -Pk "${ROOT}" 2>/dev/null | awk 'NR==2{print $4}')"; then
  avail_gib=$((avail_kb / 1024 / 1024))
  if [ "${avail_gib}" -ge 10 ]; then
    pass "工作区剩余磁盘 ${avail_gib} GiB"
  else
    warn "工作区剩余磁盘仅 ${avail_gib} GiB（评测会持续写 eval_result/）"
  fi
fi

# ----------------------------- 汇总 ----------------------------------------
printf '\n%s\n' "$(c '1;36' '== 汇总 ==')"
printf '  PASS=%d  WARN=%d  FAIL=%d\n' "${PASS}" "${WARN}" "${FAIL}"
if [ "${FAIL}" -gt 0 ]; then
  printf '  %s\n' "$(c 31 '存在 FAIL：先按上面的提示修复，再跑评测（详见 SETUP_COMMANDS.md 第 8/9/11/12 节）')"
  if [ "${SOFT}" -eq 1 ]; then
    printf '  %s\n' "$(c 33 '--soft：返回 0，不中断容器创建流程；请手动修复上面的 FAIL')"
    exit 0
  fi
  exit 1
fi
printf '  %s\n' "$(c 32 '检查通过，可以跑冒烟：bash scripts/eval_policy.sh multitask --config env_cfg/eval/smoke_tasks.yml --policy-name Pi_05 --ckpt-name lerobot_v30 --env-cfg-type aloha_agilex --policy-conda-env uv --eval-env-conda-env RoboTwin --action-type joint --test-num 5')"
exit 0
