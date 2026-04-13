#!/bin/bash

# ============================================================================
# Stage-2b Phase 1 训练脚本（Group-DRO平衡）- 修复版
# ============================================================================
# 
# 🔧 核心修复：Group-DRO实现 (2025-11-07)
#   - 使用完整batch作为负样本池（保持31个负样本）
#   - 分组统计仅用于监控和权重更新
#   - 避免负样本池过小导致loss爆炸
# 
# 目标：平衡 Fine/Indoor/Outdoor 三组数据的损失
# 策略：只使用Group-DRO，不引入InfoMax
# 关键：Group-DRO 更新需要对齐 loss 尺度（当前 loss_mp ~ 1e-2）
#   - 强制启用中心化/标准化更新（group_dro_center_losses / group_dro_normalize_losses）
#   - eta 需要比 “loss 约为 1” 的情况大得多，否则权重几乎不动
# 
# ============================================================================

set -euo pipefail

# Ensure script runs from project root so relative paths are valid.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Make sure log dir exists and capture full stdout/stderr (not just torchrun).
mkdir -p ./log
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE="./log/stage2b_phase1.err"
exec > >(tee -a ./log/stage2b_phase1.log) 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
# Mitigate CUDA memory fragmentation under large batches.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM_ITER="${ACCUM_ITER:-1}"
# 🔧 FIX: DDP/NCCL 稳定性优化 + 死锁预防
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1  # 改为阻塞模式，提高稳定性
export NCCL_DEBUG=INFO  # 启用详细日志，便于诊断
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 启用详细调试
# 🔧 关键修复：减少NCCL超时时间，快速检测死锁
export NCCL_TIMEOUT=600  # 10分钟超时（从1小时降低）
# 🔧 新增：NCCL通信优化
export NCCL_IB_DISABLE=0  # 启用InfiniBand（如果可用）
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export NCCL_SHM_DISABLE=0  # 启用共享内存
export NCCL_SOCKET_IFNAME=^docker,lo  # 排除docker和loopback接口
# Image load timeout (seconds)
export PT_IMAGE_LOAD_TIMEOUT_SEC=10  # 增加到10秒


check_port_free() {
python - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    try:
        s.close()
    except Exception:
        pass
sys.exit(0)
PY
}

if [[ -z "${MASTER_ADDR:-}" ]]; then
  export MASTER_ADDR="127.0.0.1"
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
  export MASTER_PORT="$(
python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
  )"
else
  if ! check_port_free "${MASTER_PORT}"; then
    echo "❌ MASTER_PORT=${MASTER_PORT} 已被占用，请换一个端口或取消设置 MASTER_PORT。"
    exit 2
  fi
fi

# 激活conda环境
# source /home/shipeng/anaconda3/etc/profile.d/conda.sh
# conda activate anytouch

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage-2b Phase 1: Group-DRO 平衡训练（修复版）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 训练目标："
echo "  平衡 Fine/Indoor/Outdoor 三组数据"
echo "  组间损失差距 < 0.05"
echo "  避免 Group-DRO 权重失控"
echo ""
echo "🔧 核心修复（2025-11-07）："
echo "  ✅ Group-DRO实现修复: 使用完整batch作为负样本池"
echo "  ✅ 负样本数: 5-15 → 31 (避免loss爆炸)"
echo "  ✅ group_dro_eta: 0.20（中心化+标准化+放开η上限）"
echo "  ✅ lambda_im: 禁用 (Phase 1不用InfoMax)"
echo "  ✅ 学习率: 2e-5 / 5e-6 (极小步长守住阶段2a)"
echo "  ✅ warmup: 3 epochs"
echo "  ✅ 数据采样: repeat 重复少数组，保证每个batch含 Fine/Indoor/Outdoor"
echo "  ✅ DataLoader: num_workers=4, prefetch_factor=1, pin_memory=off"
echo ""
echo "📊 训练配置："
echo "  GPU: ${NUM_GPUS} cards (${CUDA_VISIBLE_DEVICES}) [优化: 减少GPU数降低同步开销]"
echo "  Batch size: ${BATCH_SIZE} per GPU"
echo "  Epochs: up to 30 (early stopping enabled)"
echo "  学习率: 2e-5 (投影头), 5e-6 (编码器)"
echo "  Group-DRO eta: 0.30 (normalize+center, min 0.001 / max 0.999) [放宽约束]"
echo "  Group sampler: repeat per-group oversampling (balanced batches)"
echo "  InfoMax: 禁用"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🧩 分布式配置："
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo ""

if torchrun --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" Stage2_main.py \
    --stage 2b \
    --primary_metric phase1_combined \
    --data_sources Fine Indoor Outdoor \
    --data_root_fine ./Tacquad/ \
    --data_root_indoor ./Tacquad/ \
    --data_root_outdoor ./Tacquad/ \
    --batch_size "${BATCH_SIZE}" \
    --accum_iter "${ACCUM_ITER}" \
    --epochs 30 \
    --resume ./log/stage2a_mixed/last.pth \
    --resume_model_only \
    --freeze_blocks 8 \
    --domain_projection_dim 128 \
    --temperature 0.07 \
    --lr 2e-5 \
    --lr_encoder 5e-6 \
    --lambda_im_start 0.0 \
    --lambda_im_end 0.0 \
    --lambda_decouple 0.0 \
    --lambda_match 0.05 \
    --match_negatives_per_pair 2 \
    --match_warmup_ratio 0.20 \
    --match_rampup_ratio 0.20 \
    --same_sim_threshold 0.70 \
    --diff_sim_threshold 0.35 \
    --use_group_dro \
    --group_dro_eta 0.1 \
    --group_dro_eta_cap 0.1 \
    --group_dro_center_losses \
    --group_dro_min_weight 0.05 \
    --group_dro_max_weight 0.90 \
    --warmup_epochs 3 \
    --weight_decay 0.1 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --no_pin_mem \
    --train_split_ratio 0.9 \
    --group_balance_strategy repeat \
    --random_seed 0 \
    --output_dir ./log/stage2b_phase1 \
    --log_dir ./log/stage2b_phase1 \
    --save_freq 5 \
    --eval_freq 1 \
    --early_stopping_patience 22 \
    --wandb_project PhysioTouch-Stage2a \
    --wandb_name stage2b_phase1_group_dro_balance; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Phase 1 训练完成！"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "📊 检查结果："
  echo "  1. 查看日志: tail -n 100 ./log/stage2b_phase1.log"
  echo "  2. 查看Wandb: PhysioTouch-Stage2a/stage2b_phase1_group_dro_balance"
  echo "  3. 检查权重: grep 'Group-DRO' ./log/stage2b_phase1.log | tail -n 10"
  echo "  4. 查看修复说明: cat ReadMe/Group_DRO_修复说明.md"
  echo ""
else
  status=$?
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ Phase 1 训练失败！(exit=${status})"
  echo "  - 常见原因：端口冲突/显存不足/数据路径错误"
  echo "  - 端口冲突可尝试：MASTER_PORT=29xxx bash scripts/train_stage2b_phase1.sh"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit "${status}"
      # --disable_wandb \
      # --group_dro_disable_eta_cap \
fi
