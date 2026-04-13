#!/bin/bash

################################################################################
# PhysioTouch Stage1 联合训练脚本
# 实验配置: 平衡权重 (alpha=0.75, beta=0.25) + 稳定性优化
# 
# 版本: v2.3 (Stability Enhancement)
# 优化重点:
# 1. 降低batch size (192→128) 预留33% GPU内存余量，防止OOM
# 2. 降低学习率 (blr: 1e-3→5e-4) 减少loss暴涨幅度
# 3. 延长warmup (40→60 epochs) 更平滑的学习率增长
# 4. 增强梯度裁剪 (1.0→0.5) 防止loss暴涨后的数值不稳定
# 5. 添加Loss暴涨检测和定期内存清理
#
# 说明: 基于loss尺度调整，使静态和动态对总loss的贡献接近1:1
################################################################################

echo "========================================="
echo "  PhysioTouch Stage1 联合训练"
echo "  实验: 平衡权重配置"
echo "========================================="
echo ""

# 实验配置
EXPERIMENT_NAME="joint_alpha0.75_beta0.25"
OUTPUT_DIR="./log/${EXPERIMENT_NAME}"
ALPHA=0.75   # 静态loss权重 (增大以平衡较小的loss_static)
BETA=0.25    # 动态loss权重 (降低以平衡较大的loss_dynamic)
GPU_IDS="0,1,2,3"  # ⭐ 指定使用的GPU，例如 "0" 或 "0,1,2,3"

mkdir -p ${OUTPUT_DIR}

# ⭐ 设置可见的GPU
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "🎮 使用GPU: $GPU_IDS"
else
    echo "🎮 使用所有可用GPU"
fi

echo "⚙️  训练配置:"
echo "  - 实验名称: ${EXPERIMENT_NAME}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 训练模式: 🎯 联合训练 (Joint Training)"
echo "  - Alpha (静态权重): ${ALPHA}"
echo "  - Beta (动态权重): ${BETA}"
echo "  - 设计理由: 平衡loss_static和loss_dynamic的贡献"
echo "    (预期loss_dynamic约为loss_static的3-4倍)"
echo ""
echo "📊 优化参数 (v2.3 Stability Enhancement):"
echo "  - Batch size: 128 (每GPU ~43, 降低33%预留内存)"
echo "  - Base LR: 5e-4 (降低50%减少loss暴涨)"
echo "  - Warmup: 5 epochs (延长50%平滑增长)"
echo "  - Gradient clip: 0.5 (增强裁剪)"
echo "  - Epochs: 100"
echo "  - 掩码比例: 75%"
echo "  - 梯度累积: 2"
echo "  - 分布式端口: 29502"
echo "  - Num workers: 8 (降低CPU负载)"
echo ""

export MASTER_PORT=29502

echo "🚀 启动联合训练..."
echo "========================================="
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    Main_stage1.py \
    --batch_size 196 \
    --epochs 20 \
    --blr 2e-4 \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --num_workers 12 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --use_video \
    --mask_ratio 0.75 \
    --use_joint_training \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --save_freq 10 \
    --keep_last_n 2 \
    --accum_iter 1  \
    --resume "log/joint_alpha0.75_beta0.25/last.pth"
    

echo ""
echo "========================================="
echo "  训练完成！"
echo "  检查结果: ${OUTPUT_DIR}/"
echo "  - best.pth: 最佳模型"
echo "  - last.pth: 最后一个epoch"
echo "  - log.txt: 训练日志"
echo "========================================="