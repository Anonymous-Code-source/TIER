#!/bin/bash

# ============================================================================
# Stage-2a 训练脚本（基础对齐）
# ============================================================================
# 
# 目标：学习跨传感器的基础对齐
# 数据：Fine + Indoor + Outdoor (110 objects, ~70K 样本)
# 损失：Multi-Positive InfoNCE only
# 
# 预期效果：
#   - 训练集 same_contact_sim: 0.75-0.80
#   - 验证集 same_contact_sim: 0.60-0.70
#   - 训练-验证差距: < 0.15
# 
# ============================================================================

# cd /home/shipeng/PhysioTouch

# # 激活conda环境
# source /home/shipeng/anaconda3/etc/profile.d/conda.sh
# conda activate anytouch

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stage-2a 训练开始：基础对齐（InfoNCE only）- 8卡分布式"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "数据配置："
echo "  数据源: Fine + Indoor + Outdoor"
echo "  物体数: 110 objects"
echo "  样本数: ~70K samples"
echo ""
echo "模型配置："
echo "  freeze_blocks: 8 (冻结前8层，保守策略)"
echo "  可训练参数: ~80M"
echo ""
echo "训练配置："
echo "  GPU数量: 8 (GPU 0-7)"
echo "  batch_size: 32 × 8卡 = 256 (每卡32)"
echo "  accum_iter: 1"
echo "  有效batch: 256"
echo "  lr: 5e-4 (投影头), 1e-4 (编码器)"
echo "  epochs: 50"
echo "  temperature: 0.07"
echo "  lambda_match: 0.08 (小权重辅助)"
echo "  match_negatives_per_pair: 2"
echo "  stage2_use_sensor_token: 关闭"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

torchrun --nproc_per_node=8 --master_port=29501 Stage2_main.py \
    --stage 2a \
    --data_sources Fine Indoor Outdoor \
    --data_root_fine ./Tacquad/ \
    --data_root_indoor ./Tacquad/ \
    --data_root_outdoor ./Tacquad/ \
    --batch_size 32 \
    --accum_iter 1 \
    --epochs 50 \
    --stage1_checkpoint ./log/joint_alpha0.75_beta0.25/best.pth \
    --freeze_blocks 8 \
    --temperature 0.07 \
    --dynamic_similarity_window \
    --similarity_window_low_start 0.0 \
    --similarity_window_high_start 1.0 \
    --similarity_window_low_end 0.2 \
    --similarity_window_high_end 0.8 \
    --lambda_match 0.08 \
    --match_negatives_per_pair 2 \
    --match_warmup_ratio 0.25 \
    --match_rampup_ratio 0.25 \
    --base_frame_jitter 0 \
    --base_frame_jitter_prob 0 \
    --same_sim_threshold 0.60 \
    --diff_sim_threshold 0.40 \
    --lr 5e-4 \
    --lr_encoder 1e-4 \
    --warmup_epochs 10 \
    --num_workers 12 \
    --train_split_ratio 0.9 \
    --random_seed 42 \
    --output_dir ./log/stage2a_mixed \
    --log_dir ./log/stage2a_mixed \
    --save_freq 5 \
    --eval_freq 1 \
    --early_stopping_patience 130 \
    --wandb_project PhysioTouch-Stage2a \
    --wandb_name stage2a_mixed_all_data \
    2>&1 | tee ./log/stage2a_mixed.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage-2a 训练完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "结果检查："
echo "  日志文件: ./log/stage2a_mixed.log"
echo "  模型目录: ./log/stage2a_mixed/"
echo "  Wandb: PhysioTouch-Stage2/stage2a_mixed_all_data"
echo ""
echo "关键指标（验证集）："
echo "  ✅ same_contact_sim > 0.60"
echo "  ✅ diff_contact_sim < same_contact_sim"
echo "  ✅ 训练-验证差距 < 0.15"
echo ""
echo "如果满足上述条件，继续训练 Stage-2b："
echo "  Phase1建议从 ./log/stage2a_mixed/last.pth 恢复（更稳妥）"
echo ""
