"""
config_v1.py - 精简版本
保留所有配置参数，添加更详细的注释说明

版本: v1.0
改进: 
- 添加详细的参数分组和注释
- 标注当前 Stage1 训练中实际使用的参数
- 标注未来多模态训练可能使用的参数
"""

import argparse


def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(description='PhysioTouch Stage1 Training Configuration')

    # ============================================================
    # 基础训练参数 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument('--output_dir', default='output_dir',
                        help='模型和日志保存路径')
    parser.add_argument('--log_dir', default='output_dir',
                        help='TensorBoard日志保存路径')
    parser.add_argument('--device', default='cuda',
                        help='训练设备: cuda 或 cpu')
    parser.add_argument('--gpu_ids', default='0,1,2', type=str,
                        help='指定使用的GPU ID，例如: "0" 或 "0,1,2,3"。留空则使用所有可用GPU')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每个GPU的批次大小')
    parser.add_argument('--epochs', default=400, type=int,
                        help='训练总轮数')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积步数，用于增加有效批次大小')

    # ============================================================
    # MAE 模型参数 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='MAE掩码比例，默认75%')
    parser.add_argument("--norm_pix_loss", action='store_true',
                        help='是否对像素值进行归一化后计算损失')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--use_video", action='store_true',
                        help='是否使用视频模式（4帧序列）')
    
    # ⭐ 新增：静态重建损失相关参数（参考 AnyTouch）
    parser.add_argument('--static_ratio', type=float, default=0.5,
                        help='静态重建模式的使用比例（0.0-1.0），默认0.5表示50%%静态，50%%动态')
    parser.add_argument("--use_static", action='store_true',
                        help='是否启用静态重建损失（单帧图像MAE）')
    
    # ⭐ 新增：联合训练权重参数
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='静态损失权重（用于联合训练），默认0.5')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='动态损失权重（用于联合训练），默认0.5')
    
    # ⭐ 新增：联合训练模式开关
    parser.add_argument('--use_joint_training', action='store_true',
                        help='是否使用联合训练模式（同时计算静态和动态loss）')
    parser.add_argument('--monitor_gradient', action='store_true',
                        help='是否监控梯度范数和冲突情况（用于调试）')
    
    # ⭐ 新增：检查点保存策略参数
    parser.add_argument('--save_freq', type=int, default=10,
                        help='每多少个epoch保存一次带编号的检查点checkpoint-N.pth（默认10），用于恢复训练')
    parser.add_argument('--keep_last_n', type=int, default=2,
                        help='只保留最近N个带编号的检查点checkpoint-N.pth（默认2），best.pth和last.pth始终保留')

    # ============================================================
    # 传感器相关参数 (⚠️ Stage1 未使用，为多传感器预留)
    # ============================================================
    parser.add_argument("--use_sensor_token", action='store_true',
                        help='是否使用传感器token')
    parser.add_argument("--use_same_patchemb", action='store_true',
                        help='是否使用相同的patch embedding')
    parser.add_argument("--sensor_token_for_all", action='store_true',
                        help='是否为所有层使用传感器token')
    parser.add_argument("--new_decoder_sensor_token", action='store_true',
                        help='是否使用新的解码器传感器token')
    parser.add_argument("--beta_start", type=float, default=0.0,
                        help='Beta调度起始值')
    parser.add_argument("--beta_end", type=float, default=0.75,
                        help='Beta调度结束值')

    # ============================================================
    # 多模态参数 (❌ Stage1 不使用，仅用于多模态训练)
    # ============================================================
    parser.add_argument("--alpha_vl", type=float, default=0.2,
                        help='视觉-语言对比损失权重')
    parser.add_argument("--alpha_vt", type=float, default=0.2,
                        help='视觉-触觉对比损失权重')
    parser.add_argument("--alpha_lt", type=float, default=1.0,
                        help='语言-触觉对比损失权重')
    parser.add_argument("--cross_alpha", type=float, default=1.0,
                        help='跨传感器匹配损失权重')
    parser.add_argument("--no_mae", action='store_true',
                        help='是否禁用MAE损失')
    parser.add_argument("--TAG_times", type=int, default=1,
                        help='TAG数据集重复次数')
    parser.add_argument("--cross_iter", type=int, default=6,
                        help='交叉注意力迭代次数')

    # ============================================================
    # 优化器参数 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减系数')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='学习率（绝对值），如不指定则根据blr计算')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='基础学习率: lr = blr * batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='学习率下界')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='学习率预热轮数')

    # ============================================================
    # 数据加载参数 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument('--num_workers', type=int, default=32, metavar='N',
                        help='数据加载的工作进程数')

    # ============================================================
    # 检查点和恢复 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument('--resume', default='',
                        help='从检查点恢复训练的路径')
    parser.add_argument('--mae_dir', default=None,
                        help='MAE模型检查点路径')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='起始epoch编号')
    # ⚠️ 注意：--save_freq 参数已移至上方"检查点保存策略参数"部分（第58行）

    # ============================================================
    # 其他参数
    # ============================================================
    parser.add_argument('--init_temp', type=float, default=0.07,
                        help='初始温度参数（用于对比学习）')
    parser.add_argument("--seed", type=int, default=0,
                        help='随机种子')

    # ============================================================
    # LoRA 参数 (❌ Stage1 不使用)
    # ============================================================
    parser.add_argument("--convert_to_lora", action='store_true',
                        help='是否转换为LoRA模型')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout概率')

    # ============================================================
    # 分布式训练参数 (✅ Stage1 使用)
    # ============================================================
    parser.add_argument("--distributed", action='store_true',
                        help='是否使用分布式训练')
    parser.add_argument('--world_size', default=4, type=int,
                        help='分布式训练的总进程数')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='分布式训练的本地rank')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='是否在ITP上进行分布式训练')
    parser.add_argument('--dist_url', default='env://',
                        help='分布式训练的URL')

    return parser

