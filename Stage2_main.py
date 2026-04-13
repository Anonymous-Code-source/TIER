"""
Stage2_main.py - Stage-2 跨传感器无感表征学习主训练脚本

功能:
- Stage-2a: 对齐预热（λ_im = 0）
- Stage-2b: 去域优化（λ_im = 0.05→0.2）
- Stage-2c: Group-DRO（可选）

版本: v2.0
日期: 2025-11-01
更新: 使用 config2.py 管理配置，与 Stage-1 完全分离
"""

import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model.stage2_model import load_stage1_encoder, Stage2Model
from dataloader.build_dataloaders import build_stage2_dataloader
from Stage2_engine import train_one_epoch, validate

# 从 config2.py 导入配置管理函数
from config2 import (
    parse_args,
    apply_stage_defaults,
    print_config,
    validate_config,
    mark_explicit_args,
)

# Wandb (可选)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb not installed, logging disabled")


def get_lambda_im_schedule(epoch, args):
    """
    获取当前 epoch 的 lambda_im 值
    
    Stage-2a: lambda_im = 0 (只对齐)
    Stage-2b: lambda_im 延迟启动 + 渐进增长
        - 前30% epochs: lambda_im = 0 (专注判别学习)
        - 中间40% epochs: lambda_im 线性增长 0 → lambda_im_end
        - 后30% epochs: lambda_im = lambda_im_end (平衡状态)
    Stage-2c: lambda_im = 0.2 (固定)
    """
    if args.stage == '2a':
        return 0.0
    elif args.stage == '2b':
        # 延迟启动策略 (改进版)
        warmup_ratio = getattr(args, 'lambda_im_warmup_ratio', 0.3)  # 前30%不启用
        rampup_ratio = getattr(args, 'lambda_im_rampup_ratio', 0.4)  # 中间40%渐进
        
        total_epochs = args.epochs
        warmup_epochs = int(total_epochs * warmup_ratio)
        rampup_epochs = int(total_epochs * rampup_ratio)
        
        if epoch < warmup_epochs:
            # 阶段1: 专注判别学习
            return 0.0
        elif epoch < warmup_epochs + rampup_epochs:
            # 阶段2: 渐进引入域不变性
            progress = (epoch - warmup_epochs) / rampup_epochs
            lambda_im = args.lambda_im_start + (args.lambda_im_end - args.lambda_im_start) * progress
            return lambda_im
        else:
            # 阶段3: 平衡状态
            return args.lambda_im_end
    elif args.stage == '2c':
        return args.lambda_im_end
    else:
        return args.lambda_im


def compute_combined_metric(val_stats, metric_type='phase1_combined', args=None):
    """
    计算综合指标（v2.0 - 修复版）
    
    Args:
        val_stats: 验证统计字典
        metric_type: 'phase1_combined' 或 'phase2_combined'
    
    Returns:
        combined_score: 综合得分 (0-1之间，越高越好)
    
    修复点:
    - Phase-2 结合多个域不变性指标（domain_acc, domain_acc_std, domain_entropy）
    - 避免单一指标饱和导致优化信号消失
    """
    same_sim = val_stats.get('same_contact_sim', 0.0)
    group_gap = val_stats.get('group_loss_gap', 0.0)
    domain_acc = val_stats.get('domain_acc', 0.333)
    
    # 🆕 获取详细域指标（Phase-2 专用）
    domain_acc_std = val_stats.get('domain_acc_std', 0.0)
    domain_entropy = val_stats.get('domain_entropy', 1.099)
    
    # 归一化 group_loss_gap 到 [0, 1]，gap越小越好
    # gap=0 → 1.0, gap=0.08 → 0.0, gap>0.08 → 0.0
    group_balance_score = max(0.0, 1.0 - group_gap / 0.08)
    
    if metric_type == 'phase1_combined':
        # Phase 1: 偏重对比学习质量，兼顾组平衡
        combined_score = (
            0.6 * same_sim +                    # 对比学习质量（偏重）
            0.4 * group_balance_score            # 组平衡
        )
    elif metric_type == 'phase2_combined':
        # 🆕 Phase 2: 改进的域不变性得分（结合多个指标）
        # 1. domain_acc_score: 越接近33%越好（允许可配置容忍区间）
        num_sensors = getattr(args, 'num_sensors', 3) if args is not None else 3
        target_acc = 1.0 / max(num_sensors, 1)
        tol = getattr(args, 'phase2_probe_acc_tol', 0.067) if args is not None else 0.067
        if tol > 0:
            domain_acc_score = 1.0 - abs(domain_acc - target_acc) / tol
        else:
            domain_acc_score = 1.0 - abs(domain_acc - target_acc)
        domain_acc_score = max(0.0, min(1.0, domain_acc_score))
        
        # 2. domain_std_score: 标准差越小越好（per-class acc 越均衡）
        # std=0 → 1.0, std=0.4 → 0.0, std>0.4 → 0.0
        domain_std_score = 1.0 - min(domain_acc_std / 0.4, 1.0)
        domain_std_score = max(0.0, domain_std_score)
        
        # 3. domain_entropy_score: 熵越高越好（分布越均匀）
        # 最大熵 = log(3) ≈ 1.099
        domain_entropy_score = domain_entropy / 1.099
        domain_entropy_score = max(0.0, min(1.0, domain_entropy_score))
        
        # 综合域不变性得分（加权平均）
        domain_invariance_score = (
            0.5 * domain_acc_score +      # 主要指标：准确率接近33%
            0.3 * domain_std_score +      # 辅助指标：per-class 均衡
            0.2 * domain_entropy_score    # 辅助指标：分布均匀
        )
        domain_invariance_score = max(0.0, min(1.0, domain_invariance_score))
        
        # Phase 2: 平衡对比学习、域不变性和组平衡
        combined_score = (
            0.4 * same_sim +                     # 对比学习质量
            0.3 * domain_invariance_score +      # 域不变性（改进版）
            0.3 * group_balance_score            # 组平衡
        )
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    
    return combined_score


def filter_mismatched_state_dict(model_state, checkpoint_state):
    """
    Filter out checkpoint keys whose tensor shapes don't match current model.
    """
    filtered = {}
    mismatched = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            mismatched.append((key, tuple(value.shape), None))
            continue
        if model_state[key].shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value
    return filtered, mismatched


def sanitize_projection_norm_transition(filtered_state, checkpoint_state):
    """
    处理 ProjectionHead 从 BatchNorm -> LayerNorm 的恢复兼容：
    若检测到旧checkpoint含 running_mean/running_var，则丢弃对应norm层的weight/bias，
    保留当前模型初始化的LN参数，避免统计语义迁移污染。
    """
    dropped = []
    norm_prefixes = set()
    for key in checkpoint_state.keys():
        if (
            key.endswith('running_mean') and
            '.net.1.' in key and
            'projection_head' in key
        ):
            norm_prefixes.add(key[:-len('running_mean')])
    if len(norm_prefixes) == 0:
        return filtered_state, dropped

    patched = dict(filtered_state)
    for prefix in sorted(norm_prefixes):
        for suffix in ('weight', 'bias'):
            target_key = f'{prefix}{suffix}'
            if target_key in patched:
                patched.pop(target_key)
                dropped.append(target_key)
    return patched, dropped


def main(args):
    """主训练函数"""
    
    # ====================================================================
    # 应用阶段默认值并验证配置
    # ====================================================================
    args = apply_stage_defaults(args)
    validate_config(args)
    
    # ====================================================================
    # 初始化分布式训练
    # ====================================================================
    misc.init_distributed_mode(args)
    
    # 美化打印配置
    print_config(args)
    
    device = torch.device(args.device)
    
    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    cudnn.benchmark = True
    
    # ====================================================================
    # 创建输出目录
    # ====================================================================
    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    
    # ====================================================================
    # 创建 DataLoader
    # ====================================================================
    print('\n' + '=' * 70)
    print('Loading Data')
    print('=' * 70)
    
    # 🔧 FIX 6: 构建 DataLoader（启用分布式sampler）
    train_loader = build_stage2_dataloader(
        stage=args.stage,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=getattr(args, 'prefetch_factor', 1),
        train_split_ratio=args.train_split_ratio,
        random_seed=args.random_seed,
        data_sources=args.data_sources,
        data_root_fine=args.data_root_fine,
        data_root_indoor=args.data_root_indoor,
        data_root_outdoor=args.data_root_outdoor,
        base_frame_jitter=getattr(args, 'base_frame_jitter', 0),
        base_frame_jitter_prob=getattr(args, 'base_frame_jitter_prob', 0.0),
        distributed=True,  # 启用分布式sampler
        group_balance_strategy=getattr(args, 'group_balance_strategy', 'none'),
        contact_sampler_enabled=getattr(args, 'contact_sampler_enabled', False),
        contact_sampler_max_per_object_per_round=getattr(
            args, 'contact_sampler_max_per_object_per_round', 1
        ),
    )
    
    val_loader = build_stage2_dataloader(
        stage=args.stage,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=getattr(args, 'prefetch_factor', 1),
        train_split_ratio=args.train_split_ratio,
        random_seed=args.random_seed,
        data_sources=args.data_sources,
        data_root_fine=args.data_root_fine,
        data_root_indoor=args.data_root_indoor,
        data_root_outdoor=args.data_root_outdoor,
        base_frame_jitter=getattr(args, 'base_frame_jitter', 0),
        base_frame_jitter_prob=getattr(args, 'base_frame_jitter_prob', 0.0),
        distributed=True,
        group_balance_strategy='none',
        contact_sampler_enabled=False
    )
    
    print(f'✅ Train samples: {len(train_loader.dataset)}')
    print(f'✅ Val samples: {len(val_loader.dataset)}')
    print(f'✅ Train batches: {len(train_loader)}')
    print(f'✅ Val batches: {len(val_loader)}')
    
    # ====================================================================
    # 创建模型
    # ====================================================================
    print('\n' + '=' * 70)
    print('Creating Model')
    print('=' * 70)
    
    # 加载 Stage-1 encoder
    stage1_model = load_stage1_encoder(
        args.stage1_checkpoint,
        device=device
    )
    
    # 创建 Stage-2 model
    adversarial_feature_target = getattr(args, 'adversarial_feature_target', 'domain')
    domain_projection_dim = int(getattr(args, 'domain_projection_dim', args.projection_dim))

    def _adv_target_feature_dim(target: str) -> int:
        return (
            int(args.projection_dim)
            if target in ('contact', 'contact_residual')
            else domain_projection_dim
        )

    discriminator_input_dim = _adv_target_feature_dim(adversarial_feature_target)
    if bool(getattr(args, 'phase2_adv_target_switch', False)):
        late_target = str(getattr(args, 'phase2_adv_target_late', 'contact_residual'))
        late_dim = _adv_target_feature_dim(late_target)
        if late_dim != discriminator_input_dim:
            raise ValueError(
                "phase2_adv_target_switch requires same discriminator input dim for "
                f"base target '{adversarial_feature_target}' (dim={discriminator_input_dim}) "
                f"and late target '{late_target}' (dim={late_dim}). "
                "Please choose targets with matching dims or disable switch."
            )
    print(f"🧭 Adversarial feature target: {adversarial_feature_target} "
          f"(discriminator input dim={discriminator_input_dim})")
    model = Stage2Model(
        stage1_model=stage1_model,
        freeze_blocks=args.freeze_blocks,
        projection_dim=args.projection_dim,
        domain_projection_dim=domain_projection_dim,
        discriminator_input_dim=discriminator_input_dim,
        use_discriminator=True,
        discriminator_hidden_dim=args.discriminator_hidden_dim,
        discriminator_dropout=args.discriminator_dropout,
        num_sensors=args.num_sensors,
        use_matching_head=(getattr(args, 'lambda_match', 0.0) > 0),
        use_domain_classifier=(float(getattr(args, 'lambda_domain_cls', 0.0)) > 0.0)
    )
    model.to(device)
    
    # 打印参数统计
    param_stats = model.get_param_count()
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {param_stats['total']:,}")
    print(f"   Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.1f}%)")
    print(f"   - Encoder: {param_stats['encoder']:,}")
    print(f"   - Projection: {param_stats['projection']:,}")
    print(f"   - Domain projection: {param_stats.get('domain_projection', 0):,}")
    print(f"   - Discriminator: {param_stats['discriminator']:,}")
    print(f"   - Domain classifier: {param_stats.get('domain_classifier', 0):,}")
    print(f"   - Matching head: {param_stats['matching']:,}")
    
    model_without_ddp = model
    print(f"✅ Model created on {device}")
    
    # ====================================================================
    # 创建优化器（支持对抗式训练）
    # ====================================================================
    print('\n' + '=' * 70)
    print('Creating Optimizer')
    print('=' * 70)
    
    param_groups = model.get_trainable_params()
    
    # 🆕 对抗式训练：分离Encoder和Discriminator优化器
    use_adversarial = getattr(args, 'use_adversarial_training', False)
    
    if use_adversarial and args.stage == '2b':
        # 对抗式训练模式
        base_lr = args.lr if args.lr > 0 else 1e-6
        lr_scale_encoder = args.lr_encoder / base_lr if base_lr > 0 else 1.0
        
        # Encoder + Projection 优化器
        optimizer_encoder = torch.optim.AdamW(
            [
                {
                    'params': param_groups['encoder'],
                    'lr': args.lr_encoder,
                    'name': 'encoder',
                    'lr_scale': lr_scale_encoder
                },
                {
                    'params': param_groups['projection'],
                    'lr': args.lr,
                    'name': 'projection',
                    'lr_scale': 1.0
                },
                {
                    'params': param_groups['domain_projection'],
                    'lr': args.lr,
                    'name': 'domain_projection',
                    'lr_scale': 1.0
                },
                {
                    'params': param_groups['domain_classifier'],
                    'lr': args.lr,
                    'name': 'domain_classifier',
                    'lr_scale': 1.0
                },
                {
                    'params': param_groups['matching'],
                    'lr': args.lr,
                    'name': 'matching',
                    'lr_scale': 1.0
                }
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Discriminator 优化器（独立）
        lr_discriminator = getattr(args, 'lr_discriminator', args.lr)
        optimizer_discriminator = torch.optim.AdamW(
            param_groups['discriminator'],
            lr=lr_discriminator,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 返回字典形式
        optimizer = {
            'encoder': optimizer_encoder,
            'discriminator': optimizer_discriminator
        }
        
        print(f"✅ Adversarial Training Mode:")
        print(f"   - Encoder Optimizer: AdamW (lr={args.lr_encoder}, scale={lr_scale_encoder:.3f})")
        print(f"   - Projection Optimizer: AdamW (contact/domain heads lr={args.lr})")
        print(f"   - Discriminator Optimizer: AdamW (lr={lr_discriminator}) [INDEPENDENT]")
        print(f"   - Weight decay: {args.weight_decay}")
    else:
        # 标准训练模式（Phase-1或兼容模式）
        base_lr = args.lr if args.lr > 0 else 1e-6
        lr_scale_encoder = args.lr_encoder / base_lr if base_lr > 0 else 1.0
        optimizer_params = [
            {
                'params': param_groups['encoder'],
                'lr': args.lr_encoder,
                'name': 'encoder',
                'lr_scale': lr_scale_encoder
            },
            {
                'params': param_groups['projection'],
                'lr': args.lr,
                'name': 'projection',
                'lr_scale': 1.0
            },
                {
                    'params': param_groups['domain_projection'],
                    'lr': args.lr,
                    'name': 'domain_projection',
                    'lr_scale': 1.0
                },
                {
                    'params': param_groups['domain_classifier'],
                    'lr': args.lr,
                    'name': 'domain_classifier',
                    'lr_scale': 1.0
                },
                {
                    'params': param_groups['matching'],
                    'lr': args.lr,
                    'name': 'matching',
                    'lr_scale': 1.0
            },
            {
                'params': param_groups['discriminator'],
                'lr': args.lr,
                'name': 'discriminator',
                'lr_scale': 1.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        print(f"✅ Standard Training Mode:")
        print(f"   - Optimizer: AdamW")
        print(f"   - Encoder LR: {args.lr_encoder} (scale={lr_scale_encoder:.3f})")
        print(f"   - Projection LR (contact/domain): {args.lr}")
        print(f"   - Discriminator LR: {args.lr}")
        print(f"   - Weight decay: {args.weight_decay}")
    
    loss_scaler = NativeScaler()
    
    # ====================================================================
    # 加载 checkpoint (如果需要)
    # ====================================================================
    if args.resume:
        print(f'\n📂 Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_state = model_without_ddp.state_dict()
        filtered_state, mismatched = filter_mismatched_state_dict(model_state, checkpoint['model'])
        filtered_state, dropped_norm_keys = sanitize_projection_norm_transition(
            filtered_state,
            checkpoint['model']
        )
        incompatible = model_without_ddp.load_state_dict(filtered_state, strict=False)
        if misc.is_main_process():
            if mismatched:
                preview = ", ".join([m[0] for m in mismatched[:5]])
                suffix = "..." if len(mismatched) > 5 else ""
                print(f'⚠️  Skipped {len(mismatched)} mismatched keys: {preview}{suffix}')
            if dropped_norm_keys:
                preview = ", ".join(dropped_norm_keys[:5])
                suffix = "..." if len(dropped_norm_keys) > 5 else ""
                print(
                    f'⚠️  BN→LN transition detected, dropped {len(dropped_norm_keys)} norm keys: '
                    f'{preview}{suffix}'
                )
                print('   Using current LayerNorm initialization for projection heads.')
            if incompatible.missing_keys:
                print(f'⚠️  Missing keys after load: {len(incompatible.missing_keys)}')
            if incompatible.unexpected_keys:
                print(f'⚠️  Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)}')
        if not args.resume_model_only and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            # BN->LN 迁移时，跳过优化器状态恢复，避免旧动量污染新LN参数。
            skip_optimizer_resume = len(dropped_norm_keys) > 0
            if skip_optimizer_resume:
                args.start_epoch = checkpoint['epoch'] + 1
                if misc.is_main_process():
                    print(
                        '⚠️  Skip optimizer/scaler resume due to BN→LN transition; '
                        f'keeping start_epoch={args.start_epoch}'
                    )
            else:
                # 🆕 支持字典形式的optimizer（对抗式训练）
                if isinstance(optimizer, dict):
                    if isinstance(checkpoint['optimizer'], dict):
                        # 新格式 → 新格式
                        optimizer['encoder'].load_state_dict(checkpoint['optimizer']['encoder'])
                        optimizer['discriminator'].load_state_dict(checkpoint['optimizer']['discriminator'])
                    else:
                        # 旧格式 → 新格式（只加载encoder部分，discriminator从头训练）
                        print('⚠️  Loading from standard optimizer to adversarial mode (discriminator reset)')
                else:
                    # 标准模式
                    if isinstance(checkpoint['optimizer'], dict):
                        print('⚠️  Loading from adversarial optimizer to standard mode (partial load)')
                    else:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print(f'✅ Resumed optimizer/scaler到 epoch {args.start_epoch}')
        else:
            print('✅ Loaded model weights (optimizer state reset)')
    
    # ====================================================================
    # 初始化 Wandb
    # ====================================================================
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb and misc.is_main_process()
    
    if use_wandb:
        run_name = args.wandb_name or f"stage{args.stage}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[f'stage{args.stage}']
        )
        
        # 定义指标
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        
        print(f'✅ Wandb initialized: {run_name}')
    else:
        print('⚠️  Wandb disabled')
    
    # ====================================================================
    # 训练循环
    # ====================================================================
    print('\n' + '=' * 70)
    print(f'Starting Training - Stage {args.stage}')
    print('=' * 70)
    print(f'Total epochs: {args.epochs}')
    print(f'Start epoch: {args.start_epoch}')
    print(f'Stage: {args.stage}')
    
    # 主指标：默认 same_contact_sim，可切换为 group_loss_gap
    primary_metric = getattr(args, 'primary_metric', 'same_sim')
    metric_higher_is_better = primary_metric != 'group_loss_gap'
    best_metric_value = None
    best_same_sim = -1.0
    best_diff_sim = float('inf')
    best_epoch = -1
    # 记录“真实/未约束”的最佳指标，便于W&B展示
    best_metric_value_any = None
    best_same_sim_any = -1.0
    best_diff_sim_any = None
    best_epoch_any = -1
    # 新增：多目标保存（便于消融与回溯）
    best_semantic_sim = -1.0
    best_semantic_epoch = -1
    best_domain_acc = float('inf')
    best_domain_epoch = -1
    best_pareto_score = None
    best_pareto_epoch = -1
    phase2_domain_acc_bad_epochs = 0
    phase2_accel_bad_epochs = 0
    phase2_conditional_bad_epochs = 0
    phase2_conditional_good_epochs = 0
    
    # Early Stopping设置（优化后的阈值）
    early_stopping_patience = getattr(args, 'early_stopping_patience', 15)
    early_stopping_min_delta = 0.00001  # 修改：从 0.00005 改为 0.00001
    epochs_no_improve = 0
    
    start_time = time.time()

    def _optimizer_state_dict_for_save():
        if isinstance(optimizer, dict):
            return {
                'encoder': optimizer['encoder'].state_dict(),
                'discriminator': optimizer['discriminator'].state_dict()
            }
        return optimizer.state_dict()

    def _save_named_checkpoint(filename, epoch_id, extra_payload):
        if not misc.is_main_process():
            return
        save_path = os.path.join(args.output_dir, filename)
        payload = {
            'model': model_without_ddp.state_dict(),
            'optimizer': _optimizer_state_dict_for_save(),
            'epoch': epoch_id,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
        payload.update(extra_payload)
        torch.save(payload, save_path)

    def _is_finite_scalar(value):
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    # 保存策略阈值
    save_domain_same_floor = float(getattr(args, 'save_domain_same_floor', 0.78))
    save_pareto_same_floor = float(getattr(args, 'save_pareto_same_floor', 0.78))
    save_pareto_probe_ceiling = float(getattr(args, 'save_pareto_probe_ceiling', 0.60))
    # 条件护栏恢复基线（用于“超阈值时增强，恢复后回落”）
    base_lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
    base_lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
    base_lambda_inter = float(getattr(args, 'lambda_inter', 0.0))
    base_lambda_inter_sensor = float(getattr(args, 'lambda_inter_sensor', 0.0))
    base_lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
    base_adv_target = str(getattr(args, 'adversarial_feature_target', 'domain'))
    base_adv_switch = bool(getattr(args, 'phase2_adv_target_switch', False))
    base_adv_switch_ratio = float(getattr(args, 'phase2_adv_target_switch_ratio', 0.5))
    base_adv_target_late = str(getattr(args, 'phase2_adv_target_late', 'contact_residual'))
    base_digit_focus_weight = float(getattr(args, 'digit_focus_weight', 1.0))
    
    for epoch in range(args.start_epoch, args.epochs):
        
        # 更新 lambda_im (Stage-2b 线性增长)
        lambda_im = get_lambda_im_schedule(epoch, args)
        args.lambda_im = lambda_im
        stop_due_to_early = False
        
        print(f'\n{"="*70}')
        print(f'Epoch {epoch}/{args.epochs-1}')
        print(f'{"="*70}')
        print(f'Lambda_im: {lambda_im:.4f}')
        
        # ================================================================
        # 训练
        # ================================================================
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
            wandb_log=use_wandb
        )
        
        # ================================================================
        # 验证
        # ================================================================
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            # 供W&B记录使用，避免val指标错位到上一epoch
            args.current_epoch = epoch
            val_stats = validate(
                model=model,
                data_loader=val_loader,
                device=device,
                args=args,
                wandb_log=use_wandb
            )
            
            # ================================================================
            # 🆕 Probe崩溃检测和Lambda退火机制
            # ================================================================
            # 🔧 FIX: 所有rank都需要执行这段代码，确保args.lambda_im_end同步
            probe_is_collapsed = val_stats.get('probe_is_collapsed', False)
            probe_collapse_ratio = val_stats.get('probe_collapse_ratio', 0.0)
            domain_collapse_ratio = val_stats.get('domain_collapse_ratio', 0.0)
            domain_acc_std = val_stats.get('domain_acc_std', 0.0)
            domain_acc_gelsight = val_stats.get('domain_acc_gelsight', 0.0)
            domain_acc_digit = val_stats.get('domain_acc_digit', 0.0)
            domain_acc_duragel = val_stats.get('domain_acc_duragel', 0.0)
            
            # 🆕 Lambda退火机制（检测到崩溃时降低lambda_im）
            lambda_annealing_enabled = getattr(args, 'lambda_annealing_enabled', True)
            lambda_annealing_factor = getattr(args, 'lambda_annealing_factor', 0.5)
            lambda_annealing_min = getattr(args, 'lambda_annealing_min', 0.01)
            
            if lambda_annealing_enabled and probe_is_collapsed and lambda_im > lambda_annealing_min:
                old_lambda_im_end = args.lambda_im_end
                args.lambda_im_end = max(args.lambda_im_end * lambda_annealing_factor, lambda_annealing_min)
                if misc.is_main_process():
                    print(f'\n🔥 Lambda Annealing Triggered!')
                    print(f'   Probe collapse detected: {probe_collapse_ratio:.2%}')
                    print(f'   Reducing lambda_im_end: {old_lambda_im_end:.4f} → {args.lambda_im_end:.4f}')
                    print(f'   Current lambda_im will be recalculated in next epoch\n')
            
            # 检测严重崩溃（超过50%的batch出现崩溃）
            if misc.is_main_process():
                if domain_collapse_ratio > 0.5:
                    print(f'\n🚨🚨🚨 CRITICAL WARNING: Domain Discriminator Collapse! 🚨🚨🚨')
                    print(f'   Collapse detected in {domain_collapse_ratio*100:.1f}% of validation batches!')
                    print(f'   Per-class acc: GelSight={domain_acc_gelsight*100:.1f}%, DIGIT={domain_acc_digit*100:.1f}%, DuraGel={domain_acc_duragel*100:.1f}%')
                    print(f'   Std={domain_acc_std:.4f} (should be < 0.05)')
                    print(f'   Recommendation: Consider reducing lambda_im or stopping training')
                    print(f'   Training may be producing degenerate representations!\n')
                
                # 检测部分崩溃（10-50%）
                elif domain_collapse_ratio > 0.1:
                    print(f'\n⚠️  WARNING: Potential domain discriminator instability')
                    print(f'   Collapse detected in {domain_collapse_ratio*100:.1f}% of validation batches')
                    print(f'   Per-class acc: GelSight={domain_acc_gelsight*100:.1f}%, DIGIT={domain_acc_digit*100:.1f}%, DuraGel={domain_acc_duragel*100:.1f}%')
                    print(f'   Monitor closely in next epochs\n')

            # 🆕 Phase-2: domain_acc偏高时自适应增强lambda_im_end
            use_adaptive_lambda = (
                args.stage == '2b' and getattr(args, 'use_independent_probe', False)
            )
            if use_adaptive_lambda:
                domain_acc = val_stats.get('domain_acc', 0.333)
                phase2_boost_factor = float(getattr(args, 'phase2_lambda_im_boost', 1.0))
                phase2_lambda_im_max = float(getattr(args, 'phase2_lambda_im_max', args.lambda_im_end))
                phase2_patience = int(getattr(args, 'phase2_domain_acc_patience', 2))
                phase2_acc_threshold = float(getattr(args, 'phase2_probe_acc_threshold', 0.40))
                if (not probe_is_collapsed) and domain_acc > phase2_acc_threshold:
                    phase2_domain_acc_bad_epochs += 1
                else:
                    phase2_domain_acc_bad_epochs = 0
                if (
                    phase2_boost_factor > 1.0 and
                    phase2_patience > 0 and
                    phase2_domain_acc_bad_epochs >= phase2_patience and
                    args.lambda_im_end < phase2_lambda_im_max
                ):
                    old_lambda_im_end = args.lambda_im_end
                    args.lambda_im_end = min(args.lambda_im_end * phase2_boost_factor, phase2_lambda_im_max)
                    phase2_domain_acc_bad_epochs = 0
                    if misc.is_main_process():
                        print(f'\n🚀 Adaptive Lambda Boost Triggered!')
                        print(f'   domain_acc={domain_acc:.4f} > {phase2_acc_threshold:.4f} for {phase2_patience} evals')
                        print(f'   Increasing lambda_im_end: {old_lambda_im_end:.4f} → {args.lambda_im_end:.4f}')
                        print(f'   Current lambda_im will be recalculated in next epoch\n')

            # 🆕 Phase-2: domain_acc偏高时提前进入lambda_im增长阶段
            if use_adaptive_lambda:
                phase2_accel_factor = float(getattr(args, 'phase2_accel_factor', 1.0))
                phase2_warmup_min = float(getattr(args, 'phase2_accel_warmup_min', 0.1))
                phase2_rampup_min = float(getattr(args, 'phase2_accel_rampup_min', 0.2))
                phase2_accel_patience = int(getattr(args, 'phase2_accel_patience', 2))
                current_warmup = float(getattr(args, 'lambda_im_warmup_ratio', 0.3))
                current_rampup = float(getattr(args, 'lambda_im_rampup_ratio', 0.4))
                if (not probe_is_collapsed) and domain_acc > phase2_acc_threshold:
                    phase2_accel_bad_epochs += 1
                else:
                    phase2_accel_bad_epochs = 0
                if (
                    phase2_accel_factor < 1.0 and
                    phase2_accel_patience > 0 and
                    phase2_accel_bad_epochs >= phase2_accel_patience and
                    (current_warmup > phase2_warmup_min or current_rampup > phase2_rampup_min)
                ):
                    old_warmup = current_warmup
                    old_rampup = current_rampup
                    # 只允许缩短 warmup/rampup，禁止“反向抬高”导致lambda_im回落到0。
                    raw_new_warmup = current_warmup * phase2_accel_factor
                    raw_new_rampup = current_rampup * phase2_accel_factor
                    new_warmup = min(current_warmup, max(raw_new_warmup, phase2_warmup_min))
                    new_rampup = min(current_rampup, max(raw_new_rampup, phase2_rampup_min))
                    # 保留至少10%的稳定阶段
                    if new_warmup + new_rampup > 0.9:
                        new_rampup = max(0.9 - new_warmup, phase2_rampup_min)
                    args.lambda_im_warmup_ratio = new_warmup
                    args.lambda_im_rampup_ratio = new_rampup
                    phase2_accel_bad_epochs = 0
                    if misc.is_main_process():
                        print(f'\n⚡ Early Lambda Ramp-Up Triggered!')
                        print(f'   domain_acc={domain_acc:.4f} > {phase2_acc_threshold:.4f} for {phase2_accel_patience} evals')
                        print(f'   Warmup ratio: {old_warmup:.2f} → {new_warmup:.2f}')
                        print(f'   Rampup ratio: {old_rampup:.2f} → {new_rampup:.2f}')
                        print(f'   Current lambda_im will be recalculated in next epoch\n')

            # 🧯 Phase-2语义护栏：防止same_sim先塌
            if use_adaptive_lambda and bool(getattr(args, 'phase2_semantic_guardrail', False)):
                current_same_sim_guard = float(val_stats.get('same_contact_sim', 0.0))
                current_domain_acc_guard = float(val_stats.get('domain_acc', 1.0 / max(getattr(args, 'num_sensors', 3), 1)))
                same_floor = float(getattr(args, 'phase2_same_sim_threshold', 0.85))
                probe_ceiling = float(getattr(args, 'phase2_probe_acc_threshold', 0.40))
                same_margin = float(getattr(args, 'phase2_semantic_margin', 0.01))
                lambda_decay = float(getattr(args, 'phase2_guardrail_lambda_decay', 0.8))
                im_scale_decay = float(getattr(args, 'phase2_guardrail_im_scale_decay', 0.8))
                lambda_min = float(getattr(args, 'phase2_guardrail_lambda_min', 0.10))
                im_scale_min = float(getattr(args, 'phase2_guardrail_im_scale_min', 1.0))
                pair_step = float(getattr(args, 'phase2_guardrail_pair_step', 0.02))
                pair_max = float(getattr(args, 'phase2_guardrail_pair_max', 0.20))
                pair_tail_step_sem = float(
                    getattr(args, 'phase2_guardrail_pair_tail_step', max(pair_step * 0.5, 0.01))
                )
                pair_tail_max_sem = float(
                    getattr(args, 'phase2_guardrail_pair_tail_max', max(base_lambda_pair_tail, 0.18))
                )
                residual_compact_step_sem = float(
                    getattr(args, 'phase2_guardrail_residual_compact_step', max(pair_step * 0.5, 0.01))
                )
                residual_compact_max_sem = float(
                    getattr(
                        args,
                        'phase2_guardrail_residual_compact_max',
                        max(base_lambda_residual_compact, 0.16)
                    )
                )
                domain_boost = float(getattr(args, 'phase2_guardrail_domain_boost', 1.05))
                lambda_cap = float(getattr(args, 'phase2_lambda_im_max', args.lambda_im_end))

                if current_same_sim_guard < same_floor:
                    old_lambda_end = float(args.lambda_im_end)
                    old_im_scale = float(getattr(args, 'im_loss_scale', 1.0))
                    old_lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
                    old_lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
                    old_lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
                    args.lambda_im_end = max(old_lambda_end * lambda_decay, lambda_min)
                    args.im_loss_scale = max(old_im_scale * im_scale_decay, im_scale_min)
                    args.lambda_pair = min(old_lambda_pair + pair_step, pair_max)
                    args.lambda_pair_tail = min(
                        old_lambda_pair_tail + pair_tail_step_sem,
                        pair_tail_max_sem
                    )
                    args.lambda_residual_compact = min(
                        old_lambda_residual_compact + residual_compact_step_sem,
                        residual_compact_max_sem
                    )
                    if misc.is_main_process():
                        print(f'\n🧯 Semantic Guardrail Triggered (same_sim={current_same_sim_guard:.4f} < {same_floor:.4f})')
                        print(f'   lambda_im_end: {old_lambda_end:.4f} → {args.lambda_im_end:.4f}')
                        print(f'   im_loss_scale: {old_im_scale:.4f} → {args.im_loss_scale:.4f}')
                        print(f'   lambda_pair: {old_lambda_pair:.4f} → {args.lambda_pair:.4f}')
                        print(
                            f'   lambda_pair_tail: '
                            f'{old_lambda_pair_tail:.4f} → {args.lambda_pair_tail:.4f}'
                        )
                        print(
                            f'   lambda_residual_compact: '
                            f'{old_lambda_residual_compact:.4f} → '
                            f'{args.lambda_residual_compact:.4f}'
                        )
                elif (
                    current_domain_acc_guard > probe_ceiling and
                    current_same_sim_guard >= same_floor + same_margin and
                    domain_boost > 1.0
                ):
                    old_lambda_end = float(args.lambda_im_end)
                    args.lambda_im_end = min(old_lambda_end * domain_boost, lambda_cap)
                    if misc.is_main_process() and args.lambda_im_end > old_lambda_end + 1e-8:
                        print(
                            f'\n🧪 Guardrail Domain Boost: '
                            f'domain_acc={current_domain_acc_guard:.4f}, same_sim={current_same_sim_guard:.4f}'
                        )
                        print(f'   lambda_im_end: {old_lambda_end:.4f} → {args.lambda_im_end:.4f}')

                # 🧪 Phase-2 条件指标护栏：直接盯住“同接触内传感器残留”和“pair团内距离”
                # 超阈值：增强团内/团间约束，并优先切到 contact_residual 对抗目标；
                # 恢复后：温和回落到base配置，避免长期过度约束。
                use_conditional_guardrail = bool(getattr(args, 'phase2_use_conditional_constraints', False))
                use_global_mix_guardrail = bool(getattr(args, 'phase2_use_global_mix_constraints', False))
                cond_probe_threshold = float(getattr(args, 'phase2_conditional_probe_threshold', 0.42))
                cond_probe_sensor_threshold = float(
                    getattr(args, 'phase2_conditional_probe_sensor_threshold', cond_probe_threshold)
                )
                cond_probe_threshold_gelsight = float(
                    getattr(args, 'phase2_conditional_probe_gelsight_threshold', -1.0)
                )
                if cond_probe_threshold_gelsight < 0:
                    cond_probe_threshold_gelsight = cond_probe_sensor_threshold
                cond_probe_threshold_digit = float(
                    getattr(args, 'phase2_conditional_probe_digit_threshold', -1.0)
                )
                if cond_probe_threshold_digit < 0:
                    cond_probe_threshold_digit = cond_probe_sensor_threshold
                cond_probe_threshold_duragel = float(
                    getattr(args, 'phase2_conditional_probe_duragel_threshold', -1.0)
                )
                if cond_probe_threshold_duragel < 0:
                    cond_probe_threshold_duragel = cond_probe_sensor_threshold
                pair_link_p90_threshold = float(getattr(args, 'phase2_pair_link_p90_threshold', 0.35))
                sensor_centroid_dist_threshold = float(
                    getattr(args, 'phase2_sensor_centroid_dist_threshold', 0.45)
                )
                knn_mix_rate_threshold = float(
                    getattr(args, 'phase2_knn_mix_rate_threshold', 0.55)
                )
                cond_probe_residual = val_stats.get('conditional_probe_acc_residual', None)
                cond_probe_residual_gelsight = val_stats.get('conditional_probe_acc_residual_gelsight', None)
                cond_probe_residual_digit = val_stats.get('conditional_probe_acc_residual_digit', None)
                cond_probe_residual_duragel = val_stats.get('conditional_probe_acc_residual_duragel', None)
                pair_link_len_p90 = val_stats.get('pair_link_len_p90', None)
                sensor_centroid_dist = val_stats.get('sensor_centroid_dist', None)
                cross_sensor_knn_mix_rate = val_stats.get('cross_sensor_knn_mix_rate', None)
                inter_step = float(getattr(args, 'phase2_guardrail_inter_step', max(pair_step, 0.01)))
                inter_max = float(getattr(args, 'phase2_guardrail_inter_max', max(base_lambda_inter, 0.14)))
                inter_sensor_step = float(
                    getattr(args, 'phase2_guardrail_inter_sensor_step', max(inter_step, 0.01))
                )
                inter_sensor_max = float(
                    getattr(
                        args,
                        'phase2_guardrail_inter_sensor_max',
                        max(base_lambda_inter_sensor, 0.16)
                    )
                )
                pair_tail_step = float(
                    getattr(args, 'phase2_guardrail_pair_tail_step', max(pair_step * 0.5, 0.01))
                )
                pair_tail_max = float(
                    getattr(args, 'phase2_guardrail_pair_tail_max', max(base_lambda_pair_tail, 0.18))
                )
                residual_compact_step = float(
                    getattr(args, 'phase2_guardrail_residual_compact_step', max(inter_step * 0.5, 0.01))
                )
                residual_compact_max = float(
                    getattr(
                        args,
                        'phase2_guardrail_residual_compact_max',
                        max(base_lambda_residual_compact, 0.16)
                    )
                )
                conditional_restore_decay = float(
                    getattr(args, 'phase2_guardrail_conditional_restore_decay', 0.95)
                )
                conditional_escalate_patience = int(
                    getattr(args, 'phase2_guardrail_escalate_patience', 3)
                )
                conditional_restore_patience = int(
                    getattr(args, 'phase2_guardrail_restore_patience', 3)
                )
                guardrail_im_scale_step = float(
                    getattr(args, 'phase2_guardrail_im_scale_step', 0.5)
                )
                guardrail_im_scale_cap = float(
                    getattr(args, 'phase2_guardrail_im_scale_cap', 6.0)
                )
                guardrail_lambda_boost = float(
                    getattr(args, 'phase2_guardrail_lambda_boost', 1.08)
                )
                guardrail_lambda_cap = float(
                    getattr(args, 'phase2_guardrail_lambda_cap', lambda_cap)
                )
                cond_valid = _is_finite_scalar(cond_probe_residual) if use_conditional_guardrail else True
                cond_sensor_valid = (
                    _is_finite_scalar(cond_probe_residual_gelsight) and
                    _is_finite_scalar(cond_probe_residual_digit) and
                    _is_finite_scalar(cond_probe_residual_duragel)
                ) if use_conditional_guardrail else True
                pair_valid = _is_finite_scalar(pair_link_len_p90) if use_conditional_guardrail else True
                cond_missing = not cond_valid
                cond_sensor_missing = not cond_sensor_valid
                pair_missing = not pair_valid
                cond_violation = (
                    cond_missing or (float(cond_probe_residual) > cond_probe_threshold)
                ) if use_conditional_guardrail else False
                cond_sensor_violation = (
                    cond_sensor_missing or
                    float(cond_probe_residual_gelsight) > cond_probe_threshold_gelsight or
                    float(cond_probe_residual_digit) > cond_probe_threshold_digit or
                    float(cond_probe_residual_duragel) > cond_probe_threshold_duragel
                ) if use_conditional_guardrail else False
                pair_violation = (
                    pair_missing or (float(pair_link_len_p90) > pair_link_p90_threshold)
                ) if use_conditional_guardrail else False
                centroid_valid = _is_finite_scalar(sensor_centroid_dist) if use_global_mix_guardrail else True
                knn_valid = _is_finite_scalar(cross_sensor_knn_mix_rate) if use_global_mix_guardrail else True
                centroid_missing = not centroid_valid
                knn_missing = not knn_valid
                global_mix_violation = (
                    centroid_missing or
                    knn_missing or
                    float(sensor_centroid_dist) > sensor_centroid_dist_threshold or
                    float(cross_sensor_knn_mix_rate) < knn_mix_rate_threshold
                ) if use_global_mix_guardrail else False
                any_guardrail_enabled = use_conditional_guardrail or use_global_mix_guardrail
                any_violation = (
                    (use_conditional_guardrail and (cond_violation or cond_sensor_violation or pair_violation)) or
                    (use_global_mix_guardrail and global_mix_violation)
                )

                if any_guardrail_enabled and any_violation:
                    phase2_conditional_bad_epochs += 1
                    phase2_conditional_good_epochs = 0
                    old_lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
                    old_lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
                    old_lambda_inter = float(getattr(args, 'lambda_inter', 0.0))
                    old_lambda_inter_sensor = float(getattr(args, 'lambda_inter_sensor', 0.0))
                    old_lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
                    old_adv_target = str(getattr(args, 'adversarial_feature_target', 'domain'))
                    old_adv_switch = bool(getattr(args, 'phase2_adv_target_switch', False))
                    old_im_scale = float(getattr(args, 'im_loss_scale', 1.0))
                    old_lambda_end = float(getattr(args, 'lambda_im_end', 0.0))

                    args.lambda_pair = min(old_lambda_pair + pair_step, pair_max)
                    args.lambda_pair_tail = min(old_lambda_pair_tail + pair_tail_step, pair_tail_max)
                    args.lambda_inter = min(old_lambda_inter + inter_step, inter_max)
                    args.lambda_inter_sensor = min(
                        old_lambda_inter_sensor + inter_sensor_step,
                        inter_sensor_max
                    )
                    args.lambda_residual_compact = min(
                        old_lambda_residual_compact + residual_compact_step,
                        residual_compact_max
                    )
                    args.adversarial_feature_target = 'contact_residual'
                    args.phase2_adv_target_switch = False
                    args.phase2_adv_target_late = 'contact_residual'

                    pair_saturated = args.lambda_pair >= (pair_max - 1e-8)
                    pair_tail_saturated = args.lambda_pair_tail >= (pair_tail_max - 1e-8)
                    inter_saturated = args.lambda_inter >= (inter_max - 1e-8)
                    inter_sensor_saturated = args.lambda_inter_sensor >= (inter_sensor_max - 1e-8)
                    residual_compact_saturated = (
                        args.lambda_residual_compact >= (residual_compact_max - 1e-8)
                    )
                    should_escalate = (
                        pair_saturated and
                        pair_tail_saturated and
                        inter_saturated and
                        inter_sensor_saturated and
                        residual_compact_saturated and
                        phase2_conditional_bad_epochs >= max(1, conditional_escalate_patience)
                    )
                    if should_escalate:
                        args.im_loss_scale = min(
                            old_im_scale + guardrail_im_scale_step,
                            guardrail_im_scale_cap
                        )
                        args.lambda_im_end = min(
                            old_lambda_end * guardrail_lambda_boost,
                            min(guardrail_lambda_cap, lambda_cap)
                        )

                    if misc.is_main_process():
                        print('\n🎯 Constraint Guardrail Triggered!')
                        if use_conditional_guardrail:
                            print(
                                f'   conditional_probe_acc_residual: {cond_probe_residual} '
                                f'(threshold={cond_probe_threshold:.4f})'
                            )
                            if cond_missing:
                                print(
                                    '   ⚠️ conditional_probe_acc_residual is missing/invalid, '
                                    'treat as violation'
                                )
                            print(
                                "   conditional residual per-sensor: "
                                f"GelSight={cond_probe_residual_gelsight} (th={cond_probe_threshold_gelsight:.4f}), "
                                f"DIGIT={cond_probe_residual_digit} (th={cond_probe_threshold_digit:.4f}), "
                                f"DuraGel={cond_probe_residual_duragel} (th={cond_probe_threshold_duragel:.4f})"
                            )
                            if cond_sensor_missing:
                                print(
                                    '   ⚠️ conditional residual per-sensor is missing/invalid, '
                                    'treat as violation'
                                )
                            print(
                                f'   pair_link_len_p90: {pair_link_len_p90} '
                                f'(threshold={pair_link_p90_threshold:.4f})'
                            )
                            if pair_missing:
                                print(
                                    '   ⚠️ pair_link_len_p90 is missing/invalid, '
                                    'treat as violation'
                                )
                        if use_global_mix_guardrail:
                            print(
                                f'   sensor_centroid_dist: {sensor_centroid_dist} '
                                f'(threshold<={sensor_centroid_dist_threshold:.4f})'
                            )
                            if centroid_missing:
                                print(
                                    '   ⚠️ sensor_centroid_dist is missing/invalid, '
                                    'treat as violation'
                                )
                            print(
                                f'   cross_sensor_knn_mix_rate: {cross_sensor_knn_mix_rate} '
                                f'(threshold>={knn_mix_rate_threshold:.4f})'
                            )
                            if knn_missing:
                                print(
                                    '   ⚠️ cross_sensor_knn_mix_rate is missing/invalid, '
                                    'treat as violation'
                                )
                        print(f'   lambda_pair: {old_lambda_pair:.4f} → {args.lambda_pair:.4f}')
                        print(
                            f'   lambda_pair_tail: '
                            f'{old_lambda_pair_tail:.4f} → {args.lambda_pair_tail:.4f}'
                        )
                        print(f'   lambda_inter: {old_lambda_inter:.4f} → {args.lambda_inter:.4f}')
                        print(
                            f'   lambda_residual_compact: '
                            f'{old_lambda_residual_compact:.4f} → {args.lambda_residual_compact:.4f}'
                        )
                        print(
                            f'   lambda_inter_sensor: '
                            f'{old_lambda_inter_sensor:.4f} → {args.lambda_inter_sensor:.4f}'
                        )
                        print(f'   adversarial_target: {old_adv_target} → {args.adversarial_feature_target}')
                        print(f'   phase2_adv_target_switch: {old_adv_switch} → {args.phase2_adv_target_switch}')
                        print(
                            f'   bad_epochs={phase2_conditional_bad_epochs}, '
                            f'saturated(pair={pair_saturated}, pair_tail={pair_tail_saturated}, '
                            f'inter={inter_saturated}, inter_sensor={inter_sensor_saturated}, '
                            f'residual_compact={residual_compact_saturated})'
                        )
                        if should_escalate:
                            print(
                                '   🚀 Escalate after saturation: '
                                f'im_loss_scale {old_im_scale:.4f} → {args.im_loss_scale:.4f}, '
                                f'lambda_im_end {old_lambda_end:.4f} → {args.lambda_im_end:.4f}'
                            )
                elif any_guardrail_enabled:
                    phase2_conditional_good_epochs += 1
                    phase2_conditional_bad_epochs = 0
                    if phase2_conditional_good_epochs >= max(1, conditional_restore_patience):
                        old_lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
                        old_lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
                        old_lambda_inter = float(getattr(args, 'lambda_inter', 0.0))
                        old_lambda_inter_sensor = float(getattr(args, 'lambda_inter_sensor', 0.0))
                        old_lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
                        old_adv_target = str(getattr(args, 'adversarial_feature_target', 'domain'))
                        old_adv_switch = bool(getattr(args, 'phase2_adv_target_switch', False))
                        old_adv_late = str(getattr(args, 'phase2_adv_target_late', 'contact_residual'))
                        restored_lambda_pair = max(base_lambda_pair, old_lambda_pair * conditional_restore_decay)
                        restored_lambda_pair_tail = max(
                            base_lambda_pair_tail,
                            old_lambda_pair_tail * conditional_restore_decay
                        )
                        restored_lambda_inter = max(base_lambda_inter, old_lambda_inter * conditional_restore_decay)
                        restored_lambda_inter_sensor = max(
                            base_lambda_inter_sensor,
                            old_lambda_inter_sensor * conditional_restore_decay
                        )
                        restored_lambda_residual_compact = max(
                            base_lambda_residual_compact,
                            old_lambda_residual_compact * conditional_restore_decay
                        )
                        args.lambda_pair = restored_lambda_pair
                        args.lambda_pair_tail = restored_lambda_pair_tail
                        args.lambda_inter = restored_lambda_inter
                        args.lambda_inter_sensor = restored_lambda_inter_sensor
                        args.lambda_residual_compact = restored_lambda_residual_compact
                        args.adversarial_feature_target = base_adv_target
                        args.phase2_adv_target_switch = base_adv_switch
                        args.phase2_adv_target_switch_ratio = base_adv_switch_ratio
                        args.phase2_adv_target_late = base_adv_target_late

                        if misc.is_main_process():
                            weight_changed = (
                                abs(restored_lambda_pair - old_lambda_pair) > 1e-8 or
                                abs(restored_lambda_pair_tail - old_lambda_pair_tail) > 1e-8 or
                                abs(restored_lambda_inter - old_lambda_inter) > 1e-8 or
                                abs(restored_lambda_inter_sensor - old_lambda_inter_sensor) > 1e-8 or
                                abs(restored_lambda_residual_compact - old_lambda_residual_compact) > 1e-8
                            )
                            target_changed = (
                                old_adv_target != args.adversarial_feature_target or
                                old_adv_switch != args.phase2_adv_target_switch or
                                old_adv_late != args.phase2_adv_target_late
                            )
                            if weight_changed or target_changed:
                                print('\n🔁 Conditional Guardrail Restore:')
                                print(f'   lambda_pair: {old_lambda_pair:.4f} → {restored_lambda_pair:.4f}')
                                print(
                                    f'   lambda_pair_tail: '
                                    f'{old_lambda_pair_tail:.4f} → {restored_lambda_pair_tail:.4f}'
                                )
                                print(f'   lambda_inter: {old_lambda_inter:.4f} → {restored_lambda_inter:.4f}')
                                print(
                                    f'   lambda_inter_sensor: '
                                    f'{old_lambda_inter_sensor:.4f} → {restored_lambda_inter_sensor:.4f}'
                                )
                                print(
                                    f'   lambda_residual_compact: '
                                    f'{old_lambda_residual_compact:.4f} → '
                                    f'{restored_lambda_residual_compact:.4f}'
                                )
                                print(
                                    f'   adversarial_target restored: {args.adversarial_feature_target}, '
                                    f'switch={args.phase2_adv_target_switch}, '
                                    f'late={args.phase2_adv_target_late}'
                                )
                    elif misc.is_main_process():
                        print(
                            '\n⏳ Conditional guardrail hold: '
                            f'good_epochs={phase2_conditional_good_epochs}/'
                            f'{max(1, conditional_restore_patience)} before restore'
                        )

                if bool(getattr(args, 'use_digit_focus_reweight', False)):
                    digit_margin = float(getattr(args, 'digit_focus_margin', 0.08))
                    digit_step = float(getattr(args, 'digit_focus_step', 0.10))
                    digit_max = float(getattr(args, 'digit_focus_max', 1.8))
                    digit_decay = float(getattr(args, 'digit_focus_decay', 0.97))
                    old_digit_focus_weight = float(getattr(args, 'digit_focus_weight', 1.0))
                    max_other_sensor_acc = max(float(domain_acc_gelsight), float(domain_acc_duragel))
                    digit_residual_higher = float(domain_acc_digit) > (max_other_sensor_acc + digit_margin)
                    digit_cond_residual_higher = (
                        _is_finite_scalar(cond_probe_residual_digit) and
                        float(cond_probe_residual_digit) > (cond_probe_threshold_digit + digit_margin * 0.5)
                    )
                    if digit_residual_higher or digit_cond_residual_higher:
                        args.digit_focus_weight = min(
                            old_digit_focus_weight + digit_step,
                            digit_max
                        )
                    else:
                        args.digit_focus_weight = max(
                            base_digit_focus_weight,
                            old_digit_focus_weight * digit_decay
                        )
                    if (
                        misc.is_main_process() and
                        abs(args.digit_focus_weight - old_digit_focus_weight) > 1e-8
                    ):
                        print(
                            '\n🎯 DIGIT Focus Reweight Update: '
                            f'{old_digit_focus_weight:.3f} → {args.digit_focus_weight:.3f} '
                            f'(digit_acc={float(domain_acc_digit):.4f}, '
                            f'others_max={max_other_sensor_acc:.4f}, '
                            f'cond_digit={cond_probe_residual_digit})'
                        )

            if misc.is_dist_avail_and_initialized():
                sync_state = [
                    args.lambda_im_end,
                    phase2_domain_acc_bad_epochs,
                    args.lambda_im_warmup_ratio,
                    args.lambda_im_rampup_ratio,
                    phase2_accel_bad_epochs,
                    phase2_conditional_bad_epochs,
                    phase2_conditional_good_epochs,
                    float(getattr(args, 'im_loss_scale', 1.0)),
                    float(getattr(args, 'lambda_pair', 0.0)),
                    float(getattr(args, 'lambda_pair_tail', 0.0)),
                    float(getattr(args, 'lambda_inter', 0.0)),
                    float(getattr(args, 'lambda_inter_sensor', 0.0)),
                    float(getattr(args, 'lambda_residual_compact', 0.0)),
                    float(getattr(args, 'digit_focus_weight', 1.0)),
                    str(getattr(args, 'adversarial_feature_target', 'domain')),
                    bool(getattr(args, 'phase2_adv_target_switch', False)),
                    float(getattr(args, 'phase2_adv_target_switch_ratio', 0.5)),
                    str(getattr(args, 'phase2_adv_target_late', 'contact_residual')),
                ]
                dist.broadcast_object_list(sync_state, src=0)
                (args.lambda_im_end,
                 phase2_domain_acc_bad_epochs,
                 args.lambda_im_warmup_ratio,
                 args.lambda_im_rampup_ratio,
                 phase2_accel_bad_epochs,
                 phase2_conditional_bad_epochs,
                 phase2_conditional_good_epochs,
                 args.im_loss_scale,
                 args.lambda_pair,
                 args.lambda_pair_tail,
                 args.lambda_inter,
                 args.lambda_inter_sensor,
                 args.lambda_residual_compact,
                 args.digit_focus_weight,
                 args.adversarial_feature_target,
                 args.phase2_adv_target_switch,
                 args.phase2_adv_target_switch_ratio,
                 args.phase2_adv_target_late) = sync_state
            
            # 更新最佳模型（主指标可配置）
            current_same_sim = val_stats['same_contact_sim']
            current_diff_sim = val_stats['diff_contact_sim']
            current_group_gap = val_stats.get('group_loss_gap')
            
            # 根据primary_metric计算指标值
            if primary_metric == 'same_sim':
                metric_value = current_same_sim
                metric_name_for_log = 'same_contact_sim'
                local_metric_direction = True  # 越高越好
            elif primary_metric == 'group_loss_gap':
                metric_value = current_group_gap
                metric_name_for_log = 'group_loss_gap'
                local_metric_direction = False  # 越低越好
            elif primary_metric == 'phase1_combined':
                metric_value = compute_combined_metric(val_stats, 'phase1_combined', args=args)
                metric_name_for_log = 'phase1_combined'
                local_metric_direction = True  # 越高越好
            elif primary_metric == 'phase2_combined':
                metric_value = compute_combined_metric(val_stats, 'phase2_combined', args=args)
                metric_name_for_log = 'phase2_combined'
                local_metric_direction = True  # 越高越好
            else:
                raise ValueError(f"Unknown primary_metric: {primary_metric}")

            if metric_value is None:
                if primary_metric == 'group_loss_gap':
                    # 若未返回group loss gap，则回退到same_sim
                    if misc.is_main_process():
                        print('⚠️  primary_metric=group_loss_gap 但验证结果中无该指标，回退为 same_sim')
                    primary_metric = 'same_sim'
                    metric_higher_is_better = True
                    best_metric_value = None
                    metric_value = current_same_sim
                    metric_name_for_log = 'same_contact_sim'
                    local_metric_direction = True
                else:
                    # 无法评估主指标，视为无改进
                    metric_value = None

            # 记录未约束的“真实最佳值”（不影响保存/早停逻辑）
            is_best_any = False
            if metric_value is not None:
                if best_metric_value_any is None:
                    best_metric_value_any = metric_value
                    best_epoch_any = epoch
                    is_best_any = True
                elif local_metric_direction and metric_value > best_metric_value_any + early_stopping_min_delta:
                    best_metric_value_any = metric_value
                    best_epoch_any = epoch
                    is_best_any = True
                elif (not local_metric_direction) and metric_value < best_metric_value_any - early_stopping_min_delta:
                    best_metric_value_any = metric_value
                    best_epoch_any = epoch
                    is_best_any = True
            if current_same_sim > best_same_sim_any:
                best_same_sim_any = current_same_sim
            if best_diff_sim_any is None or current_diff_sim < best_diff_sim_any:
                best_diff_sim_any = current_diff_sim

            # 始终保存“未约束最佳”快照，便于分析训练真实趋势
            if is_best_any and misc.is_main_process():
                _save_named_checkpoint('best_any.pth', epoch, {
                    'args': args,
                    'val_same_sim': current_same_sim,
                    'val_diff_sim': current_diff_sim,
                    'primary_metric': primary_metric,
                    'primary_metric_value': best_metric_value_any
                })
                print(f'📌 Unconstrained best saved: {os.path.join(args.output_dir, "best_any.pth")}')
                print(f'   {metric_name_for_log}: {best_metric_value_any:.4f} (epoch {best_epoch_any})')
            
            # Phase-2 目标约束（仅在2b + 独立probe评估时启用）
            enforce_phase2_targets = (
                args.stage == '2b' and getattr(args, 'use_independent_probe', False)
            )
            if enforce_phase2_targets:
                same_sim_threshold = getattr(args, 'phase2_same_sim_threshold', 0.85)
                probe_acc_threshold = getattr(args, 'phase2_probe_acc_threshold', 0.40)
            else:
                same_sim_threshold = getattr(args, 'same_sim_threshold', 0.70)
                probe_acc_threshold = None
            diff_sim_threshold = getattr(args, 'diff_sim_threshold', 0.35)
            use_conditional_constraints = (
                enforce_phase2_targets and
                bool(getattr(args, 'phase2_use_conditional_constraints', False))
            )
            use_global_mix_constraints = (
                enforce_phase2_targets and
                bool(getattr(args, 'phase2_use_global_mix_constraints', False))
            )
            cond_probe_threshold = float(
                getattr(args, 'phase2_conditional_probe_threshold', probe_acc_threshold or 0.45)
            )
            cond_probe_sensor_threshold = float(
                getattr(args, 'phase2_conditional_probe_sensor_threshold', cond_probe_threshold)
            )
            cond_probe_threshold_gelsight = float(
                getattr(args, 'phase2_conditional_probe_gelsight_threshold', -1.0)
            )
            if cond_probe_threshold_gelsight < 0:
                cond_probe_threshold_gelsight = cond_probe_sensor_threshold
            cond_probe_threshold_digit = float(
                getattr(args, 'phase2_conditional_probe_digit_threshold', -1.0)
            )
            if cond_probe_threshold_digit < 0:
                cond_probe_threshold_digit = cond_probe_sensor_threshold
            cond_probe_threshold_duragel = float(
                getattr(args, 'phase2_conditional_probe_duragel_threshold', -1.0)
            )
            if cond_probe_threshold_duragel < 0:
                cond_probe_threshold_duragel = cond_probe_sensor_threshold
            pair_link_p90_threshold = float(
                getattr(args, 'phase2_pair_link_p90_threshold', 0.35)
            )
            sensor_centroid_dist_threshold = float(
                getattr(args, 'phase2_sensor_centroid_dist_threshold', 0.45)
            )
            knn_mix_rate_threshold = float(
                getattr(args, 'phase2_knn_mix_rate_threshold', 0.55)
            )

            domain_acc = val_stats.get('domain_acc', 0.333)
            conditional_probe_acc_residual = val_stats.get('conditional_probe_acc_residual', None)
            conditional_probe_acc_residual_gelsight = val_stats.get('conditional_probe_acc_residual_gelsight', None)
            conditional_probe_acc_residual_digit = val_stats.get('conditional_probe_acc_residual_digit', None)
            conditional_probe_acc_residual_duragel = val_stats.get('conditional_probe_acc_residual_duragel', None)
            pair_link_len_p90 = val_stats.get('pair_link_len_p90', None)
            sensor_centroid_dist = val_stats.get('sensor_centroid_dist', None)
            cross_sensor_knn_mix_rate = val_stats.get('cross_sensor_knn_mix_rate', None)
            conditional_constraints_passed = True
            if use_conditional_constraints:
                conditional_sensor_constraints_passed = (
                    conditional_probe_acc_residual_gelsight is not None and
                    conditional_probe_acc_residual_digit is not None and
                    conditional_probe_acc_residual_duragel is not None and
                    conditional_probe_acc_residual_gelsight <= cond_probe_threshold_gelsight and
                    conditional_probe_acc_residual_digit <= cond_probe_threshold_digit and
                    conditional_probe_acc_residual_duragel <= cond_probe_threshold_duragel
                )
                conditional_constraints_passed = (
                    conditional_probe_acc_residual is not None and
                    pair_link_len_p90 is not None and
                    conditional_probe_acc_residual <= cond_probe_threshold and
                    pair_link_len_p90 <= pair_link_p90_threshold and
                    conditional_sensor_constraints_passed
                )
            global_mix_constraints_passed = True
            if use_global_mix_constraints:
                global_mix_constraints_passed = (
                    sensor_centroid_dist is not None and
                    cross_sensor_knn_mix_rate is not None and
                    _is_finite_scalar(sensor_centroid_dist) and
                    _is_finite_scalar(cross_sensor_knn_mix_rate) and
                    sensor_centroid_dist <= sensor_centroid_dist_threshold and
                    cross_sensor_knn_mix_rate >= knn_mix_rate_threshold
                )
            constraints_passed = (
                metric_value is not None and
                current_diff_sim < diff_sim_threshold and
                current_same_sim > same_sim_threshold and
                (probe_acc_threshold is None or domain_acc <= probe_acc_threshold) and
                conditional_constraints_passed and
                global_mix_constraints_passed
            )

            # ============================================================
            # 多目标保存：best_semantic / best_domain / best_pareto
            # ============================================================
            semantic_ready = current_diff_sim < diff_sim_threshold
            if semantic_ready and current_same_sim > best_semantic_sim + early_stopping_min_delta:
                best_semantic_sim = current_same_sim
                best_semantic_epoch = epoch
                _save_named_checkpoint('best_semantic.pth', epoch, {
                    'val_same_sim': current_same_sim,
                    'val_diff_sim': current_diff_sim,
                    'val_domain_acc': domain_acc,
                    'save_criterion': 'semantic'
                })
                if misc.is_main_process():
                    print(f'✅ Semantic best saved: {os.path.join(args.output_dir, "best_semantic.pth")}')
                    print(f'   same_sim: {current_same_sim:.4f}, diff_sim: {current_diff_sim:.4f}')

            domain_ready = (
                current_same_sim >= save_domain_same_floor and
                current_diff_sim < diff_sim_threshold
            )
            if domain_ready and domain_acc < best_domain_acc - early_stopping_min_delta:
                best_domain_acc = domain_acc
                best_domain_epoch = epoch
                _save_named_checkpoint('best_domain.pth', epoch, {
                    'val_same_sim': current_same_sim,
                    'val_diff_sim': current_diff_sim,
                    'val_domain_acc': domain_acc,
                    'save_criterion': 'domain'
                })
                if misc.is_main_process():
                    print(f'✅ Domain best saved: {os.path.join(args.output_dir, "best_domain.pth")}')
                    print(f'   domain_acc: {domain_acc:.4f}, same_sim: {current_same_sim:.4f}')

            pareto_score = compute_combined_metric(
                val_stats,
                'phase2_combined' if args.stage == '2b' else 'phase1_combined',
                args=args
            )
            pareto_ready = (
                pareto_score is not None and
                current_same_sim >= save_pareto_same_floor and
                current_diff_sim < diff_sim_threshold and
                domain_acc <= save_pareto_probe_ceiling
            )
            if pareto_ready and (
                best_pareto_score is None or
                pareto_score > best_pareto_score + early_stopping_min_delta
            ):
                best_pareto_score = pareto_score
                best_pareto_epoch = epoch
                _save_named_checkpoint('best_pareto.pth', epoch, {
                    'val_same_sim': current_same_sim,
                    'val_diff_sim': current_diff_sim,
                    'val_domain_acc': domain_acc,
                    'pareto_score': pareto_score,
                    'save_criterion': 'pareto'
                })
                if misc.is_main_process():
                    print(f'✅ Pareto best saved: {os.path.join(args.output_dir, "best_pareto.pth")}')
                    print(f'   pareto_score: {pareto_score:.4f} | same: {current_same_sim:.4f} | domain: {domain_acc:.4f}')
            
            is_best = False
            improvement_msg = ''
            if constraints_passed:
                delta = 0.0
                if best_metric_value is None:
                    improved = True
                elif local_metric_direction:
                    improved = metric_value > best_metric_value + early_stopping_min_delta
                    delta = metric_value - best_metric_value
                else:
                    improved = best_metric_value - metric_value > early_stopping_min_delta
                    delta = best_metric_value - metric_value
                if improved:
                    is_best = True
                    prev_str = f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A"
                    if best_metric_value is None:
                        change_str = "(init best)"
                    else:
                        sign = '+' if local_metric_direction else '-'
                        change_str = f"({sign}{abs(delta):.4f})"
                    improvement_msg = f"{metric_name_for_log}: {prev_str} → {metric_value:.4f} {change_str}"
            
            if is_best:
                best_metric_value = metric_value
                best_same_sim = current_same_sim
                best_diff_sim = current_diff_sim
                best_epoch = epoch
                epochs_no_improve = 0
                
                if misc.is_main_process():
                    _save_named_checkpoint('best.pth', epoch, {
                        'args': args,
                        'val_same_sim': best_same_sim,
                        'val_diff_sim': best_diff_sim,
                        'val_domain_acc': domain_acc,
                        'val_conditional_probe_acc_residual': conditional_probe_acc_residual,
                        'val_conditional_probe_acc_residual_gelsight': conditional_probe_acc_residual_gelsight,
                        'val_conditional_probe_acc_residual_digit': conditional_probe_acc_residual_digit,
                        'val_conditional_probe_acc_residual_duragel': conditional_probe_acc_residual_duragel,
                        'val_pair_link_len_p90': pair_link_len_p90,
                        'val_sensor_centroid_dist': sensor_centroid_dist,
                        'val_cross_sensor_knn_mix_rate': cross_sensor_knn_mix_rate,
                        'primary_metric': primary_metric,
                        'primary_metric_value': best_metric_value
                    })
                    print(f'✅ Best model saved: {os.path.join(args.output_dir, "best.pth")}')
                    print(f'   {improvement_msg}')
                    print(f'   same_sim: {current_same_sim:.4f} (> {same_sim_threshold})')
                    print(f'   diff_sim: {current_diff_sim:.4f} (< {diff_sim_threshold})')
                    if probe_acc_threshold is not None:
                        print(f'   domain_acc: {domain_acc:.4f} (<= {probe_acc_threshold})')
                    if use_conditional_constraints:
                        print(
                            f'   conditional_probe_acc_residual: {conditional_probe_acc_residual:.4f} '
                            f'(<= {cond_probe_threshold})'
                        )
                        print(
                            "   conditional residual per-sensor: "
                            f"GelSight={conditional_probe_acc_residual_gelsight:.4f} (<= {cond_probe_threshold_gelsight}), "
                            f"DIGIT={conditional_probe_acc_residual_digit:.4f} (<= {cond_probe_threshold_digit}), "
                            f"DuraGel={conditional_probe_acc_residual_duragel:.4f} (<= {cond_probe_threshold_duragel})"
                        )
                        print(
                            f'   pair_link_len_p90: {pair_link_len_p90:.4f} '
                            f'(<= {pair_link_p90_threshold})'
                        )
                    if use_global_mix_constraints:
                        print(
                            f'   sensor_centroid_dist: {sensor_centroid_dist:.4f} '
                            f'(<= {sensor_centroid_dist_threshold})'
                        )
                        print(
                            f'   cross_sensor_knn_mix_rate: {cross_sensor_knn_mix_rate:.4f} '
                            f'(>= {knn_mix_rate_threshold})'
                        )
            else:
                # 无改进或约束未满足
                epochs_no_improve += 1
                if misc.is_main_process():
                    if current_same_sim <= same_sim_threshold:
                        reason = f" (same_sim={current_same_sim:.4f} <= {same_sim_threshold}, 表征质量约束未通过)"
                    elif current_diff_sim >= diff_sim_threshold:
                        reason = f" (diff_sim={current_diff_sim:.4f} >= {diff_sim_threshold}, 辅助约束未通过)"
                    elif probe_acc_threshold is not None and domain_acc > probe_acc_threshold:
                        reason = f" (domain_acc={domain_acc:.4f} > {probe_acc_threshold}, 域不变性约束未通过)"
                    elif use_conditional_constraints and (
                        conditional_probe_acc_residual is None or
                        conditional_probe_acc_residual_gelsight is None or
                        conditional_probe_acc_residual_digit is None or
                        conditional_probe_acc_residual_duragel is None or
                        pair_link_len_p90 is None or
                        conditional_probe_acc_residual > cond_probe_threshold or
                        conditional_probe_acc_residual_gelsight > cond_probe_threshold_gelsight or
                        conditional_probe_acc_residual_digit > cond_probe_threshold_digit or
                        conditional_probe_acc_residual_duragel > cond_probe_threshold_duragel or
                        pair_link_len_p90 > pair_link_p90_threshold
                    ):
                        reason = (
                            f" (conditional_probe_acc_residual={conditional_probe_acc_residual}, "
                            f"cond_residual_gelsight={conditional_probe_acc_residual_gelsight}, "
                            f"cond_residual_digit={conditional_probe_acc_residual_digit}, "
                            f"cond_residual_duragel={conditional_probe_acc_residual_duragel}, "
                            f"pair_link_len_p90={pair_link_len_p90}, 条件去域约束未通过)"
                        )
                    elif use_global_mix_constraints and (
                        sensor_centroid_dist is None or
                        cross_sensor_knn_mix_rate is None or
                        (not _is_finite_scalar(sensor_centroid_dist)) or
                        (not _is_finite_scalar(cross_sensor_knn_mix_rate)) or
                        sensor_centroid_dist > sensor_centroid_dist_threshold or
                        cross_sensor_knn_mix_rate < knn_mix_rate_threshold
                    ):
                        reason = (
                            f" (sensor_centroid_dist={sensor_centroid_dist}, "
                            f"cross_sensor_knn_mix_rate={cross_sensor_knn_mix_rate}, 全局混合约束未通过)"
                        )
                    elif metric_value is None:
                        reason = f" ({metric_name_for_log} 不可用)"
                    else:
                        comparator = "<=" if metric_higher_is_better else ">="
                        best_str = f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A"
                        reason = f" ({metric_name_for_log}={metric_value:.4f} {comparator} {best_str})"
                    print(f'⚠️  No improvement for {epochs_no_improve} epochs (best epoch: {best_epoch}){reason}')
            
            # 🔄 同步最优指标/无改进计数，避免各rank分支不一致
            if misc.is_dist_avail_and_initialized():
                sync_state = [
                    best_metric_value,
                    best_same_sim,
                    best_diff_sim,
                    best_epoch,
                    epochs_no_improve
                ]
                dist.broadcast_object_list(sync_state, src=0)
                best_metric_value, best_same_sim, best_diff_sim, best_epoch, epochs_no_improve = sync_state
            
            # Early Stopping检查
            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                if misc.is_main_process():
                    best_metric_str = f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A"
                    print(f'\n🛑 Early Stopping triggered!')
                    print(f'   No improvement for {early_stopping_patience} epochs')
                    print(f'   Best {primary_metric}: {best_metric_str} (epoch {best_epoch})')
                    print(f'   Best same_contact_sim snapshot: {best_same_sim:.4f}')
                    print(f'   Best diff_contact_sim snapshot: {best_diff_sim:.4f}')
                    print(f'   Stopping training at epoch {epoch}')
                stop_due_to_early = True

            # W&B: 记录主指标与约束状态（真实值）
            if use_wandb and metric_value is not None:
                val_log = {
                    'epoch': epoch,
                    'val/primary_metric': metric_value,
                    'val/constraints_passed': float(constraints_passed),
                    'val/guardrail_bad_epochs': float(phase2_conditional_bad_epochs),
                    'val/guardrail_good_epochs': float(phase2_conditional_good_epochs),
                    'val/guardrail_cond_bad_epochs': float(phase2_conditional_bad_epochs),
                    'val/guardrail_cond_good_epochs': float(phase2_conditional_good_epochs),
                    'val/digit_focus_weight_runtime': float(getattr(args, 'digit_focus_weight', 1.0))
                }
                if use_conditional_constraints:
                    val_log['val/conditional_constraints_passed'] = float(conditional_constraints_passed)
                    if conditional_probe_acc_residual is not None:
                        val_log['val/conditional_probe_acc_residual_threshold'] = cond_probe_threshold
                        val_log['val/conditional_probe_acc_residual_gelsight_threshold'] = cond_probe_threshold_gelsight
                        val_log['val/conditional_probe_acc_residual_digit_threshold'] = cond_probe_threshold_digit
                        val_log['val/conditional_probe_acc_residual_duragel_threshold'] = cond_probe_threshold_duragel
                    if pair_link_len_p90 is not None:
                        val_log['val/pair_link_p90_threshold'] = pair_link_p90_threshold
                if use_global_mix_constraints:
                    val_log['val/global_mix_constraints_passed'] = float(global_mix_constraints_passed)
                    val_log['val/sensor_centroid_dist_threshold'] = sensor_centroid_dist_threshold
                    val_log['val/knn_mix_rate_threshold'] = knn_mix_rate_threshold
                    if _is_finite_scalar(sensor_centroid_dist):
                        val_log['val/sensor_centroid_dist'] = float(sensor_centroid_dist)
                    if _is_finite_scalar(cross_sensor_knn_mix_rate):
                        val_log['val/cross_sensor_knn_mix_rate'] = float(cross_sensor_knn_mix_rate)
                if primary_metric in ('phase1_combined', 'phase2_combined'):
                    val_log[f'val/{primary_metric}'] = metric_value
                wandb.log(val_log)
            
            # 🔄 同步早停标志，确保所有rank一致退出
            if misc.is_dist_avail_and_initialized():
                stop_flag = [stop_due_to_early]
                dist.broadcast_object_list(stop_flag, src=0)
                stop_due_to_early = stop_flag[0]

        if stop_due_to_early:
            if misc.is_dist_avail_and_initialized():
                dist.barrier()
            break
        
        # ================================================================
        # 保存 checkpoint (只保存 last.pth)
        # ================================================================
        if misc.is_main_process():
            # 每个epoch更新last.pth
            _save_named_checkpoint('last.pth', epoch, {
                'args': args,
            })
            
            # 只在第一个和最后一个epoch打印保存信息
            if epoch == args.start_epoch or epoch == args.epochs - 1:
                print(f'💾 Last checkpoint saved: {os.path.join(args.output_dir, "last.pth")}')
        
        # Wandb 日志
        if use_wandb:
            wandb_log = {
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/loss_mp': train_stats['loss_mp'],
                'train/loss_im': train_stats['loss_im'],
                'train/loss_im_contact': train_stats.get('loss_im_contact', 0.0),
                'train/loss_im_residual': train_stats.get('loss_im_residual', 0.0),
                'train/loss_pair': train_stats.get('loss_pair', 0.0),
                'train/loss_pair_tail': train_stats.get('loss_pair_tail', 0.0),
                'train/loss_inter': train_stats.get('loss_inter', 0.0),
                'train/loss_inter_sensor': train_stats.get('loss_inter_sensor', 0.0),
                'train/loss_residual_compact': train_stats.get('loss_residual_compact', 0.0),
                'train/loss_decouple': train_stats.get('loss_decouple', 0.0),
                'train/loss_domain_cls': train_stats.get('loss_domain_cls', 0.0),
                'train/loss_coral_residual': train_stats.get('loss_coral_residual', 0.0),
                'train/balance_im': train_stats.get('balance_im', 1.0),
                'train/balance_pair': train_stats.get('balance_pair', 1.0),
                'train/balance_pair_tail': train_stats.get('balance_pair_tail', 1.0),
                'train/balance_inter': train_stats.get('balance_inter', 1.0),
                'train/balance_inter_sensor': train_stats.get('balance_inter_sensor', 1.0),
                'train/balance_residual_compact': train_stats.get('balance_residual_compact', 1.0),
                'train/balance_domain_cls': train_stats.get('balance_domain_cls', 1.0),
                'train/balance_coral_residual': train_stats.get('balance_coral_residual', 1.0),
                'train/lr': train_stats['lr'],
                'lambda_im': lambda_im,
                'train/im_loss_scale_runtime': float(getattr(args, 'im_loss_scale', 1.0)),
                'train/lambda_pair_runtime': float(getattr(args, 'lambda_pair', 0.0)),
                'train/lambda_pair_tail_runtime': float(getattr(args, 'lambda_pair_tail', 0.0)),
                'train/lambda_inter_runtime': float(getattr(args, 'lambda_inter', 0.0)),
                'train/lambda_inter_sensor_runtime': float(getattr(args, 'lambda_inter_sensor', 0.0)),
                'train/lambda_residual_compact_runtime': float(
                    getattr(args, 'lambda_residual_compact', 0.0)
                ),
                'train/digit_focus_weight_runtime': float(getattr(args, 'digit_focus_weight', 1.0)),
            }
            if best_metric_value_any is not None:
                wandb_log['best_primary_metric'] = best_metric_value_any
                wandb_log['best_epoch'] = best_epoch_any
            if best_metric_value is not None:
                wandb_log['best_primary_metric_constrained'] = best_metric_value
                wandb_log['best_epoch_constrained'] = best_epoch
            if best_same_sim_any >= 0.0:
                wandb_log['best_same_sim'] = best_same_sim_any
            if best_same_sim >= 0.0:
                wandb_log['best_same_sim_constrained'] = best_same_sim
            if best_diff_sim_any is not None:
                wandb_log['best_diff_sim'] = best_diff_sim_any
            if best_diff_sim != float('inf'):
                wandb_log['best_diff_sim_constrained'] = best_diff_sim
            if best_semantic_epoch >= 0:
                wandb_log['best_semantic_same_sim'] = best_semantic_sim
                wandb_log['best_semantic_epoch'] = best_semantic_epoch
            if best_domain_epoch >= 0:
                wandb_log['best_domain_acc'] = best_domain_acc
                wandb_log['best_domain_epoch'] = best_domain_epoch
            if best_pareto_epoch >= 0 and best_pareto_score is not None:
                wandb_log['best_pareto_score'] = best_pareto_score
                wandb_log['best_pareto_epoch'] = best_pareto_epoch
            wandb.log(wandb_log)
    
    # ====================================================================
    # 训练结束
    # ====================================================================
    hard_gate_enabled = (
        args.stage == '2b' and
        bool(getattr(args, 'use_independent_probe', False)) and
        bool(getattr(args, 'phase2_require_best', False))
    )
    hard_gate_failed = False
    if hard_gate_enabled:
        best_path = os.path.join(args.output_dir, 'best.pth')
        if misc.is_main_process():
            hard_gate_failed = (best_epoch < 0) or (not os.path.exists(best_path))
        if misc.is_dist_avail_and_initialized():
            sync_fail = [hard_gate_failed]
            dist.broadcast_object_list(sync_fail, src=0)
            hard_gate_failed = bool(sync_fail[0])
        if hard_gate_failed:
            if misc.is_main_process():
                print("\n❌ Phase-2 hard gate failed: constrained best.pth was not produced.")
                print("   Required constraints were not jointly satisfied during training.")
                required_items = [
                    f"same_contact_sim>={getattr(args, 'phase2_same_sim_threshold', 0.80):.2f}",
                    f"independent_probe_acc<={getattr(args, 'phase2_probe_acc_threshold', 0.38):.2f}",
                ]
                if bool(getattr(args, 'phase2_use_conditional_constraints', False)):
                    sensor_th = float(
                        getattr(
                            args,
                            'phase2_conditional_probe_sensor_threshold',
                            getattr(args, 'phase2_conditional_probe_threshold', 0.42)
                        )
                    )
                    th_g = float(getattr(args, 'phase2_conditional_probe_gelsight_threshold', -1.0))
                    if th_g < 0:
                        th_g = sensor_th
                    th_d = float(getattr(args, 'phase2_conditional_probe_digit_threshold', -1.0))
                    if th_d < 0:
                        th_d = sensor_th
                    th_u = float(getattr(args, 'phase2_conditional_probe_duragel_threshold', -1.0))
                    if th_u < 0:
                        th_u = sensor_th
                    required_items.extend([
                        f"conditional_probe_acc_residual<={getattr(args, 'phase2_conditional_probe_threshold', 0.42):.2f}",
                        f"conditional_probe_acc_residual_gelsight<={th_g:.2f}",
                        f"conditional_probe_acc_residual_digit<={th_d:.2f}",
                        f"conditional_probe_acc_residual_duragel<={th_u:.2f}",
                        f"pair_link_len_p90<={getattr(args, 'phase2_pair_link_p90_threshold', 0.35):.2f}",
                    ])
                if bool(getattr(args, 'phase2_use_global_mix_constraints', False)):
                    required_items.extend([
                        f"sensor_centroid_dist<={getattr(args, 'phase2_sensor_centroid_dist_threshold', 0.45):.2f}",
                        f"cross_sensor_knn_mix_rate>={getattr(args, 'phase2_knn_mix_rate_threshold', 0.55):.2f}",
                    ])
                print("   Required: " + ", ".join(required_items))
            raise RuntimeError(
                "Phase2 hard acceptance gate failed: best.pth not saved with all constraints passed."
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('\n' + '=' * 70)
    print('Training Completed')
    print('=' * 70)
    print(f'Total time: {total_time_str}')
    best_metric_str = f"{best_metric_value:.4f}" if best_metric_value is not None else "N/A"
    best_semantic_str = f"{best_semantic_sim:.4f}" if best_semantic_epoch >= 0 else "N/A"
    best_domain_str = f"{best_domain_acc:.4f}" if best_domain_epoch >= 0 else "N/A"
    best_pareto_str = f"{best_pareto_score:.4f}" if best_pareto_score is not None else "N/A"
    print(f'Best {primary_metric}: {best_metric_str} (epoch {best_epoch})')
    print(f'Same_contact_sim at best: {best_same_sim:.4f}')
    print(f'Diff_contact_sim at best: {best_diff_sim:.4f}')
    print(f'Best semantic same_sim: {best_semantic_str} (epoch {best_semantic_epoch})')
    print(f'Best domain acc: {best_domain_str} (epoch {best_domain_epoch})')
    print(f'Best pareto score: {best_pareto_str} (epoch {best_pareto_epoch})')
    print(f'\n📂 Saved models:')
    print(f'   - best.pth: Best {primary_metric} (epoch {best_epoch})')
    print(f'   - best_semantic.pth: Best semantic checkpoint (epoch {best_semantic_epoch})')
    print(f'   - best_domain.pth: Best domain checkpoint (epoch {best_domain_epoch})')
    print(f'   - best_pareto.pth: Best pareto checkpoint (epoch {best_pareto_epoch})')
    print(f'   - best_any.pth: Unconstrained best primary metric')
    print(f'   - last.pth: Last epoch (epoch {args.epochs - 1})')
    print(f'\n💾 Output directory: {args.output_dir}')
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    # 使用 config2.py 的参数解析器
    parser = parse_args()
    args = parser.parse_args()
    args = mark_explicit_args(parser, args)
    
    # 创建输出目录
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
