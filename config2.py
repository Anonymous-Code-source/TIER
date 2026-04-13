"""
config2.py - PhysioTouch Stage-2 跨传感器无感表征学习配置

功能:
- Stage-2a: 对齐预热训练（Multi-Positive InfoNCE）
- Stage-2b: InfoMax-Share 去域训练
- Stage-2c: Group-DRO 鲁棒性训练

版本: v1.0
日期: 2025-11-01
作者: PhysioTouch Team

与 Stage-1 (config.py) 的区别:
- Stage-1: 单传感器触觉视频MAE预训练（静态+动态重建）
- Stage-2: 跨传感器无感表征学习（对比学习+去域）
"""

import argparse
import os
from typing import Any, Dict, Set


def _extract_parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        if not getattr(action, 'dest', None):
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def mark_explicit_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """记录哪些参数被显式覆盖，方便阶段默认值判断"""
    defaults = _extract_parser_defaults(parser)
    explicit: Set[str] = set()
    for dest, default in defaults.items():
        if not hasattr(args, dest):
            continue
        if getattr(args, dest) != default:
            explicit.add(dest)
    args._parser_defaults = defaults
    args._explicit_args = explicit
    return args


def parse_args():
    """
    Stage-2 参数解析
    
    返回:
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(
        description='PhysioTouch Stage-2 Training Configuration',
        add_help=True
    )

    # ====================================================================
    # 🎯 基础训练参数 (✅ 所有阶段通用)
    # ====================================================================
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每个GPU的批次大小 (default: 64)')
    parser.add_argument('--epochs', default=60, type=int,
                        help='训练总轮数 (default: 60，各阶段可能不同)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积步数，用于模拟更大的batch size (default: 1)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='训练设备: cuda 或 cpu (default: cuda)')
    parser.add_argument('--seed', default=0, type=int,
                        help='随机种子，用于结果复现 (default: 0)')

    # ====================================================================
    # 🔄 Stage-2 特定参数 (✅ 核心差异化配置)
    # ====================================================================
    parser.add_argument('--stage', default='2a', type=str, 
                        choices=['2a', '2b', '2c'],
                        help='训练阶段: 2a (对齐预热), 2b (去域优化), 2c (Group-DRO)')
    parser.add_argument('--primary_metric', default='same_sim', type=str,
                        choices=['same_sim', 'group_loss_gap', 'phase1_combined', 'phase2_combined'],
                        help='最佳模型与早停的主指标: same_sim, group_loss_gap, phase1_combined, phase2_combined (default: same_sim)')
    parser.add_argument('--same_sim_threshold', default=0.70, type=float,
                        help='保存best模型时的same_contact_sim下限 (default: 0.70)')
    parser.add_argument('--diff_sim_threshold', default=0.35, type=float,
                        help='保存best模型时的diff_contact_sim上限 (default: 0.35)')
    
    # InfoMax-Share 损失权重配置
    parser.add_argument('--lambda_im', default=0.0, type=float,
                        help='InfoMax-Share 损失权重 (Stage-2a默认0.0，不使用去域)')
    parser.add_argument('--lambda_im_start', default=0.0, type=float,
                        help='Stage-2b 线性调度起始值 (default: 0.0, 改进: 延迟启动)')
    parser.add_argument('--lambda_im_end', default=0.10, type=float,
                        help='Stage-2b 线性调度终点值 (default: 0.10, 降低以保持判别性)')
    parser.add_argument('--lambda_im_warmup_ratio', default=0.3, type=float,
                        help='Lambda_im warmup阶段比例 (前N%%epoch不启用域不变性, default: 0.3)')
    parser.add_argument('--lambda_im_rampup_ratio', default=0.4, type=float,
                        help='Lambda_im rampup阶段比例 (中间N%%epoch渐进增加, default: 0.4)')
    
    # 对比学习参数
    parser.add_argument('--temperature', default=0.05, type=float,
                        help='InfoNCE 损失的温度参数τ，控制相似度的锐化程度 (default: 0.05, 降低以增强判别性)')
    parser.add_argument('--use_semi_hard_negatives', action='store_true', 
                        default=True,
                        help='是否使用半硬负样本采样策略，提升对比学习质量')
    parser.add_argument('--similarity_window_low', default=0.2, type=float,
                        help='半硬负样本的相似度下界 (default: 0.2)')
    parser.add_argument('--similarity_window_high', default=0.8, type=float,
                        help='半硬负样本的相似度上界 (default: 0.8)')
    parser.add_argument('--dynamic_similarity_window', action='store_true', default=False,
                        help='是否启用半硬负样本相似度窗口动态调度 (default: False)')
    parser.add_argument('--similarity_window_low_start', default=0.0, type=float,
                        help='动态调度：相似度下界起点 (default: 0.0)')
    parser.add_argument('--similarity_window_high_start', default=1.0, type=float,
                        help='动态调度：相似度上界起点 (default: 1.0)')
    parser.add_argument('--similarity_window_low_end', default=0.2, type=float,
                        help='动态调度：相似度下界终点 (default: 0.2)')
    parser.add_argument('--similarity_window_high_end', default=0.8, type=float,
                        help='动态调度：相似度上界终点 (default: 0.8)')

    # Matching 辅助损失 (Stage-2a 可选)
    parser.add_argument('--lambda_match', default=0.0, type=float,
                        help='Cross-sensor matching 辅助损失权重 (default: 0.0)')
    parser.add_argument('--match_negatives_per_pair', default=1, type=int,
                        help='每对传感器匹配的负样本数量 (default: 1)')
    parser.add_argument('--match_warmup_ratio', default=0.0, type=float,
                        help='Matching loss 预热阶段比例 (前N%% epoch 不启用 matching, default: 0.0)')
    parser.add_argument('--match_rampup_ratio', default=0.0, type=float,
                        help='Matching loss 线性增长阶段比例 (default: 0.0)')
    parser.add_argument('--matching_hard_negative_topk', default=16, type=int,
                        help='Matching hard negative 候选池大小（按相似度top-k筛选，default: 16）')
    parser.add_argument('--enable_matching_in_phase2', action='store_true', default=False,
                        help='是否在Stage-2b(phase2)启用matching辅助损失 (default: False)')
    parser.add_argument('--lambda_pair', default=0.0, type=float,
                        help='Row-wise跨传感器配对拉近损失权重（default: 0.0）')
    parser.add_argument('--lambda_pair_tail', default=0.0, type=float,
                        help='Row-wise hard-tail配对损失权重（仅优化最难样本，default: 0.0）')
    parser.add_argument('--pair_tail_ratio', default=0.2, type=float,
                        help='hard-tail比例，取每对中最难的top-ratio样本 (default: 0.2)')
    parser.add_argument('--pair_tail_min_samples', default=16, type=int,
                        help='hard-tail最小样本数，防止小batch过度稀疏 (default: 16)')
    parser.add_argument('--lambda_inter', default=0.0, type=float,
                        help='团间分离损失权重（不同contact分离，default: 0.0）')
    parser.add_argument('--lambda_inter_sensor', default=0.0, type=float,
                        help='传感器内团间分离损失权重（每个sensor分支单独分离contact，default: 0.0）')
    parser.add_argument('--inter_sim_margin', default=0.15, type=float,
                        help='团间分离相似度阈值margin，超过即惩罚 (default: 0.15)')
    parser.add_argument('--inter_hard_topk', default=512, type=int,
                        help='团间分离仅取最高相似top-k负对计算 (default: 512, <=0表示全部)')
    parser.add_argument('--pair_loss_type', default='cosine', type=str, choices=['cosine', 'mse'],
                        help='Row-wise pair loss 类型：cosine 或 mse (default: cosine)')
    parser.add_argument('--use_pair_confidence_weighting', action='store_true', default=True,
                        help='是否按pair_confidence对row-wise pair loss加权 (default: True)')
    parser.add_argument('--no_use_pair_confidence_weighting', action='store_false',
                        dest='use_pair_confidence_weighting',
                        help='关闭pair_confidence加权，恢复等权pair loss')
    parser.add_argument('--pair_confidence_floor', default=0.05, type=float,
                        help='pair_confidence最小权重下限，避免权重为0 (default: 0.05)')
    parser.add_argument('--pair_confidence_power', default=1.0, type=float,
                        help='pair_confidence幂指数，>1更强调高置信样本 (default: 1.0)')
    parser.add_argument('--pair_confidence_min', default=0.0, type=float,
                        help='row-wise pair loss仅保留pair_confidence>=该阈值的样本 (default: 0.0, 表示不过滤)')
    parser.add_argument('--use_pair_confidence_for_domain_and_inter', action='store_true', default=False,
                        help='将pair_confidence加权/过滤扩展到L_domain与L_inter (default: False)')
    parser.add_argument('--no_use_pair_confidence_for_domain_and_inter', action='store_false',
                        dest='use_pair_confidence_for_domain_and_inter',
                        help='关闭L_domain与L_inter的置信度引导')

    # Sensor token 正则 (Stage-2a 可选)
    parser.add_argument('--stage2_use_sensor_token', action='store_true', default=False,
                        help='Stage-2 是否传入传感器ID以启用 sensor token (default: False)')
    parser.add_argument('--sensor_token_drop_start', default=0.0, type=float,
                        help='Sensor token dropout 起始概率 (default: 0.0)')
    parser.add_argument('--sensor_token_drop_end', default=0.0, type=float,
                        help='Sensor token dropout 结束概率 (default: 0.0)')
    
    # Group-DRO 参数 (Stage-2c 专用)
    parser.add_argument('--use_group_dro', action='store_true',
                        help='是否启用 Group-DRO 鲁棒性优化 (仅Stage-2c使用)')
    parser.add_argument('--group_dro_eta', default=0.01, type=float,
                        help='Group-DRO 学习率η，控制权重更新速度 (default: 0.01)')
    parser.add_argument('--group_dro_center_losses', action='store_true',
                        help='Group-DRO 权重更新时对组损失做中心化 (loss-mean(loss))，提升对损失尺度变化的鲁棒性')
    parser.add_argument('--group_dro_normalize_losses', action='store_true',
                        help='Group-DRO 权重更新时对组损失做标准化 ((loss-mean)/std)，提升对损失尺度变化的鲁棒性')
    parser.add_argument('--group_dro_step_size', default=0.01, type=float,
                        help='Group-DRO 步长α (default: 0.01)')
    parser.add_argument('--group_dro_eta_cap', default=0.05, type=float,
                        help='启用loss标准化时允许的最大 Group-DRO η，防止指数更新失控')
    parser.add_argument('--group_dro_disable_eta_cap', action='store_true',
                        help='禁用 Group-DRO η 的截断（仅在确认数值稳定时使用）')
    parser.add_argument('--group_dro_min_weight', default=0.0, type=float,
                        help='Group-DRO 权重的最小值（>0 可防止某个组被完全挤压）')
    parser.add_argument('--group_dro_max_weight', default=1.0, type=float,
                        help='Group-DRO 权重的最大值（<1 可避免单组完全支配）')
    
    # 方差正则化参数 (防止特征坍缩)
    parser.add_argument('--use_variance_reg', action='store_true', default=True,
                        help='是否启用方差正则化，防止特征坍缩 (default: True)')
    parser.add_argument('--lambda_var', default=0.1, type=float,
                        help='方差正则化权重 (default: 0.1)')
    parser.add_argument('--target_std', default=0.15, type=float,
                        help='目标表征标准差 (健康范围: 0.10-0.20, default: 0.15)')
    parser.add_argument('--stage2_num_frames', default=3, type=int,
                        help='Stage-2输入使用的时间帧数（从batch前序截取，default: 3）')

    # ====================================================================
    # 🤖 模型参数 (✅ Stage-2 模型配置)
    # ====================================================================
    parser.add_argument('--stage1_checkpoint', 
                        default='./log/joint_alpha0.75_beta0.25/best.pth',
                        type=str,
                        help='Stage-1 预训练编码器的检查点路径')
    parser.add_argument('--freeze_blocks', default=8, type=int,
                        help='冻结编码器的前N层 (ViT-B/16总共12层，默认冻结前8层)')
    parser.add_argument('--projection_dim', default=128, type=int,
                        help='Projection Head 输出维度 (768→128，用于对比学习)')
    parser.add_argument('--domain_projection_dim', default=128, type=int,
                        help='Domain 分支投影维度 (768→128，用于域对抗，default: 128)')
    parser.add_argument('--projection_hidden_dim', default=256, type=int,
                        help='Projection Head 隐藏层维度 (default: 256)')
    parser.add_argument('--discriminator_hidden_dim', default=64, type=int,
                        help='Domain Discriminator 隐藏层维度 (default: 64)')
    parser.add_argument('--discriminator_dropout', default=0.1, type=float,
                        help='Domain Discriminator Dropout 概率 (default: 0.1)')
    parser.add_argument('--num_sensors', default=3, type=int,
                        help='传感器数量 (GelSight, DIGIT, DuraGel = 3)')
    
    # 🆕 对抗式训练参数（Phase-2修复）
    parser.add_argument('--use_adversarial_training', action='store_true',
                        help='使用对抗式训练（分离Encoder和Discriminator优化器）')
    parser.add_argument('--adversarial_feature_target', type=str, default='domain',
                        choices=['domain', 'contact', 'contact_residual'],
                        help='域对抗/InfoMax作用分支: domain/contact/contact_residual (default: domain)')
    parser.add_argument('--phase2_adv_target_switch', action='store_true', default=False,
                        help='Phase-2后半程切换域对抗分支（例如 contact -> contact_residual）')
    parser.add_argument('--phase2_adv_target_switch_ratio', type=float, default=0.5,
                        help='Phase-2域对抗分支切换比例（按epoch进度，default: 0.5）')
    parser.add_argument('--phase2_adv_target_late', type=str, default='contact_residual',
                        choices=['domain', 'contact', 'contact_residual'],
                        help='Phase-2切换后使用的域对抗分支 (default: contact_residual)')
    parser.add_argument('--lr_discriminator', type=float, default=1e-4,
                        help='判别器学习率（对抗式训练模式）(default: 1e-4)')
    parser.add_argument('--disc_update_every', type=int, default=1,
                        help='判别器更新频率（每N步更新一次，default: 1）')
    parser.add_argument('--disc_steps_per_iter', type=int, default=1,
                        help='每次判别器触发更新时，连续优化的步数（default: 1）')
    parser.add_argument('--im_loss_scale', type=float, default=1.0,
                        help='InfoMax损失缩放系数，总损失使用 lambda_im * im_loss_scale * loss_im (default: 1.0)')
    parser.add_argument('--im_dual_target', action='store_true', default=False,
                        help='启用双目标InfoMax：同时对contact与contact_residual做去域 (default: False)')
    parser.add_argument('--im_contact_ratio_start', type=float, default=0.8,
                        help='双目标InfoMax中contact分支权重起点 (default: 0.8)')
    parser.add_argument('--im_contact_ratio_end', type=float, default=0.4,
                        help='双目标InfoMax中contact分支权重终点 (default: 0.4)')
    parser.add_argument('--use_digit_focus_reweight', action='store_true', default=False,
                        help='启用DIGIT定向域损失重加权（仅用于抑制DIGIT残余泄漏）')
    parser.add_argument('--digit_focus_weight', type=float, default=1.0,
                        help='DIGIT域损失重加权系数（1.0表示不加权）')
    parser.add_argument('--digit_focus_margin', type=float, default=0.08,
                        help='DIGIT准确率高于其余传感器的触发边界 (default: 0.08)')
    parser.add_argument('--digit_focus_step', type=float, default=0.10,
                        help='每次触发时digit_focus_weight增加步长 (default: 0.10)')
    parser.add_argument('--digit_focus_max', type=float, default=1.8,
                        help='digit_focus_weight上限 (default: 1.8)')
    parser.add_argument('--digit_focus_decay', type=float, default=0.97,
                        help='DIGIT不再领先时digit_focus_weight的恢复衰减系数 (default: 0.97)')
    parser.add_argument('--enable_dynamic_loss_balance', action='store_true', default=False,
                        help='启用辅助损失EMA量纲平衡（仅训练期，默认False）')
    parser.add_argument('--loss_balance_ema_decay', type=float, default=0.98,
                        help='EMA衰减系数，越大越平滑 (default: 0.98)')
    parser.add_argument('--loss_balance_eps', type=float, default=1e-6,
                        help='动态平衡数值稳定项 (default: 1e-6)')
    parser.add_argument('--loss_balance_min_factor', type=float, default=0.2,
                        help='动态平衡因子下限 (default: 0.2)')
    parser.add_argument('--loss_balance_max_factor', type=float, default=5.0,
                        help='动态平衡因子上限 (default: 5.0)')
    parser.add_argument('--loss_balance_warmup_steps', type=int, default=20,
                        help='动态平衡启用前的warmup迭代步数 (default: 20)')
    parser.add_argument('--lambda_decouple', type=float, default=0.0,
                        help='解耦损失权重（约束 z_contact 与 z_domain 低相关，default: 0.0）')
    parser.add_argument('--lambda_domain_cls', type=float, default=0.0,
                        help='z_domain 域监督损失权重（显式训练域分支，default: 0.0）')
    parser.add_argument('--lambda_coral', type=float, default=0.0,
                        help='z_contact 跨传感器分布对齐(CORAL)损失权重（default: 0.0）')
    parser.add_argument('--lambda_coral_residual', type=float, default=0.0,
                        help='z_contact残差(同pair去均值后)的条件CORAL权重（default: 0.0）')
    parser.add_argument('--lambda_residual_compact', type=float, default=0.0,
                        help='同contact内残差紧致损失权重（default: 0.0）')
    parser.add_argument('--coral_include_mean', action='store_true', default=True,
                        help='CORAL是否包含均值对齐项（默认True）')
    parser.add_argument('--no_coral_include_mean', action='store_false', dest='coral_include_mean',
                        help='关闭CORAL均值对齐，仅对齐协方差')
    parser.add_argument('--coral_residual_include_mean', action='store_true', default=False,
                        help='条件CORAL是否包含均值对齐（默认False）')
    parser.add_argument('--no_coral_residual_include_mean', action='store_false',
                        dest='coral_residual_include_mean',
                        help='关闭条件CORAL均值对齐，仅对齐协方差')
    parser.add_argument('--decouple_detach_domain', action='store_true', default=True,
                        help='解耦损失是否只对 z_contact 反传（默认True，单向解耦）')
    parser.add_argument('--no_decouple_detach_domain', action='store_false', dest='decouple_detach_domain',
                        help='关闭单向解耦，恢复双向解耦反传')
    parser.add_argument('--domain_cls_detach_encoder', action='store_true', default=True,
                        help='domain_cls 是否从 h.detach() 分支训练，避免把域信息写回encoder（默认True）')
    parser.add_argument('--no_domain_cls_detach_encoder', action='store_false', dest='domain_cls_detach_encoder',
                        help='关闭 domain_cls 的encoder隔离（不推荐）')
    parser.add_argument('--exclude_same_object_negatives', action='store_true', default=False,
                        help='InfoNCE中排除同object负样本，减少语义假负样本（默认False）')
    parser.add_argument('--prioritize_same_object_diff_frame_negatives', action='store_true', default=False,
                        help='InfoNCE优先保留同object异frame负样本，增强接触位点可分性（默认False）')
    parser.add_argument('--same_object_negative_keep_ratio', type=float, default=0.25,
                        help='优先保留同object异frame负样本时，额外保留其他负样本比例 (default: 0.25)')
    
    # 🆕 独立Probe评估参数（Phase-2修复）
    parser.add_argument('--use_independent_probe', action='store_true',
                        help='使用独立probe评估域不变性（避免"自证正确"）')
    parser.add_argument('--use_linear_probe', action='store_true',
                        help='使用线性探针替代MLP（更严格评估，默认False）')
    parser.add_argument('--probe_train_epochs', type=int, default=10,
                        help='独立probe训练轮数 (default: 10)')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='独立probe学习率 (default: 1e-3)')
    parser.add_argument('--phase2_same_sim_threshold', type=float, default=0.85,
                        help='Phase-2 same_sim 约束阈值 (仅在2b+独立probe时启用)')
    parser.add_argument('--phase2_probe_acc_threshold', type=float, default=0.40,
                        help='Phase-2 probe_acc 约束阈值 (仅在2b+独立probe时启用)')
    parser.add_argument('--phase2_semantic_guardrail', action='store_true', default=False,
                        help='Phase-2语义护栏：same_sim过低时自动降去域强度并提高pair强度')
    parser.add_argument('--phase2_semantic_margin', type=float, default=0.01,
                        help='触发“域高可增强去域”时要求 same_sim >= threshold + margin')
    parser.add_argument('--phase2_guardrail_lambda_decay', type=float, default=0.8,
                        help='same_sim跌破阈值时，lambda_im_end 的衰减系数 (default: 0.8)')
    parser.add_argument('--phase2_guardrail_im_scale_decay', type=float, default=0.8,
                        help='same_sim跌破阈值时，im_loss_scale 的衰减系数 (default: 0.8)')
    parser.add_argument('--phase2_guardrail_lambda_min', type=float, default=0.10,
                        help='语义护栏下 lambda_im_end 的下限 (default: 0.10)')
    parser.add_argument('--phase2_guardrail_im_scale_min', type=float, default=1.0,
                        help='语义护栏下 im_loss_scale 的下限 (default: 1.0)')
    parser.add_argument('--phase2_guardrail_pair_step', type=float, default=0.02,
                        help='same_sim跌破阈值时，lambda_pair 的增量 (default: 0.02)')
    parser.add_argument('--phase2_guardrail_pair_max', type=float, default=0.20,
                        help='语义护栏下 lambda_pair 上限 (default: 0.20)')
    parser.add_argument('--phase2_guardrail_inter_step', type=float, default=0.01,
                        help='conditional约束不达标时，lambda_inter 的增量 (default: 0.01)')
    parser.add_argument('--phase2_guardrail_inter_max', type=float, default=0.14,
                        help='语义护栏下 lambda_inter 上限 (default: 0.14)')
    parser.add_argument('--phase2_guardrail_inter_sensor_step', type=float, default=0.01,
                        help='conditional约束不达标时，lambda_inter_sensor 的增量 (default: 0.01)')
    parser.add_argument('--phase2_guardrail_inter_sensor_max', type=float, default=0.16,
                        help='语义护栏下 lambda_inter_sensor 上限 (default: 0.16)')
    parser.add_argument('--phase2_guardrail_pair_tail_step', type=float, default=0.01,
                        help='conditional约束不达标时，lambda_pair_tail 的增量 (default: 0.01)')
    parser.add_argument('--phase2_guardrail_pair_tail_max', type=float, default=0.18,
                        help='语义护栏下 lambda_pair_tail 上限 (default: 0.18)')
    parser.add_argument('--phase2_guardrail_residual_compact_step', type=float, default=0.01,
                        help='conditional约束不达标时，lambda_residual_compact 的增量 (default: 0.01)')
    parser.add_argument('--phase2_guardrail_residual_compact_max', type=float, default=0.16,
                        help='语义护栏下 lambda_residual_compact 上限 (default: 0.16)')
    parser.add_argument('--phase2_guardrail_conditional_restore_decay', type=float, default=0.95,
                        help='conditional约束恢复期的衰减系数 (default: 0.95)')
    parser.add_argument('--phase2_guardrail_domain_boost', type=float, default=1.05,
                        help='domain高且same稳时，lambda_im_end 增强系数 (default: 1.05)')
    parser.add_argument('--phase2_guardrail_escalate_patience', type=int, default=3,
                        help='conditional违规持续N次且pair/inter触顶后，触发二次增强 (default: 3)')
    parser.add_argument('--phase2_guardrail_restore_patience', type=int, default=3,
                        help='conditional连续达标N次后才执行恢复 (default: 3)')
    parser.add_argument('--phase2_guardrail_im_scale_step', type=float, default=0.5,
                        help='二次增强时im_loss_scale增加量 (default: 0.5)')
    parser.add_argument('--phase2_guardrail_im_scale_cap', type=float, default=6.0,
                        help='二次增强时im_loss_scale上限 (default: 6.0)')
    parser.add_argument('--phase2_guardrail_lambda_boost', type=float, default=1.08,
                        help='二次增强时lambda_im_end乘法放大系数 (default: 1.08)')
    parser.add_argument('--phase2_guardrail_lambda_cap', type=float, default=0.8,
                        help='二次增强时lambda_im_end的硬上限 (default: 0.8)')
    parser.add_argument('--phase2_use_conditional_constraints', action='store_true', default=False,
                        help='Phase-2是否启用条件约束（conditional probe + pair link长度）')
    parser.add_argument('--phase2_conditional_probe_threshold', type=float, default=0.45,
                        help='Phase-2条件probe阈值（residual，越低越好）')
    parser.add_argument('--phase2_conditional_probe_sensor_threshold', type=float, default=0.45,
                        help='Phase-2条件probe分传感器统一阈值（residual，越低越好）')
    parser.add_argument('--phase2_conditional_probe_gelsight_threshold', type=float, default=-1.0,
                        help='GelSight residual条件probe阈值；<0时回退到phase2_conditional_probe_sensor_threshold')
    parser.add_argument('--phase2_conditional_probe_digit_threshold', type=float, default=-1.0,
                        help='DIGIT residual条件probe阈值；<0时回退到phase2_conditional_probe_sensor_threshold')
    parser.add_argument('--phase2_conditional_probe_duragel_threshold', type=float, default=-1.0,
                        help='DuraGel residual条件probe阈值；<0时回退到phase2_conditional_probe_sensor_threshold')
    parser.add_argument('--phase2_pair_link_p90_threshold', type=float, default=0.35,
                        help='Phase-2同位置跨传感器连线长度p90阈值（越低越好）')
    parser.add_argument('--phase2_use_global_mix_constraints', action='store_true', default=False,
                        help='Phase-2是否启用全局混合约束（sensor centroid + cross-sensor kNN mix）')
    parser.add_argument('--phase2_sensor_centroid_dist_threshold', type=float, default=0.45,
                        help='Phase-2全局混合约束：传感器质心距离上限（越低越好）')
    parser.add_argument('--phase2_knn_mix_rate_threshold', type=float, default=0.55,
                        help='Phase-2全局混合约束：跨传感器kNN混合率下限（越高越好）')
    parser.add_argument('--global_mix_knn_k', type=int, default=20,
                        help='验证时计算cross-sensor kNN混合率的k (default: 20)')
    parser.add_argument('--global_mix_knn_max_samples_per_sensor', type=int, default=512,
                        help='验证时每个传感器用于kNN混合率计算的最大样本数 (default: 512)')
    parser.add_argument('--phase2_require_best', action='store_true', default=False,
                        help='Phase-2硬验收：若best.pth未产生则训练退出失败')
    parser.add_argument('--save_domain_same_floor', type=float, default=0.78,
                        help='保存best_domain.pth时same_sim下限 (default: 0.78)')
    parser.add_argument('--save_pareto_same_floor', type=float, default=0.78,
                        help='保存best_pareto.pth时same_sim下限 (default: 0.78)')
    parser.add_argument('--save_pareto_probe_ceiling', type=float, default=0.60,
                        help='保存best_pareto.pth时domain_acc上限 (default: 0.60)')
    parser.add_argument('--phase2_probe_acc_tol', type=float, default=0.067,
                        help='Phase-2 combined指标中 domain_acc 容忍区间 (default: 0.067)')
    parser.add_argument('--phase2_lambda_im_boost', type=float, default=1.0,
                        help='Phase-2 domain_acc偏高时lambda_im_end放大系数 (>1启用, default: 1.0)')
    parser.add_argument('--phase2_lambda_im_max', type=float, default=0.20,
                        help='Phase-2 lambda_im_end上限 (default: 0.20)')
    parser.add_argument('--phase2_domain_acc_patience', type=int, default=2,
                        help='Phase-2 domain_acc连续超阈值的触发次数 (default: 2)')
    parser.add_argument('--phase2_accel_factor', type=float, default=1.0,
                        help='Phase-2提前进入lambda_im增长阶段的缩放因子 (<1启用, default: 1.0)')
    parser.add_argument('--phase2_accel_warmup_min', type=float, default=0.10,
                        help='Phase-2 warmup比例下限 (default: 0.10)')
    parser.add_argument('--phase2_accel_rampup_min', type=float, default=0.20,
                        help='Phase-2 rampup比例下限 (default: 0.20)')
    parser.add_argument('--phase2_accel_patience', type=int, default=2,
                        help='Phase-2 domain_acc连续超阈值触发提前加速的次数 (default: 2)')

    # ====================================================================
    # 📈 优化器参数 (✅ Stage-2 特定学习率策略)
    # ====================================================================
    # 学习率
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Projection Head 和 Discriminator 的学习率 (default: 3e-4)')
    parser.add_argument('--lr_encoder', default=2e-4, type=float,
                        help='Encoder 微调学习率 (更小，避免破坏预训练权重) (default: 1e-4)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Cosine 学习率调度的最小值 (default: 1e-6)')
    
    # 学习率调度
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='学习率预热轮数 (default: 5)')
    parser.add_argument('--lr_schedule', default='cosine', type=str,
                        choices=['cosine', 'step', 'constant'],
                        help='学习率调度策略 (default: cosine)')
    
    # 正则化
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='AdamW 权重衰减系数 (default: 0.05)')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                        help='梯度裁剪最大范数，防止梯度爆炸 (default: 1.0)')

    # ====================================================================
    # 📊 数据加载参数 (✅ Stage-2 专用)
    # ====================================================================
    # 多数据集支持（新增，用于两阶段训练架构）
    parser.add_argument('--data_sources', 
                        default=['Fine'], 
                        type=str, 
                        nargs='+',
                        choices=['Fine', 'Indoor', 'Outdoor'],
                        help='数据源列表，支持多数据集混合训练 (例如: Fine Indoor Outdoor)')
    parser.add_argument('--data_root_fine', 
                        default='./Tacquad/',
                        type=str,
                        help='Fine数据集路径 (精细数据，30 objects)')
    parser.add_argument('--data_root_indoor', 
                        default='./Tacquad/',
                        type=str,
                        help='Indoor数据集路径 (室内数据，40 objects)')
    parser.add_argument('--data_root_outdoor', 
                        default='./Tacquad/',
                        type=str,
                        help='Outdoor数据集路径 (室外数据，40 objects)')
    parser.add_argument('--base_frame_jitter', default=0, type=int,
                        help='训练时基准帧抖动范围 (±k帧, default: 0)')
    parser.add_argument('--base_frame_jitter_prob', default=0.0, type=float,
                        help='基准帧抖动概率 (default: 0.0)')
    parser.add_argument('--contact_sampler_enabled', action='store_true', default=False,
                        help='训练集启用contact-centric object轮转采样 (default: False)')
    parser.add_argument('--contact_sampler_max_per_object_per_round', default=1, type=int,
                        help='contact采样每轮每个object最多连续样本数 (default: 1)')
    
    parser.add_argument('--num_workers', default=8, type=int,
                        help='DataLoader 的工作进程数 (default: 8)')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='是否将数据固定在CPU内存，加速GPU传输 (default: True)')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='禁用 pin_memory（降低主机内存占用，提升稳定性）')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='是否启用数据增强 (ColorJitter等，不含RandomFlip)')
    parser.add_argument('--color_jitter_strength', default=0.4, type=float,
                        help='ColorJitter 强度 (default: 0.4)')
    parser.add_argument('--train_split_ratio', default=0.9, type=float,
                        help='训练集比例 (default: 0.9, 即90%% train, 10%% val)')
    parser.add_argument('--group_balance_strategy', default='none', choices=['none', 'repeat'],
                        help='Stage-2 训练集按组均衡策略，可选: none / repeat (重复少数组样本)')
    parser.add_argument('--random_seed', default=42, type=int,
                        help='数据分割随机种子 (default: 42, 用于可复现性)')
    parser.add_argument('--prefetch_factor', default=1, type=int,
                        help='DataLoader prefetch_factor (default: 1, 仅在 num_workers>0 时生效)')

    # ====================================================================
    # 💾 检查点和日志参数 (✅ Stage-2 专用路径)
    # ====================================================================
    parser.add_argument('--output_dir', default='./log/stage2a', type=str,
                        help='检查点和日志保存路径 (按阶段区分)')
    parser.add_argument('--log_dir', default='./log/stage2a', type=str,
                        help='TensorBoard/Wandb 日志保存路径')
    parser.add_argument('--resume', default='', type=str,
                        help='从检查点恢复训练的路径 (用于续训或阶段衔接)')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='只加载模型权重，不恢复优化器/epoch状态')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='起始epoch编号 (从检查点恢复时使用)')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='每N个epoch保存一次带编号的检查点 (default: 10)')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='每N个epoch进行一次验证集评估 (default: 5)')
    parser.add_argument('--keep_last_n', default=2, type=int,
                        help='只保留最近N个带编号的检查点，节省磁盘空间 (default: 2)')
    parser.add_argument('--early_stopping_patience', default=15, type=int,
                        help='Early Stopping耐心值：N个epoch无改进则停止 (default: 15, 0表示禁用)')
    parser.add_argument('--early_stopping_min_delta', default=0.0005, type=float,
                        help='Early Stopping最小改进阈值 (default: 0.001)')

    # ====================================================================
    # 📡 Wandb 参数 (✅ Stage-2 专用实验跟踪)
    # ====================================================================
    parser.add_argument('--wandb_project', default='PhysioTouch-Stage2', 
                        type=str,
                        help='Wandb 项目名称')
    parser.add_argument('--wandb_name', default=None, type=str,
                        help='Wandb 运行名称 (默认自动生成: 日期_stage)')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='是否禁用 Wandb 日志记录')
    parser.add_argument('--wandb_tags', default=None, type=str, nargs='+',
                        help='Wandb 标签，用于实验分类 (例如: ablation, baseline)')

    # ====================================================================
    # 🌐 分布式训练参数 (✅ 多GPU训练支持)
    # ====================================================================
    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式训练的总进程数 (单GPU=1)')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='分布式训练的本地rank (-1表示非分布式)')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='是否在ITP集群上进行分布式训练')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='分布式训练初始化URL (default: env://)')
    parser.add_argument('--distributed', action='store_true',
                        help='是否启用分布式训练 (多GPU时设置)')

    # ====================================================================
    # 🧪 调试和监控参数 (✅ Stage-2 特定)
    # ====================================================================
    parser.add_argument('--print_freq', default=20, type=int,
                        help='训练日志打印频率 (每N个iteration打印一次)')
    parser.add_argument('--monitor_gradient', action='store_true',
                        help='是否监控梯度范数和冲突（用于调试）')
    parser.add_argument('--debug_mode', action='store_true',
                        help='调试模式：只训练少量batch，快速验证流程')
    parser.add_argument('--visualize_freq', default=50, type=int,
                        help='可视化频率 (每N个epoch可视化一次t-SNE)')

    return parser


# ====================================================================
# 🔧 阶段特定的默认配置函数
# ====================================================================
def get_stage2a_defaults():
    """
    Stage-2a 默认配置：对齐预热
    
    目标: 学习跨传感器的对齐表征，不进行去域
    策略: 只使用 Multi-Positive InfoNCE Loss
    """
    return {
        'stage': '2a',
        'epochs': 30,
        'batch_size': 64,
        'lambda_im': 0.0,  # 不使用去域损失
        'lambda_decouple': 0.0,
        'lambda_inter': 0.0,
        'use_pair_confidence_for_domain_and_inter': False,
        'domain_projection_dim': 128,
        'lr': 3e-4,
        'lr_encoder': 1e-4,
        'freeze_blocks': 8,  # 冻结前8层，微调后4层
        'warmup_epochs': 5,
        'temperature': 0.07,
        'use_semi_hard_negatives': True,
        'dynamic_similarity_window': True,
        'similarity_window_low_start': 0.0,
        'similarity_window_high_start': 1.0,
        'similarity_window_low_end': 0.2,
        'similarity_window_high_end': 0.8,
        'lambda_match': 0.08,
        'match_negatives_per_pair': 1,
        'match_warmup_ratio': 0.3,
        'match_rampup_ratio': 0.3,
        'stage2_use_sensor_token': True,
        'sensor_token_drop_start': 0.0,
        'sensor_token_drop_end': 0.5,
        'data_sources': ['Fine', 'Indoor', 'Outdoor'],
        'base_frame_jitter': 1,
        'base_frame_jitter_prob': 0.3,
        'output_dir': './log/stage2a_alignment',
        'log_dir': './log/stage2a_alignment',
        'wandb_name': 'stage2a_alignment',
    }


def get_stage2b_defaults():
    """
    Stage-2b 默认配置：去域优化 (改进版)
    
    目标: 显式移除传感器身份信息，增强跨域泛化
    策略: Multi-Positive InfoNCE + InfoMax-Share (延迟启动+渐进增长)
    
    改进点:
    - Temperature降低: 0.07 → 0.05 (增强判别性)
    - Lambda_im降低: 0.05→0.2 改为 0.0→0.10 (减少特征坍缩)
    - 延迟启动: 前30% epoch专注判别学习
    - 渐进增长: 中间40% epoch逐步引入域不变性
    """
    return {
        'stage': '2b',
        'epochs': 60,
        'batch_size': 64,
        'lambda_im_start': 0.0,       # 延迟启动，从0开始
        'lambda_im_end': 0.10,        # 降低最大值，避免特征坍缩
        'lambda_decouple': 0.02,      # 轻量解耦，分离接触与域特征
        'lambda_pair_tail': 0.0,
        'pair_tail_ratio': 0.2,
        'pair_tail_min_samples': 16,
        'lambda_inter': 0.02,         # 轻量团间分离，避免不同接触粘连
        'lambda_inter_sensor': 0.0,   # 可选：每个sensor内团间分离
        'inter_sim_margin': 0.15,
        'inter_hard_topk': 512,
        'lambda_coral_residual': 0.0,
        'lambda_residual_compact': 0.0,
        'domain_projection_dim': 128,
        'lambda_im_warmup_ratio': 0.3,  # 前30% epoch (18 epochs) lambda_im=0
        'lambda_im_rampup_ratio': 0.4,  # 中间40% epoch (24 epochs) 渐进增长
        'adversarial_feature_target': 'contact_residual',  # 条件去域：减少对接触主语义的误伤
        'decouple_detach_domain': True,          # 单向解耦，减少目标冲突
        'domain_cls_detach_encoder': True,       # 域监督不回流共享干路
        'exclude_same_object_negatives': False,  # 保留同object异位点负样本，强化接触可分
        'prioritize_same_object_diff_frame_negatives': True,  # 优先同object异frame难负样本
        'same_object_negative_keep_ratio': 0.25,
        'use_pair_confidence_weighting': True,
        'pair_confidence_floor': 0.05,
        'pair_confidence_power': 1.0,
        'pair_confidence_min': 0.0,
        'use_pair_confidence_for_domain_and_inter': True,
        'phase2_use_conditional_constraints': False,
        'contact_sampler_enabled': True,
        'contact_sampler_max_per_object_per_round': 1,
        'lr': 3e-4,
        'lr_encoder': 1e-4,
        'freeze_blocks': 8,
        'warmup_epochs': 5,
        'temperature': 0.05,          # 降低温度，增强判别性
        'use_semi_hard_negatives': True,
        'resume': './log/stage2a_alignment/best.pth',  # 从Stage-2a衔接
        'output_dir': './log/stage2b_improved',
        'log_dir': './log/stage2b_improved',
        'wandb_name': 'stage2b_improved',
    }


def get_stage2c_defaults():
    """
    Stage-2c 默认配置：Group-DRO 鲁棒性
    
    目标: 平衡不同传感器的性能，避免某些传感器欠拟合
    策略: Multi-Positive InfoNCE + InfoMax-Share + Group-DRO
    """
    return {
        'stage': '2c',
        'epochs': 30,
        'batch_size': 64,
        'lambda_im': 0.2,  # 固定最大值
        'lambda_decouple': 0.02,
        'lambda_inter': 0.0,
        'use_pair_confidence_for_domain_and_inter': False,
        'domain_projection_dim': 128,
        'use_group_dro': True,
        'group_dro_eta': 0.01,
        'lr': 1e-4,  # 更小的学习率，避免过度微调
        'lr_encoder': 5e-5,  # 更小的encoder学习率
        'freeze_blocks': 6,  # 解冻更多层（前6层冻结，后6层微调）
        'warmup_epochs': 3,
        'temperature': 0.07,
        'use_semi_hard_negatives': True,
        'resume': './log/stage2b_infomax/best.pth',  # 从Stage-2b衔接
        'output_dir': './log/stage2c_groupdro',
        'log_dir': './log/stage2c_groupdro',
        'wandb_name': 'stage2c_groupdro',
    }


def apply_stage_defaults(args):
    """
    根据训练阶段自动应用默认配置
    
    用法:
        parser = parse_args()
        args = parser.parse_args()
        args = apply_stage_defaults(args)
    
    优先级: 命令行参数 > 阶段默认值
    """
    stage_defaults = {
        '2a': get_stage2a_defaults(),
        '2b': get_stage2b_defaults(),
        '2c': get_stage2c_defaults(),
    }
    
    if args.stage in stage_defaults:
        defaults = stage_defaults[args.stage]
        parser_defaults = getattr(args, '_parser_defaults', {})
        explicit_args = getattr(args, '_explicit_args', set())
        
        for key, value in defaults.items():
            if key in explicit_args:
                continue
            current_value = getattr(args, key, None)
            default_value = parser_defaults.get(key)
            if current_value is None or default_value is None:
                setattr(args, key, value)
            elif current_value == default_value:
                setattr(args, key, value)
            elif not hasattr(args, key):
                setattr(args, key, value)
    
    return args


# ====================================================================
# 📝 配置打印和验证
# ====================================================================
def print_config(args):
    """美化打印配置信息"""
    print("\n" + "=" * 70)
    print(f"  PhysioTouch Stage-2 Training Configuration")
    print(f"  Training Stage: {args.stage.upper()}")
    print("=" * 70)
    
    print("\n🎯 基础配置:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Gradient accumulation: {args.accum_iter}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    
    print("\n🔄 Stage-2 特定:")
    print(f"  Lambda_im: {args.lambda_im}")
    print(f"  Lambda_decouple: {getattr(args, 'lambda_decouple', 0.0)}")
    print(f"  Lambda_domain_cls: {getattr(args, 'lambda_domain_cls', 0.0)}")
    print(f"  Lambda_coral: {getattr(args, 'lambda_coral', 0.0)}")
    print(f"  Lambda_coral_residual: {getattr(args, 'lambda_coral_residual', 0.0)}")
    print(f"  CORAL include mean: {getattr(args, 'coral_include_mean', True)}")
    print(f"  Residual CORAL include mean: {getattr(args, 'coral_residual_include_mean', False)}")
    print(f"  Decouple detach domain: {getattr(args, 'decouple_detach_domain', True)}")
    print(f"  Domain cls detach encoder: {getattr(args, 'domain_cls_detach_encoder', True)}")
    if args.stage == '2b':
        print(f"  Lambda_im schedule: {args.lambda_im_start:.3f} → {args.lambda_im_end:.3f} (linear)")
    print(f"  Adversarial target: {getattr(args, 'adversarial_feature_target', 'domain')}")
    print(
        f"  Phase2 target switch: {getattr(args, 'phase2_adv_target_switch', False)} "
        f"(ratio={getattr(args, 'phase2_adv_target_switch_ratio', 0.5):.2f}, "
        f"late={getattr(args, 'phase2_adv_target_late', 'contact_residual')})"
    )
    print(f"  IM loss scale: {getattr(args, 'im_loss_scale', 1.0)}")
    print(
        f"  IM dual target: {getattr(args, 'im_dual_target', False)} "
        f"(contact_ratio: {getattr(args, 'im_contact_ratio_start', 0.8):.2f} → "
        f"{getattr(args, 'im_contact_ratio_end', 0.4):.2f})"
    )
    print(
        f"  DIGIT focus reweight: {getattr(args, 'use_digit_focus_reweight', False)} "
        f"(weight={getattr(args, 'digit_focus_weight', 1.0):.3f}, "
        f"margin={getattr(args, 'digit_focus_margin', 0.08):.3f}, "
        f"step={getattr(args, 'digit_focus_step', 0.10):.3f}, "
        f"max={getattr(args, 'digit_focus_max', 1.8):.3f}, "
        f"decay={getattr(args, 'digit_focus_decay', 0.97):.3f})"
    )
    print(
        f"  Dynamic loss balance: {getattr(args, 'enable_dynamic_loss_balance', False)} "
        f"(ema={getattr(args, 'loss_balance_ema_decay', 0.98):.3f}, "
        f"factor={getattr(args, 'loss_balance_min_factor', 0.2):.2f}~"
        f"{getattr(args, 'loss_balance_max_factor', 5.0):.2f}, "
        f"warmup_steps={getattr(args, 'loss_balance_warmup_steps', 20)})"
    )
    print(f"  Disc update every: {getattr(args, 'disc_update_every', 1)}")
    print(f"  Disc steps/iter: {getattr(args, 'disc_steps_per_iter', 1)}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Stage2 num frames: {getattr(args, 'stage2_num_frames', 3)}")
    print(f"  Semi-hard negatives: {args.use_semi_hard_negatives}")
    if getattr(args, 'dynamic_similarity_window', False):
        print(f"  Similarity window (dynamic): "
              f"{args.similarity_window_low_start:.2f}-{args.similarity_window_high_start:.2f} "
              f"→ {args.similarity_window_low_end:.2f}-{args.similarity_window_high_end:.2f}")
    else:
        print(f"  Similarity window: {args.similarity_window_low:.2f}-{args.similarity_window_high:.2f}")
    print(f"  Matching loss (lambda_match): {getattr(args, 'lambda_match', 0.0)}")
    print(f"  Row-wise pair loss (lambda_pair): {getattr(args, 'lambda_pair', 0.0)} ({getattr(args, 'pair_loss_type', 'cosine')})")
    print(
        f"  Row-wise hard-tail pair (lambda_pair_tail): {getattr(args, 'lambda_pair_tail', 0.0)} "
        f"(ratio={getattr(args, 'pair_tail_ratio', 0.2):.2f}, "
        f"min_samples={getattr(args, 'pair_tail_min_samples', 16)})"
    )
    print(
        f"  Inter-contact separation (lambda_inter): {getattr(args, 'lambda_inter', 0.0)} "
        f"(margin={getattr(args, 'inter_sim_margin', 0.15):.3f}, "
        f"hard_topk={getattr(args, 'inter_hard_topk', 512)})"
    )
    print(
        f"  Inter-contact per sensor (lambda_inter_sensor): "
        f"{getattr(args, 'lambda_inter_sensor', 0.0)}"
    )
    print(
        f"  Residual compact loss (lambda_residual_compact): "
        f"{getattr(args, 'lambda_residual_compact', 0.0)}"
    )
    print(
        f"  Pair confidence weighting: {getattr(args, 'use_pair_confidence_weighting', True)} "
        f"(floor={getattr(args, 'pair_confidence_floor', 0.05):.2f}, "
        f"power={getattr(args, 'pair_confidence_power', 1.0):.2f}, "
        f"min={getattr(args, 'pair_confidence_min', 0.0):.2f})"
    )
    print(
        f"  Confidence-guided domain/inter: "
        f"{getattr(args, 'use_pair_confidence_for_domain_and_inter', False)}"
    )
    print(f"  Exclude same-object negatives: {getattr(args, 'exclude_same_object_negatives', False)}")
    print(
        f"  Prioritize same-object diff-frame negatives: "
        f"{getattr(args, 'prioritize_same_object_diff_frame_negatives', False)} "
        f"(keep_ratio={getattr(args, 'same_object_negative_keep_ratio', 0.25):.2f})"
    )
    if getattr(args, 'lambda_match', 0.0) > 0:
        print(f"  Matching schedule: warmup={getattr(args, 'match_warmup_ratio', 0.0):.2f}, "
              f"rampup={getattr(args, 'match_rampup_ratio', 0.0):.2f}")
        print(f"  Matching hard negatives top-k: {getattr(args, 'matching_hard_negative_topk', 16)}")
    print(f"  Enable matching in phase2: {getattr(args, 'enable_matching_in_phase2', False)}")
    print(f"  Use sensor token (Stage-2): {getattr(args, 'stage2_use_sensor_token', False)}")
    if getattr(args, 'stage2_use_sensor_token', False):
        print(f"  Sensor token dropout: {args.sensor_token_drop_start:.2f} → {args.sensor_token_drop_end:.2f}")
    if hasattr(args, 'use_group_dro') and args.use_group_dro:
        print(f"  Group-DRO: Enabled (η={args.group_dro_eta})")
    print(f"  Best-save thresholds: same_sim>{args.same_sim_threshold:.2f}, diff_sim<{args.diff_sim_threshold:.2f}")
    if getattr(args, 'stage', '2a') == '2b':
        print(
            f"  Extra saves: domain_same_floor>={getattr(args, 'save_domain_same_floor', 0.78):.2f}, "
            f"pareto_same_floor>={getattr(args, 'save_pareto_same_floor', 0.78):.2f}, "
            f"pareto_domain<={getattr(args, 'save_pareto_probe_ceiling', 0.60):.2f}"
        )
        print(
            f"  Conditional constraints: {getattr(args, 'phase2_use_conditional_constraints', False)} "
            f"(probe_res<={getattr(args, 'phase2_conditional_probe_threshold', 0.45):.2f}, "
            f"pair_link_p90<={getattr(args, 'phase2_pair_link_p90_threshold', 0.35):.2f})"
        )
        print(
            f"  Global mix constraints: {getattr(args, 'phase2_use_global_mix_constraints', False)} "
            f"(centroid_dist<={getattr(args, 'phase2_sensor_centroid_dist_threshold', 0.45):.2f}, "
            f"knn_mix@{int(getattr(args, 'global_mix_knn_k', 20))}>="
            f"{getattr(args, 'phase2_knn_mix_rate_threshold', 0.55):.2f})"
        )
        print(f"  Phase2 hard gate require best: {getattr(args, 'phase2_require_best', False)}")
        print(
            f"  Semantic guardrail: {getattr(args, 'phase2_semantic_guardrail', False)} "
            f"(margin={getattr(args, 'phase2_semantic_margin', 0.01):.2f}, "
            f"lambda_decay={getattr(args, 'phase2_guardrail_lambda_decay', 0.8):.2f}, "
            f"im_scale_decay={getattr(args, 'phase2_guardrail_im_scale_decay', 0.8):.2f}, "
            f"pair_step={getattr(args, 'phase2_guardrail_pair_step', 0.02):.3f}, "
            f"pair_tail_step={getattr(args, 'phase2_guardrail_pair_tail_step', 0.01):.3f}, "
            f"inter_step={getattr(args, 'phase2_guardrail_inter_step', 0.01):.3f}, "
            f"inter_sensor_step={getattr(args, 'phase2_guardrail_inter_sensor_step', 0.01):.3f}, "
            f"residual_compact_step={getattr(args, 'phase2_guardrail_residual_compact_step', 0.01):.3f}, "
            f"inter_max={getattr(args, 'phase2_guardrail_inter_max', 0.14):.3f}, "
            f"inter_sensor_max={getattr(args, 'phase2_guardrail_inter_sensor_max', 0.16):.3f}, "
            f"pair_tail_max={getattr(args, 'phase2_guardrail_pair_tail_max', 0.18):.3f}, "
            f"residual_compact_max={getattr(args, 'phase2_guardrail_residual_compact_max', 0.16):.3f}, "
            f"cond_restore_decay={getattr(args, 'phase2_guardrail_conditional_restore_decay', 0.95):.2f})"
        )
        print(
            f"  Conditional escalation: patience={getattr(args, 'phase2_guardrail_escalate_patience', 3)}, "
            f"restore_patience={getattr(args, 'phase2_guardrail_restore_patience', 3)}, "
            f"im_scale_step={getattr(args, 'phase2_guardrail_im_scale_step', 0.5):.3f}, "
            f"im_scale_cap={getattr(args, 'phase2_guardrail_im_scale_cap', 6.0):.3f}, "
            f"lambda_boost={getattr(args, 'phase2_guardrail_lambda_boost', 1.08):.3f}, "
            f"lambda_cap={getattr(args, 'phase2_guardrail_lambda_cap', 0.8):.3f}"
        )
    
    print("\n🤖 模型配置:")
    print(f"  Stage-1 checkpoint: {args.stage1_checkpoint}")
    print(f"  Freeze blocks: {args.freeze_blocks}/12 (ViT-B/16)")
    print(f"  Projection dim: 768 → {args.projection_dim}")
    print(f"  Domain projection dim: 768 → {getattr(args, 'domain_projection_dim', args.projection_dim)}")
    print(f"  Projection hidden: {args.projection_hidden_dim}")
    if hasattr(args, 'use_group_dro') and args.use_group_dro:
        print(f"  Discriminator hidden: {args.discriminator_hidden_dim}")
    
    print("\n📈 优化器:")
    print(f"  LR (projection): {args.lr}")
    print(f"  LR (encoder): {args.lr_encoder}")
    print(f"  Min LR: {args.min_lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clip: {args.clip_grad}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  LR schedule: {args.lr_schedule}")
    
    print("\n📊 数据加载:")
    print(f"  Data sources: {', '.join(args.data_sources)}")
    print(f"  Data root (Fine): {args.data_root_fine}")
    print(f"  Data root (Indoor): {args.data_root_indoor}")
    print(f"  Data root (Outdoor): {args.data_root_outdoor}")
    print(f"  Base frame jitter: ±{getattr(args, 'base_frame_jitter', 0)} "
          f"(p={getattr(args, 'base_frame_jitter_prob', 0.0):.2f})")
    print(
        f"  Contact sampler: {getattr(args, 'contact_sampler_enabled', False)} "
        f"(max/object/round={getattr(args, 'contact_sampler_max_per_object_per_round', 1)})"
    )
    print(f"  Num workers: {args.num_workers}")
    print(f"  Pin memory: {args.pin_mem}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    
    print("\n💾 输出路径:")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Log dir: {args.log_dir}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    
    print("\n📡 Logging:")
    print(f"  Wandb project: {args.wandb_project}")
    if not args.disable_wandb:
        print(f"  Wandb run: {args.wandb_name if args.wandb_name else 'auto-generated'}")
    else:
        print(f"  Wandb: Disabled")
    
    print("=" * 70 + "\n")


def validate_config(args):
    """
    验证配置合法性
    
    检查项:
    1. Stage-1 checkpoint 是否存在
    2. freeze_blocks 是否在合理范围
    3. lambda_im 是否在 [0, 1] 范围
    4. 数据路径是否存在
    5. 阶段衔接检查 (Stage-2b/2c 是否有 resume)
    """
    errors = []
    warnings = []
    
    # 检查 Stage-1 checkpoint 是否存在
    if not os.path.exists(args.stage1_checkpoint):
        errors.append(
            f"Stage-1 checkpoint not found: {args.stage1_checkpoint}\n"
            f"   Please train Stage-1 first or provide correct path."
        )
    
    # 检查 freeze_blocks 合理性
    if args.freeze_blocks < 0 or args.freeze_blocks > 12:
        errors.append(
            f"freeze_blocks must be in [0, 12], got {args.freeze_blocks}\n"
            f"   ViT-B/16 has 12 transformer blocks in total."
        )
    
    # 检查 lambda_im 范围
    if args.lambda_im < 0 or args.lambda_im > 1:
        errors.append(f"lambda_im must be in [0, 1], got {args.lambda_im}")
    if getattr(args, 'lambda_decouple', 0.0) < 0 or getattr(args, 'lambda_decouple', 0.0) > 1:
        errors.append(f"lambda_decouple must be in [0, 1], got {args.lambda_decouple}")
    if getattr(args, 'lambda_domain_cls', 0.0) < 0 or getattr(args, 'lambda_domain_cls', 0.0) > 1:
        errors.append(f"lambda_domain_cls must be in [0, 1], got {args.lambda_domain_cls}")
    if getattr(args, 'lambda_coral', 0.0) < 0 or getattr(args, 'lambda_coral', 0.0) > 1:
        errors.append(f"lambda_coral must be in [0, 1], got {args.lambda_coral}")
    if getattr(args, 'lambda_coral_residual', 0.0) < 0 or getattr(args, 'lambda_coral_residual', 0.0) > 1:
        errors.append(f"lambda_coral_residual must be in [0, 1], got {args.lambda_coral_residual}")

    if args.same_sim_threshold < 0 or args.same_sim_threshold > 1:
        errors.append(f"same_sim_threshold must be in [0, 1], got {args.same_sim_threshold}")

    if args.diff_sim_threshold < 0 or args.diff_sim_threshold > 1:
        errors.append(f"diff_sim_threshold must be in [0, 1], got {args.diff_sim_threshold}")
    
    if hasattr(args, 'lambda_im_start') and (args.lambda_im_start < 0 or args.lambda_im_start > 1):
        errors.append(f"lambda_im_start must be in [0, 1], got {args.lambda_im_start}")
    
    if hasattr(args, 'lambda_im_end') and (args.lambda_im_end < 0 or args.lambda_im_end > 1):
        errors.append(f"lambda_im_end must be in [0, 1], got {args.lambda_im_end}")
    if getattr(args, 'domain_projection_dim', 128) <= 0:
        errors.append(f"domain_projection_dim must be > 0, got {args.domain_projection_dim}")
    if getattr(args, 'adversarial_feature_target', 'domain') not in (
        'domain', 'contact', 'contact_residual'
    ):
        errors.append(
            f"adversarial_feature_target must be 'domain' / 'contact' / 'contact_residual', "
            f"got {args.adversarial_feature_target}"
        )
    if getattr(args, 'phase2_adv_target_switch_ratio', 0.5) < 0 or getattr(args, 'phase2_adv_target_switch_ratio', 0.5) > 1:
        errors.append(
            f"phase2_adv_target_switch_ratio must be in [0, 1], got {args.phase2_adv_target_switch_ratio}"
        )
    if getattr(args, 'phase2_adv_target_late', 'contact_residual') not in (
        'domain', 'contact', 'contact_residual'
    ):
        errors.append(
            "phase2_adv_target_late must be 'domain' / 'contact' / 'contact_residual', "
            f"got {args.phase2_adv_target_late}"
        )
    if getattr(args, 'stage', '2a') == '2b' and bool(getattr(args, 'phase2_adv_target_switch', False)):
        base_target = str(getattr(args, 'adversarial_feature_target', 'domain'))
        late_target = str(getattr(args, 'phase2_adv_target_late', 'contact_residual'))
        base_dim = (
            int(getattr(args, 'projection_dim', 128))
            if base_target in ('contact', 'contact_residual')
            else int(getattr(args, 'domain_projection_dim', getattr(args, 'projection_dim', 128)))
        )
        late_dim = (
            int(getattr(args, 'projection_dim', 128))
            if late_target in ('contact', 'contact_residual')
            else int(getattr(args, 'domain_projection_dim', getattr(args, 'projection_dim', 128)))
        )
        if base_dim != late_dim:
            errors.append(
                "phase2_adv_target_switch requires matching discriminator input dims: "
                f"base_target={base_target}(dim={base_dim}) vs "
                f"late_target={late_target}(dim={late_dim})"
            )
    if getattr(args, 'same_object_negative_keep_ratio', 0.25) < 0 or getattr(args, 'same_object_negative_keep_ratio', 0.25) > 1:
        errors.append(
            f"same_object_negative_keep_ratio must be in [0, 1], "
            f"got {args.same_object_negative_keep_ratio}"
        )
    if getattr(args, 'disc_update_every', 1) < 1:
        errors.append(f"disc_update_every must be >= 1, got {args.disc_update_every}")
    if getattr(args, 'disc_steps_per_iter', 1) < 1:
        errors.append(f"disc_steps_per_iter must be >= 1, got {args.disc_steps_per_iter}")
    if getattr(args, 'im_loss_scale', 1.0) < 0:
        errors.append(f"im_loss_scale must be >= 0, got {args.im_loss_scale}")
    if getattr(args, 'im_contact_ratio_start', 0.8) < 0 or getattr(args, 'im_contact_ratio_start', 0.8) > 1:
        errors.append(
            f"im_contact_ratio_start must be in [0, 1], got {args.im_contact_ratio_start}"
        )
    if getattr(args, 'im_contact_ratio_end', 0.4) < 0 or getattr(args, 'im_contact_ratio_end', 0.4) > 1:
        errors.append(
            f"im_contact_ratio_end must be in [0, 1], got {args.im_contact_ratio_end}"
        )
    if getattr(args, 'digit_focus_weight', 1.0) < 1.0:
        errors.append(
            f"digit_focus_weight must be >= 1.0, got {args.digit_focus_weight}"
        )
    if getattr(args, 'digit_focus_margin', 0.08) < 0 or getattr(args, 'digit_focus_margin', 0.08) > 1:
        errors.append(
            f"digit_focus_margin must be in [0, 1], got {args.digit_focus_margin}"
        )
    if getattr(args, 'digit_focus_step', 0.10) < 0:
        errors.append(
            f"digit_focus_step must be >= 0, got {args.digit_focus_step}"
        )
    if getattr(args, 'digit_focus_max', 1.8) < 1.0:
        errors.append(
            f"digit_focus_max must be >= 1.0, got {args.digit_focus_max}"
        )
    if getattr(args, 'digit_focus_weight', 1.0) > getattr(args, 'digit_focus_max', 1.8):
        errors.append(
            "digit_focus_weight must be <= digit_focus_max, "
            f"got weight={args.digit_focus_weight}, max={args.digit_focus_max}"
        )
    if getattr(args, 'stage2_num_frames', 3) < 1 or getattr(args, 'stage2_num_frames', 3) > 4:
        errors.append(
            f"stage2_num_frames must be in [1, 4], got {args.stage2_num_frames}"
        )
    if getattr(args, 'digit_focus_decay', 0.97) <= 0 or getattr(args, 'digit_focus_decay', 0.97) > 1:
        errors.append(
            f"digit_focus_decay must be in (0, 1], got {args.digit_focus_decay}"
        )
    if getattr(args, 'loss_balance_ema_decay', 0.98) <= 0 or getattr(args, 'loss_balance_ema_decay', 0.98) >= 1:
        errors.append(
            f"loss_balance_ema_decay must be in (0, 1), got {args.loss_balance_ema_decay}"
        )
    if getattr(args, 'loss_balance_eps', 1e-6) <= 0:
        errors.append(f"loss_balance_eps must be > 0, got {args.loss_balance_eps}")
    if getattr(args, 'loss_balance_min_factor', 0.2) <= 0:
        errors.append(
            f"loss_balance_min_factor must be > 0, got {args.loss_balance_min_factor}"
        )
    if getattr(args, 'loss_balance_max_factor', 5.0) < getattr(args, 'loss_balance_min_factor', 0.2):
        errors.append(
            "loss_balance_max_factor must be >= loss_balance_min_factor, "
            f"got max={args.loss_balance_max_factor}, min={args.loss_balance_min_factor}"
        )
    if getattr(args, 'loss_balance_warmup_steps', 20) < 0:
        errors.append(
            f"loss_balance_warmup_steps must be >= 0, got {args.loss_balance_warmup_steps}"
        )

    # 检查 matching loss 权重
    if getattr(args, 'lambda_match', 0.0) < 0 or getattr(args, 'lambda_match', 0.0) > 1:
        errors.append(f"lambda_match must be in [0, 1], got {args.lambda_match}")
    if getattr(args, 'lambda_pair', 0.0) < 0 or getattr(args, 'lambda_pair', 0.0) > 1:
        errors.append(f"lambda_pair must be in [0, 1], got {args.lambda_pair}")
    if getattr(args, 'lambda_pair_tail', 0.0) < 0 or getattr(args, 'lambda_pair_tail', 0.0) > 1:
        errors.append(f"lambda_pair_tail must be in [0, 1], got {args.lambda_pair_tail}")
    if getattr(args, 'pair_tail_ratio', 0.2) <= 0 or getattr(args, 'pair_tail_ratio', 0.2) > 1:
        errors.append(f"pair_tail_ratio must be in (0, 1], got {args.pair_tail_ratio}")
    if getattr(args, 'pair_tail_min_samples', 16) < 1:
        errors.append(f"pair_tail_min_samples must be >= 1, got {args.pair_tail_min_samples}")
    if getattr(args, 'lambda_inter', 0.0) < 0 or getattr(args, 'lambda_inter', 0.0) > 1:
        errors.append(f"lambda_inter must be in [0, 1], got {args.lambda_inter}")
    if getattr(args, 'lambda_inter_sensor', 0.0) < 0 or getattr(args, 'lambda_inter_sensor', 0.0) > 1:
        errors.append(
            f"lambda_inter_sensor must be in [0, 1], got {args.lambda_inter_sensor}"
        )
    if getattr(args, 'lambda_residual_compact', 0.0) < 0 or getattr(args, 'lambda_residual_compact', 0.0) > 1:
        errors.append(
            f"lambda_residual_compact must be in [0, 1], got {args.lambda_residual_compact}"
        )
    if getattr(args, 'inter_sim_margin', 0.15) < -1 or getattr(args, 'inter_sim_margin', 0.15) > 1:
        errors.append(
            f"inter_sim_margin must be in [-1, 1], got {args.inter_sim_margin}"
        )
    if getattr(args, 'inter_hard_topk', 512) < 0:
        errors.append(f"inter_hard_topk must be >= 0, got {args.inter_hard_topk}")
    if getattr(args, 'pair_loss_type', 'cosine') not in ('cosine', 'mse'):
        errors.append(f"pair_loss_type must be 'cosine' or 'mse', got {args.pair_loss_type}")
    if getattr(args, 'pair_confidence_floor', 0.05) < 0 or getattr(args, 'pair_confidence_floor', 0.05) > 1:
        errors.append(
            f"pair_confidence_floor must be in [0, 1], got {args.pair_confidence_floor}"
        )
    if getattr(args, 'pair_confidence_power', 1.0) <= 0:
        errors.append(
            f"pair_confidence_power must be > 0, got {args.pair_confidence_power}"
        )
    if getattr(args, 'pair_confidence_min', 0.0) < 0 or getattr(args, 'pair_confidence_min', 0.0) > 1:
        errors.append(
            f"pair_confidence_min must be in [0, 1], got {args.pair_confidence_min}"
        )
    if getattr(args, 'matching_hard_negative_topk', 16) < 1:
        errors.append(f"matching_hard_negative_topk must be >= 1, got {args.matching_hard_negative_topk}")
    if getattr(args, 'contact_sampler_max_per_object_per_round', 1) < 1:
        errors.append(
            f"contact_sampler_max_per_object_per_round must be >= 1, "
            f"got {args.contact_sampler_max_per_object_per_round}"
        )
    if getattr(args, 'save_domain_same_floor', 0.78) < 0 or getattr(args, 'save_domain_same_floor', 0.78) > 1:
        errors.append(
            f"save_domain_same_floor must be in [0, 1], got {args.save_domain_same_floor}"
        )
    if getattr(args, 'save_pareto_same_floor', 0.78) < 0 or getattr(args, 'save_pareto_same_floor', 0.78) > 1:
        errors.append(
            f"save_pareto_same_floor must be in [0, 1], got {args.save_pareto_same_floor}"
        )
    if getattr(args, 'save_pareto_probe_ceiling', 0.60) < 0 or getattr(args, 'save_pareto_probe_ceiling', 0.60) > 1:
        errors.append(
            f"save_pareto_probe_ceiling must be in [0, 1], got {args.save_pareto_probe_ceiling}"
        )
    if getattr(args, 'phase2_conditional_probe_threshold', 0.45) < 0 or getattr(args, 'phase2_conditional_probe_threshold', 0.45) > 1:
        errors.append(
            "phase2_conditional_probe_threshold must be in [0, 1], "
            f"got {args.phase2_conditional_probe_threshold}"
        )
    if getattr(args, 'phase2_conditional_probe_sensor_threshold', 0.45) < 0 or getattr(args, 'phase2_conditional_probe_sensor_threshold', 0.45) > 1:
        errors.append(
            "phase2_conditional_probe_sensor_threshold must be in [0, 1], "
            f"got {args.phase2_conditional_probe_sensor_threshold}"
        )
    for sensor_name, th in [
        ('phase2_conditional_probe_gelsight_threshold', getattr(args, 'phase2_conditional_probe_gelsight_threshold', -1.0)),
        ('phase2_conditional_probe_digit_threshold', getattr(args, 'phase2_conditional_probe_digit_threshold', -1.0)),
        ('phase2_conditional_probe_duragel_threshold', getattr(args, 'phase2_conditional_probe_duragel_threshold', -1.0)),
    ]:
        if th >= 0 and th > 1:
            errors.append(f"{sensor_name} must be in [0, 1] when enabled, got {th}")
    if getattr(args, 'phase2_pair_link_p90_threshold', 0.35) < 0:
        errors.append(
            f"phase2_pair_link_p90_threshold must be >= 0, got {args.phase2_pair_link_p90_threshold}"
        )
    if getattr(args, 'phase2_sensor_centroid_dist_threshold', 0.45) < 0:
        errors.append(
            "phase2_sensor_centroid_dist_threshold must be >= 0, "
            f"got {args.phase2_sensor_centroid_dist_threshold}"
        )
    if getattr(args, 'phase2_knn_mix_rate_threshold', 0.55) < 0 or getattr(args, 'phase2_knn_mix_rate_threshold', 0.55) > 1:
        errors.append(
            "phase2_knn_mix_rate_threshold must be in [0, 1], "
            f"got {args.phase2_knn_mix_rate_threshold}"
        )
    if getattr(args, 'global_mix_knn_k', 20) < 1:
        errors.append(
            f"global_mix_knn_k must be >= 1, got {args.global_mix_knn_k}"
        )
    if getattr(args, 'global_mix_knn_max_samples_per_sensor', 512) < 1:
        errors.append(
            "global_mix_knn_max_samples_per_sensor must be >= 1, "
            f"got {args.global_mix_knn_max_samples_per_sensor}"
        )
    if getattr(args, 'phase2_semantic_margin', 0.01) < 0:
        errors.append(
            f"phase2_semantic_margin must be >= 0, got {args.phase2_semantic_margin}"
        )
    if getattr(args, 'phase2_guardrail_lambda_decay', 0.8) <= 0 or getattr(args, 'phase2_guardrail_lambda_decay', 0.8) > 1:
        errors.append(
            f"phase2_guardrail_lambda_decay must be in (0, 1], got {args.phase2_guardrail_lambda_decay}"
        )
    if getattr(args, 'phase2_guardrail_im_scale_decay', 0.8) <= 0 or getattr(args, 'phase2_guardrail_im_scale_decay', 0.8) > 1:
        errors.append(
            f"phase2_guardrail_im_scale_decay must be in (0, 1], got {args.phase2_guardrail_im_scale_decay}"
        )
    if getattr(args, 'phase2_guardrail_lambda_min', 0.10) < 0 or getattr(args, 'phase2_guardrail_lambda_min', 0.10) > 1:
        errors.append(
            f"phase2_guardrail_lambda_min must be in [0, 1], got {args.phase2_guardrail_lambda_min}"
        )
    if getattr(args, 'phase2_guardrail_im_scale_min', 1.0) < 0:
        errors.append(
            f"phase2_guardrail_im_scale_min must be >= 0, got {args.phase2_guardrail_im_scale_min}"
        )
    if getattr(args, 'phase2_guardrail_pair_step', 0.02) < 0:
        errors.append(
            f"phase2_guardrail_pair_step must be >= 0, got {args.phase2_guardrail_pair_step}"
        )
    if getattr(args, 'phase2_guardrail_pair_max', 0.20) < 0 or getattr(args, 'phase2_guardrail_pair_max', 0.20) > 1:
        errors.append(
            f"phase2_guardrail_pair_max must be in [0, 1], got {args.phase2_guardrail_pair_max}"
        )
    if getattr(args, 'phase2_guardrail_inter_step', 0.01) < 0:
        errors.append(
            f"phase2_guardrail_inter_step must be >= 0, got {args.phase2_guardrail_inter_step}"
        )
    if getattr(args, 'phase2_guardrail_inter_max', 0.14) < 0 or getattr(args, 'phase2_guardrail_inter_max', 0.14) > 1:
        errors.append(
            f"phase2_guardrail_inter_max must be in [0, 1], got {args.phase2_guardrail_inter_max}"
        )
    if getattr(args, 'phase2_guardrail_inter_sensor_step', 0.01) < 0:
        errors.append(
            "phase2_guardrail_inter_sensor_step must be >= 0, "
            f"got {args.phase2_guardrail_inter_sensor_step}"
        )
    if (
        getattr(args, 'phase2_guardrail_inter_sensor_max', 0.16) < 0
        or getattr(args, 'phase2_guardrail_inter_sensor_max', 0.16) > 1
    ):
        errors.append(
            "phase2_guardrail_inter_sensor_max must be in [0, 1], "
            f"got {args.phase2_guardrail_inter_sensor_max}"
        )
    if getattr(args, 'phase2_guardrail_pair_tail_step', 0.01) < 0:
        errors.append(
            "phase2_guardrail_pair_tail_step must be >= 0, "
            f"got {args.phase2_guardrail_pair_tail_step}"
        )
    if (
        getattr(args, 'phase2_guardrail_pair_tail_max', 0.18) < 0
        or getattr(args, 'phase2_guardrail_pair_tail_max', 0.18) > 1
    ):
        errors.append(
            "phase2_guardrail_pair_tail_max must be in [0, 1], "
            f"got {args.phase2_guardrail_pair_tail_max}"
        )
    if getattr(args, 'phase2_guardrail_residual_compact_step', 0.01) < 0:
        errors.append(
            "phase2_guardrail_residual_compact_step must be >= 0, "
            f"got {args.phase2_guardrail_residual_compact_step}"
        )
    if (
        getattr(args, 'phase2_guardrail_residual_compact_max', 0.16) < 0
        or getattr(args, 'phase2_guardrail_residual_compact_max', 0.16) > 1
    ):
        errors.append(
            "phase2_guardrail_residual_compact_max must be in [0, 1], "
            f"got {args.phase2_guardrail_residual_compact_max}"
        )
    if getattr(args, 'stage', '2a') == '2b':
        if getattr(args, 'phase2_guardrail_pair_max', 0.20) < getattr(args, 'lambda_pair', 0.0):
            errors.append(
                "phase2_guardrail_pair_max must be >= lambda_pair, "
                f"got pair_max={args.phase2_guardrail_pair_max}, lambda_pair={args.lambda_pair}"
            )
        if getattr(args, 'phase2_guardrail_inter_max', 0.14) < getattr(args, 'lambda_inter', 0.0):
            errors.append(
                "phase2_guardrail_inter_max must be >= lambda_inter, "
                f"got inter_max={args.phase2_guardrail_inter_max}, lambda_inter={args.lambda_inter}"
            )
        if getattr(args, 'phase2_guardrail_inter_sensor_max', 0.16) < getattr(args, 'lambda_inter_sensor', 0.0):
            errors.append(
                "phase2_guardrail_inter_sensor_max must be >= lambda_inter_sensor, "
                f"got inter_sensor_max={args.phase2_guardrail_inter_sensor_max}, "
                f"lambda_inter_sensor={args.lambda_inter_sensor}"
            )
        if getattr(args, 'phase2_guardrail_pair_tail_max', 0.18) < getattr(args, 'lambda_pair_tail', 0.0):
            errors.append(
                "phase2_guardrail_pair_tail_max must be >= lambda_pair_tail, "
                f"got pair_tail_max={args.phase2_guardrail_pair_tail_max}, "
                f"lambda_pair_tail={args.lambda_pair_tail}"
            )
        if (
            getattr(args, 'phase2_guardrail_residual_compact_max', 0.16)
            < getattr(args, 'lambda_residual_compact', 0.0)
        ):
            errors.append(
                "phase2_guardrail_residual_compact_max must be >= lambda_residual_compact, "
                f"got residual_compact_max={args.phase2_guardrail_residual_compact_max}, "
                f"lambda_residual_compact={args.lambda_residual_compact}"
            )
    if (
        getattr(args, 'phase2_guardrail_conditional_restore_decay', 0.95) <= 0
        or getattr(args, 'phase2_guardrail_conditional_restore_decay', 0.95) > 1
    ):
        errors.append(
            "phase2_guardrail_conditional_restore_decay must be in (0, 1], "
            f"got {args.phase2_guardrail_conditional_restore_decay}"
        )
    if getattr(args, 'phase2_guardrail_domain_boost', 1.05) < 1.0:
        errors.append(
            f"phase2_guardrail_domain_boost must be >= 1.0, got {args.phase2_guardrail_domain_boost}"
        )
    if getattr(args, 'phase2_guardrail_escalate_patience', 3) < 1:
        errors.append(
            "phase2_guardrail_escalate_patience must be >= 1, "
            f"got {args.phase2_guardrail_escalate_patience}"
        )
    if getattr(args, 'phase2_guardrail_restore_patience', 3) < 1:
        errors.append(
            "phase2_guardrail_restore_patience must be >= 1, "
            f"got {args.phase2_guardrail_restore_patience}"
        )
    if getattr(args, 'phase2_guardrail_im_scale_step', 0.5) < 0:
        errors.append(
            f"phase2_guardrail_im_scale_step must be >= 0, got {args.phase2_guardrail_im_scale_step}"
        )
    if getattr(args, 'phase2_guardrail_im_scale_cap', 6.0) < 0:
        errors.append(
            f"phase2_guardrail_im_scale_cap must be >= 0, got {args.phase2_guardrail_im_scale_cap}"
        )
    if getattr(args, 'phase2_guardrail_lambda_boost', 1.08) < 1.0:
        errors.append(
            f"phase2_guardrail_lambda_boost must be >= 1.0, got {args.phase2_guardrail_lambda_boost}"
        )
    if getattr(args, 'phase2_guardrail_lambda_cap', 0.8) < 0 or getattr(args, 'phase2_guardrail_lambda_cap', 0.8) > 1:
        errors.append(
            f"phase2_guardrail_lambda_cap must be in [0, 1], got {args.phase2_guardrail_lambda_cap}"
        )

    # 检查 sensor token dropout
    if getattr(args, 'sensor_token_drop_start', 0.0) < 0 or getattr(args, 'sensor_token_drop_start', 0.0) > 1:
        errors.append(f"sensor_token_drop_start must be in [0, 1], got {args.sensor_token_drop_start}")
    if getattr(args, 'sensor_token_drop_end', 0.0) < 0 or getattr(args, 'sensor_token_drop_end', 0.0) > 1:
        errors.append(f"sensor_token_drop_end must be in [0, 1], got {args.sensor_token_drop_end}")

    # 检查相似度窗口范围
    if args.similarity_window_low > args.similarity_window_high:
        errors.append("similarity_window_low must be <= similarity_window_high")
    if getattr(args, 'dynamic_similarity_window', False):
        if args.similarity_window_low_start > args.similarity_window_high_start:
            errors.append("similarity_window_low_start must be <= similarity_window_high_start")
        if args.similarity_window_low_end > args.similarity_window_high_end:
            errors.append("similarity_window_low_end must be <= similarity_window_high_end")
    
    # 检查数据路径是否存在
    for source in args.data_sources:
        if source == 'Fine':
            data_root = args.data_root_fine
        elif source == 'Indoor':
            data_root = args.data_root_indoor
        elif source == 'Outdoor':
            data_root = args.data_root_outdoor
        else:
            continue
        
        if not os.path.exists(data_root):
            warnings.append(
                f"Data path not found: {data_root} ({source})\n"
                f"   Please ensure TacQuad dataset is downloaded."
            )
    
    # 检查阶段衔接
    if args.stage == '2b' and not args.resume:
        warnings.append(
            f"Stage-2b training without resume from Stage-2a.\n"
            f"   Recommended: --resume ./log/stage2a_alignment/best.pth"
        )
    if args.stage == '2b' and getattr(args, 'phase2_use_conditional_constraints', False) and not getattr(args, 'use_independent_probe', False):
        errors.append(
            "phase2_use_conditional_constraints=True but use_independent_probe=False.\n"
            "   Conditional probe约束需要独立probe评估，请启用 --use_independent_probe"
        )
    if args.stage == '2b' and getattr(args, 'phase2_use_global_mix_constraints', False) and not getattr(args, 'use_independent_probe', False):
        errors.append(
            "phase2_use_global_mix_constraints=True but use_independent_probe=False.\n"
            "   当前全局混合指标在独立probe路径计算，请启用 --use_independent_probe"
        )
    
    if args.stage == '2c' and not args.resume:
        warnings.append(
            f"Stage-2c training without resume from Stage-2b.\n"
            f"   Recommended: --resume ./log/stage2b_infomax/best.pth"
        )
    
    # 检查 Group-DRO 配置
    if args.stage == '2c' and not getattr(args, 'use_group_dro', False):
        warnings.append(
            f"Stage-2c is designed for Group-DRO, but use_group_dro=False.\n"
            f"   Consider setting --use_group_dro"
        )
    
    # 打印结果
    if errors:
        print("\n❌ Configuration Errors:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        raise ValueError("Configuration validation failed! Please fix the errors above.")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")
        print()
    
    if not errors and not warnings:
        print("✅ Configuration validation passed!\n")


# ====================================================================
# 🧪 测试和示例
# ====================================================================
if __name__ == '__main__':
    """测试配置文件"""
    import sys
    
    print("\n" + "=" * 70)
    print("  config2.py - Stage-2 Configuration Test")
    print("=" * 70)
    
    # 测试 Stage-2a
    print("\n📋 Testing Stage-2a defaults...")
    parser = parse_args()
    args = parser.parse_args(['--stage', '2a'])
    args = apply_stage_defaults(args)
    print_config(args)
    
    try:
        validate_config(args)
    except ValueError as e:
        print(f"Expected validation error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Config2.py test completed!")
    print("=" * 70)
    
    print("\n📖 Usage Examples:")
    print("\n1. Stage-2a (Alignment Pre-training):")
    print("   python Stage2_main.py --stage 2a")
    
    print("\n2. Stage-2b (InfoMax-Share):")
    print("   python Stage2_main.py --stage 2b --resume ./log/stage2a_alignment/best.pth")
    
    print("\n3. Stage-2c (Group-DRO):")
    print("   python Stage2_main.py --stage 2c --use_group_dro --resume ./log/stage2b_infomax/best.pth")
    
    print("\n4. Custom Configuration:")
    print("   python Stage2_main.py --stage 2a --batch_size 32 --epochs 50 --lr 1e-4")
    
    print()
