"""
stage1_engine_static.py - 支持静态重建损失的训练引擎

版本: v2.3 (Memory Leak & Stability Fix)
修复:
- 🔥🔥🔥 修复NaN跳过时的梯度内存泄漏（添加optimizer.zero_grad()和torch.cuda.empty_cache()）
- 🔥🔥 增强梯度裁剪（从1.0降低到0.5）以更好地防止loss暴涨后的数值不稳定
- 🔥🔥 添加定期内存清理机制（每50个batch清理一次）
- 🔥🔥 添加Loss暴涨实时检测和警告系统
- 🔥 修复分布式训练中NaN导致的NCCL死锁问题
- 🔥 添加分布式同步的NaN检查，确保所有GPU同步跳过异常batch
- 修复 ZeroDivisionError: 延迟创建 meter，避免 count=0
- 只在实际使用时才创建和更新 loss_static 和 loss_dynamic

新增功能:
- 支持静态重建损失（data_type=0，单帧图像）
- 支持动态重建+预测损失（data_type=1，4帧视频）
- 随机选择静态或动态模式进行训练
- 分别记录和监控两种损失
- Loss暴涨检测（基于滑动窗口平均值）

参考: AnyTouch - Learning Unified Static-Dynamic Representation
"""

import math
import sys
import random
import os
from datetime import datetime

import torch
import torch.distributed as dist
import wandb  # ⭐ 添加 Wandb
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,  
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,  
    device: torch.device,   
    epoch: int,   
    loss_scaler,  
    args=None
):  
    """  
    训练一个 epoch（支持静态+动态重建损失）
    
    参数:  
        model: 要训练的模型  
        data_loader: 数据加载器，加载连续四帧触觉图片
        optimizer: 优化器  
        device: 训练设备 (cuda/cpu)
        epoch: 当前 epoch 编号  
        loss_scaler: 损失缩放器，用于混合精度训练
        args: 训练参数配置（需包含 static_ratio 参数）
        
    返回:
        dict: 训练统计信息字典
    """  
    model.train(True)  
    
    # 初始化 metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # ⭐ 重要：loss_static 和 loss_dynamic 会在第一次使用时动态添加
    # 不在此处提前创建，避免 ZeroDivisionError (count=0)
    # 延迟创建策略：
    #   - 只在实际使用时才创建 meter
    #   - 确保每个 meter 的 count 至少为 1
    #   - 避免打印未使用的 meter 时除零错误
    
    header = 'Epoch: [{}]'.format(epoch)  
    print_freq = 20  

    accum_iter = args.accum_iter  
    optimizer.zero_grad()
    
    # 获取训练模式配置
    use_joint_training = getattr(args, 'use_joint_training', False)
    static_ratio = getattr(args, 'static_ratio', 0.5)
    
    if use_joint_training:
        print(f'🎯 Training Mode: Joint Training (联合训练)')
        print(f'   Alpha (static weight): {args.alpha}')
        print(f'   Beta (dynamic weight): {args.beta}')
    else:
        print(f'🎲 Training Mode: Random Alternating (随机交替)')
        print(f'   Static ratio: {static_ratio:.1%}')

    # 统计静态和动态模式的使用次数
    static_count = 0
    dynamic_count = 0
    joint_count = 0
    
    # 🔥 新增：用于Loss暴涨检测的滑动窗口
    loss_history = []
    loss_window_size = 20  # 使用最近20个batch计算平均值
    loss_spike_threshold = 3.0  # Loss暴涨阈值（超过平均值3倍）

    # ====================================================================
    # 训练循环
    # ====================================================================
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 🔥 新增：定期清理GPU内存碎片（每50个batch）
        if data_iter_step > 0 and data_iter_step % 50 == 0:
            torch.cuda.empty_cache()  

        # 学习率调度 (每个迭代调整，而不是每个 epoch)
        if data_iter_step % accum_iter == 0:  
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  
          
        # 将数据移到设备上
        # samples shape: [batch_size, 4, 3, 224, 224]
        samples = samples.to(device, non_blocking=True)  

        # ================================================================
        # ⭐ 核心改动：联合训练 vs 随机交替
        # ================================================================
        
        # 前向传播（使用混合精度）
        with torch.cuda.amp.autocast():  
            if use_joint_training:
                # ════════════════════════════════════════════════════════
                # 🎯 联合训练模式：同时计算静态和动态loss
                # ════════════════════════════════════════════════════════
                loss_static, loss_dynamic, loss_recon, loss_pred, pred, mask = model(samples, data_type='joint')
                
                # ⭐ 检查loss有效性（本地检查）
                loss_static_valid = math.isfinite(loss_static.item())
                loss_dynamic_valid = math.isfinite(loss_dynamic.item())
                local_loss_valid = loss_static_valid and loss_dynamic_valid
                
                # 🔥 关键修复：分布式同步NaN检查
                # 将本地检查结果同步到所有GPU
                if misc.is_dist_avail_and_initialized():
                    # 创建一个tensor来存储检查结果 (1=valid, 0=invalid)
                    loss_valid_tensor = torch.tensor([1.0 if local_loss_valid else 0.0], 
                                                     device=device, dtype=torch.float32)
                    # all_reduce: 如果任何GPU检测到invalid，结果会<world_size
                    dist.all_reduce(loss_valid_tensor, op=dist.ReduceOp.SUM)
                    world_size = dist.get_world_size()
                    all_losses_valid = (loss_valid_tensor.item() == world_size)
                else:
                    all_losses_valid = local_loss_valid
                
                # 如果任何GPU检测到NaN，所有GPU同步跳过这个batch
                if not all_losses_valid:
                    rank = misc.get_rank()
                    if not local_loss_valid:
                        # 保存异常数据用于调试
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_file = os.path.join(args.output_dir, f'nan_debug_rank{rank}_epoch{epoch}_step{data_iter_step}_{timestamp}.txt')
                        try:
                            with open(debug_file, 'w') as f:
                                f.write(f"Epoch: {epoch}, Step: {data_iter_step}, Rank: {rank}\n")
                                f.write(f"loss_static: {loss_static.item()}\n")
                                f.write(f"loss_dynamic: {loss_dynamic.item()}\n")
                                f.write(f"loss_recon: {loss_recon.item()}\n")
                                f.write(f"loss_pred: {loss_pred.item()}\n")
                                f.write(f"samples shape: {samples.shape}\n")
                                f.write(f"samples min: {samples.min().item()}, max: {samples.max().item()}\n")
                                f.write(f"mask sum: {mask.sum().item()}, mask shape: {mask.shape}\n")
                                f.write(f"pred shape: {pred.shape}\n")
                                f.write(f"pred min: {pred.min().item()}, max: {pred.max().item()}\n")
                        except Exception as e:
                            print(f"⚠️ Failed to save debug info: {e}")
                    
                    print(f"🔥 [Rank {misc.get_rank()}] NaN detected in batch {data_iter_step}, all GPUs skipping synchronously")
                    
                    # 🔥🔥🔥 关键修复：清空梯度防止内存泄漏
                    optimizer.zero_grad()
                    
                    # 🔥🔥🔥 关键修复：清理GPU内存碎片
                    torch.cuda.empty_cache()
                    
                    # 同步跳过：所有GPU一起continue，保持分布式同步
                    continue
                
                # 加权组合
                alpha = args.alpha
                beta = args.beta
                loss = alpha * loss_static + beta * loss_dynamic
                
                # 🔥 新增：Loss暴涨检测
                current_loss = loss.item()
                if len(loss_history) >= loss_window_size:
                    avg_loss = sum(loss_history[-loss_window_size:]) / loss_window_size
                    if current_loss > avg_loss * loss_spike_threshold:
                        print(f"⚠️ [Rank {misc.get_rank()}] Loss spike detected at batch {data_iter_step}!")
                        print(f"   Current loss: {current_loss:.4f}")
                        print(f"   Average loss (last {loss_window_size} batches): {avg_loss:.4f}")
                        print(f"   Spike ratio: {current_loss/avg_loss:.2f}x")
                        print(f"   loss_static: {loss_static.item():.4f}, loss_dynamic: {loss_dynamic.item():.4f}")
                
                # 更新loss历史记录
                loss_history.append(current_loss)
                if len(loss_history) > loss_window_size * 2:  # 保留最近40个batch的历史
                    loss_history.pop(0)
                
                # 记录详细信息
                data_type_name = 'joint'
                loss_static_value = loss_static.item()
                loss_dynamic_value = loss_dynamic.item()
                loss_recon_value = loss_recon.item()  # ⭐ 新增：重建损失
                loss_pred_value = loss_pred.item()    # ⭐ 新增：预测损失
                joint_count += 1
                
            else:
                # ════════════════════════════════════════════════════════
                # 🎲 随机交替模式：每次只计算一种loss
                # ════════════════════════════════════════════════════════
                use_static = random.random() < static_ratio
                
                if use_static:
                    # 静态模式：只使用第1帧重建
                    loss, _, _ = model(samples[:, 0], data_type=0)
                    data_type_name = 'static'
                    static_count += 1
                else:
                    # 动态模式：使用4帧重建+预测
                    loss, _, _ = model(samples, data_type=1)
                    data_type_name = 'dynamic'
                    dynamic_count += 1

        loss_value = loss.item()  

        # 检查损失是否有效（避免 NaN 或 Inf）
        if not math.isfinite(loss_value):  
            print(f"❌ Loss is {loss_value} (mode: {data_type_name}), stopping training")
            print(f"   This should not happen after NaN check. Investigate immediately!")
            sys.exit(1)  

        # 梯度累积 + 🔥 梯度裁剪
        loss /= accum_iter  
        # 🔥 增强梯度裁剪：从1.0降低到0.5，更激进地防止梯度爆炸
        # 降低阈值可以更好地抑制loss暴涨导致的数值不稳定
        loss_scaler(
            loss, optimizer, 
            clip_grad=0.3,  # ⭐⭐⭐ 进一步增强梯度裁剪：从0.5降低到0.3，防止NaN爆炸
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )  
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  

        torch.cuda.synchronize()  

        # ================================================================
        # 更新指标（总损失）
        # ================================================================
        metric_logger.update(loss=loss_value)
        
        # ================================================================
        # ⭐ 延迟创建和更新静态/动态损失 meter
        # ================================================================
        if use_joint_training:
            # 🎯 联合训练模式：记录详细的loss分解
            if 'loss_static' not in metric_logger.meters:
                metric_logger.add_meter('loss_static', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            if 'loss_dynamic' not in metric_logger.meters:
                metric_logger.add_meter('loss_dynamic', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            if 'loss_ratio' not in metric_logger.meters:
                metric_logger.add_meter('loss_ratio', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            if 'weighted_static' not in metric_logger.meters:
                metric_logger.add_meter('weighted_static', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            if 'weighted_dynamic' not in metric_logger.meters:
                metric_logger.add_meter('weighted_dynamic', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            
            metric_logger.update(loss_static=loss_static_value)
            metric_logger.update(loss_dynamic=loss_dynamic_value)
            metric_logger.update(loss_ratio=loss_static_value / (loss_dynamic_value + 1e-8))
            metric_logger.update(weighted_static=alpha * loss_static_value)
            metric_logger.update(weighted_dynamic=beta * loss_dynamic_value)
            
        else:
            # 🎲 随机交替模式：原有逻辑
            if use_static:
                if 'loss_static' not in metric_logger.meters:
                    metric_logger.add_meter('loss_static', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
                metric_logger.update(loss_static=loss_value)
            else:
                if 'loss_dynamic' not in metric_logger.meters:
                    metric_logger.add_meter('loss_dynamic', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
                metric_logger.update(loss_dynamic=loss_value)
            
        # 更新学习率
        lr = optimizer.param_groups[0]["lr"]  
        metric_logger.update(lr=lr)  

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        # ================================================================
        # ⭐ Wandb 记录：每个 batch 上传一次（仅主进程）
        # ================================================================
        if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
            # 计算全局 iteration
            global_step = data_iter_step // accum_iter + epoch * len(data_loader) // accum_iter
            
            # 基础信息
            wandb_log = {
                "batch/loss_total": loss_value_reduce,
                "batch/lr": lr,
                "batch/epoch": epoch,
                "batch/iteration": global_step,
            }
            
            # ⭐ 联合训练 vs 随机交替的不同记录方式
            if use_joint_training:
                # 🎯 联合训练：记录详细的loss分解和权重信息
                wandb_log.update({
                    "batch/loss_static": loss_static_value,
                    "batch/loss_dynamic": loss_dynamic_value,
                    "batch/loss_recon": loss_recon_value,      # ⭐ 新增：动态重建损失(前3帧)
                    "batch/loss_pred": loss_pred_value,        # ⭐ 新增：预测损失(第4帧)
                    "batch/loss_ratio": loss_static_value / (loss_dynamic_value + 1e-8),
                    "batch/weighted_static": alpha * loss_static_value,
                    "batch/weighted_dynamic": beta * loss_dynamic_value,
                    "batch/alpha": alpha,
                    "batch/beta": beta,
                    "batch/static_contribution": (alpha * loss_static_value) / loss_value_reduce,
                    "batch/dynamic_contribution": (beta * loss_dynamic_value) / loss_value_reduce,
                    "batch/mode": 2,  # 联合模式
                })
            else:
                # 🎲 随机交替：记录当前batch的模式
                if use_static:
                    wandb_log["batch/loss_static"] = loss_value_reduce
                    wandb_log["batch/mode"] = 0  # 静态模式
                else:
                    wandb_log["batch/loss_dynamic"] = loss_value_reduce
                    wandb_log["batch/mode"] = 1  # 动态模式
            
            # 上传到 Wandb
            wandb.log(wandb_log)

    # ====================================================================
    # Epoch 结束后的统计
    # ====================================================================
    
    # 同步所有进程的统计信息
    metric_logger.synchronize_between_processes()  
    
    # 打印统计信息
    print("Averaged stats:", metric_logger)
    
    # 打印模式使用统计
    if use_joint_training:
        print(f"🎯 Joint training: {joint_count} batches")
        if joint_count > 0 and 'loss_static' in metric_logger.meters and 'loss_dynamic' in metric_logger.meters:
            print(f"   Avg loss_static: {metric_logger.meters['loss_static'].global_avg:.4f}")
            print(f"   Avg loss_dynamic: {metric_logger.meters['loss_dynamic'].global_avg:.4f}")
            print(f"   Avg loss_ratio: {metric_logger.meters['loss_ratio'].global_avg:.4f}")
    else:
        total_count = static_count + dynamic_count
        if total_count > 0:
            print(f"🎲 Static mode: {static_count} batches ({static_count/total_count:.1%})")
            print(f"🎲 Dynamic mode: {dynamic_count} batches ({dynamic_count/total_count:.1%})")
        else:
            print("⚠️ Warning: No batches processed!")
    
    # 返回统计信息
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
