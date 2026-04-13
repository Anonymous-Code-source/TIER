"""
losses/infomax_share.py
InfoMax-Share Loss - 显式去域损失

理论基础:
    最小化 I(Z; S) = E[log(p(s|z) / p(s))]
    
    实现方式: Uniform KL Divergence
    KL(p(s|z) || Uniform) → 0
"""

import math
import torch
import torch.nn.functional as F


def _kl_p_to_uniform_from_log_probs(log_probs, num_sensors=3, eps=1e-8):
    """
    计算 KL(p(s|z) || Uniform)。
    说明：F.kl_div(log_probs, uniform)默认是 KL(uniform || p)，
    与我们期望的方向相反，这里显式按 KL(p||u) 计算。
    """
    probs = log_probs.exp().clamp_min(eps)
    log_uniform = -math.log(float(num_sensors))
    return (probs * (log_probs - log_uniform)).sum(dim=1).mean()


def infomax_uniform_kl_loss(z, discriminator, num_sensors=3, eps=1e-8):
    """
    InfoMax-Share Loss (Uniform KL Divergence)
    
    目标: 使判别器无法从表征预测传感器类型
    方法: 最小化 KL(p(s|z) || Uniform([1/3, 1/3, 1/3]))
    
    Args:
        z: Tensor [batch_size, 128]
            L2-normalized 表征
        discriminator: DomainDiscriminator
            域判别器模型
        num_sensors: int
            传感器数量（默认 3）
        eps: float
            数值稳定性常数
    
    Returns:
        loss: Tensor (scalar)
            KL 散度损失
    
    示例:
        >>> from model.discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(input_dim=128, num_sensors=3)
        >>> z = torch.randn(24, 128)
        >>> z = F.normalize(z, dim=1, p=2)
        >>> loss = infomax_uniform_kl_loss(z, discriminator, num_sensors=3)
        >>> print(loss)  # 预期: 0.1-0.5
    """
    batch_size = z.shape[0]
    device = z.device
    
    # 判别器预测传感器分布 [batch_size, num_sensors]
    log_probs = discriminator(z)  # log_softmax 输出
    
    loss = _kl_p_to_uniform_from_log_probs(
        log_probs, num_sensors=num_sensors, eps=eps
    )
    return loss


def compute_domain_probe_accuracy(z_dict, discriminator):
    """
    计算域探针准确率（用于监控）- 简化版，向后兼容
    
    Args:
        z_dict: Dict[str, Tensor]
            {'gelsight': [B, 128], 'digit': [B, 128], 'duragel': [B, 128]}
        discriminator: DomainDiscriminator
            域判别器
    
    Returns:
        accuracy: float
            域探针准确率（目标: 接近 33.3%）
    """
    sensors = ['gelsight', 'digit', 'duragel']
    sensor_to_idx = {s: i for i, s in enumerate(sensors)}
    
    all_z = []
    all_labels = []
    
    for sensor, z in z_dict.items():
        all_z.append(z)
        labels = torch.full((z.shape[0],), sensor_to_idx[sensor], 
                          dtype=torch.long, device=z.device)
        all_labels.append(labels)
    
    all_z = torch.cat(all_z, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    with torch.no_grad():
        pred, _ = discriminator.predict(all_z)
        accuracy = (pred == all_labels).float().mean().item()
    
    return accuracy


def compute_domain_metrics_detailed(z_dict, discriminator):
    """
    计算详细的域判别器指标（用于检测判别器崩溃）
    
    Args:
        z_dict: Dict[str, Tensor]
            {'gelsight': [B, 128], 'digit': [B, 128], 'duragel': [B, 128]}
        discriminator: DomainDiscriminator
            域判别器
    
    Returns:
        metrics: Dict[str, float]
            {
                'domain_acc': float,  # 整体准确率
                'domain_acc_gelsight': float,  # GelSight 识别准确率
                'domain_acc_digit': float,  # DIGIT 识别准确率
                'domain_acc_duragel': float,  # DuraGel 识别准确率
                'domain_acc_std': float,  # Per-class acc 标准差
                'domain_pred_ratio_0': float,  # 预测为 GelSight 的比例
                'domain_pred_ratio_1': float,  # 预测为 DIGIT 的比例
                'domain_pred_ratio_2': float,  # 预测为 DuraGel 的比例
                'domain_entropy': float,  # 预测分布的熵
                'domain_is_collapsed': bool,  # 是否崩溃
            }
    """
    sensors = ['gelsight', 'digit', 'duragel']
    sensor_to_idx = {s: i for i, s in enumerate(sensors)}
    
    # 准备数据
    z_gelsight = z_dict['gelsight']
    z_digit = z_dict['digit']
    z_duragel = z_dict['duragel']
    
    B = z_gelsight.shape[0]
    device = z_gelsight.device
    
    with torch.no_grad():
        # 获取预测
        pred_gelsight, _ = discriminator.predict(z_gelsight)
        pred_digit, _ = discriminator.predict(z_digit)
        pred_duragel, _ = discriminator.predict(z_duragel)
        
        # ===== 1. 计算 Per-Class Accuracy =====
        acc_gelsight = (pred_gelsight == 0).float().mean().item()
        acc_digit = (pred_digit == 1).float().mean().item()
        acc_duragel = (pred_duragel == 2).float().mean().item()
        
        # 整体准确率
        all_correct = (pred_gelsight == 0).sum() + (pred_digit == 1).sum() + (pred_duragel == 2).sum()
        domain_acc = (all_correct.float() / (3 * B)).item()
        
        # ===== 2. 计算标准差（检测崩溃） =====
        per_class_accs = torch.tensor([acc_gelsight, acc_digit, acc_duragel], device=device)
        domain_acc_std = per_class_accs.std().item()
        
        # ===== 3. 计算预测分布 =====
        pred_all = torch.cat([pred_gelsight, pred_digit, pred_duragel])  # [3B]
        pred_counts = torch.bincount(pred_all, minlength=3)  # [3]
        pred_ratios = pred_counts.float() / (3 * B)
        
        pred_ratio_0 = pred_ratios[0].item()  # 预测为 GelSight 的比例
        pred_ratio_1 = pred_ratios[1].item()  # 预测为 DIGIT 的比例
        pred_ratio_2 = pred_ratios[2].item()  # 预测为 DuraGel 的比例
        
        # ===== 4. 计算熵 =====
        # 熵越高 = 越接近均匀分布（真随机）
        # 最大熵 = log(3) ≈ 1.0986
        epsilon = 1e-8
        pred_ratios_safe = pred_ratios + epsilon
        domain_entropy = -(pred_ratios_safe * torch.log(pred_ratios_safe)).sum().item()
        
        # ===== 5. 检测崩溃 =====
        # 崩溃条件：
        #   1. 标准差过大（> 0.4）
        #   2. 某个类别预测比例过高（> 0.8）
        is_collapsed = (domain_acc_std > 0.4) or (
            (pred_ratio_0 > 0.8) or (pred_ratio_1 > 0.8) or (pred_ratio_2 > 0.8)
        )
    
    return {
        'domain_acc': domain_acc,
        'domain_acc_gelsight': acc_gelsight,
        'domain_acc_digit': acc_digit,
        'domain_acc_duragel': acc_duragel,
        'domain_acc_std': domain_acc_std,
        'domain_pred_ratio_0': pred_ratio_0,
        'domain_pred_ratio_1': pred_ratio_1,
        'domain_pred_ratio_2': pred_ratio_2,
        'domain_entropy': domain_entropy,
        'domain_is_collapsed': is_collapsed,
    }


def compute_representation_std(z_dict):
    """
    计算表征标准差（防止塌缩）
    
    Args:
        z_dict: Dict[str, Tensor]
            {'gelsight': [B, 128], 'digit': [B, 128], 'duragel': [B, 128]}
    
    Returns:
        std: float
            表征的平均标准差（健康范围: > 0.1）
    """
    all_z = torch.cat([z for z in z_dict.values()], dim=0)
    std = all_z.std(dim=0).mean().item()
    return std


def infomax_uniform_kl_loss_from_logprobs(log_probs, num_sensors=3):
    """
    从已计算的log_probs计算KL散度（避免重复前向传播）
    用于对抗式训练
    
    Args:
        log_probs: Tensor [batch_size, num_sensors]
            判别器输出的log probabilities
        num_sensors: int
            传感器数量
    
    Returns:
        loss: Tensor (scalar)
    """
    return _kl_p_to_uniform_from_log_probs(
        log_probs, num_sensors=num_sensors
    )
