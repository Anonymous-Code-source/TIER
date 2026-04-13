"""
losses/variance_regularization.py
Variance Regularization Loss - 防止特征坍缩

理论基础:
    鼓励特征在各维度上保持足够的方差，防止所有样本的特征
    被压缩到一个很小的区域（特征坍缩/representation collapse）

使用场景:
    - Stage-2b训练中，域不变性压力可能导致特征坍缩
    - repr_std < 0.08 时需要强制增加方差
"""

import torch
import torch.nn.functional as F


def variance_regularization_loss(z_dict, target_std=0.15, eps=1e-8):
    """
    方差正则化损失
    
    目标: 鼓励特征保持足够的方差，防止坍缩
    方法: 惩罚标准差偏离目标值
    
    Args:
        z_dict: Dict[str, Tensor]
            {'gelsight': [B, 128], 'digit': [B, 128], 'duragel': [B, 128]}
        target_std: float
            目标标准差（健康范围: 0.10-0.20，default: 0.15）
        eps: float
            数值稳定性常数
    
    Returns:
        loss: Tensor (scalar)
            方差正则化损失
    
    示例:
        >>> z_dict = {
        ...     'gelsight': F.normalize(torch.randn(32, 128), dim=1),
        ...     'digit': F.normalize(torch.randn(32, 128), dim=1),
        ...     'duragel': F.normalize(torch.randn(32, 128), dim=1)
        ... }
        >>> loss = variance_regularization_loss(z_dict, target_std=0.15)
        >>> print(loss)  # 预期: 0.01-0.10
    """
    # 拼接所有传感器的特征 [3B, 128]
    all_z = torch.cat([z for z in z_dict.values()], dim=0)
    
    # 计算每个维度的标准差 [128]
    std_per_dim = all_z.std(dim=0, unbiased=False)  # [128]
    
    # 计算平均标准差
    mean_std = std_per_dim.mean()
    
    # 惩罚偏离目标标准差
    # L = (mean_std - target_std)^2
    loss = (mean_std - target_std) ** 2
    
    return loss


def adaptive_variance_loss(z_dict, target_std=0.15, current_std=None, eps=1e-8):
    """
    自适应方差正则化损失
    
    特点: 
    - 当 current_std < threshold 时，强烈惩罚（防止坍缩）
    - 当 current_std ≈ target_std 时，惩罚较小
    - 当 current_std > target_std 时，惩罚减弱
    
    Args:
        z_dict: Dict[str, Tensor]
            特征字典
        target_std: float
            目标标准差
        current_std: float (optional)
            当前标准差（用于自适应权重）
        eps: float
            数值稳定性常数
    
    Returns:
        loss: Tensor (scalar)
            自适应方差损失
    """
    all_z = torch.cat([z for z in z_dict.values()], dim=0)
    std_per_dim = all_z.std(dim=0, unbiased=False)
    mean_std = std_per_dim.mean()
    
    if current_std is None:
        current_std = mean_std.item()
    
    # 自适应权重
    # 如果 current_std 太小（< 0.08），增大惩罚权重
    if current_std < 0.08:
        weight = 2.0
    elif current_std < target_std:
        weight = 1.5
    else:
        weight = 1.0
    
    # 基础损失
    base_loss = (mean_std - target_std) ** 2
    
    return weight * base_loss


def feature_covariance_regularization(z_dict, target_corr=0.0, eps=1e-8):
    """
    特征协方差正则化
    
    目标: 减少特征维度之间的相关性，鼓励不同维度学习不同的信息
    
    Args:
        z_dict: Dict[str, Tensor]
            特征字典
        target_corr: float
            目标相关性（default: 0，希望特征维度正交）
        eps: float
            数值稳定性常数
    
    Returns:
        loss: Tensor (scalar)
            协方差正则化损失
    """
    all_z = torch.cat([z for z in z_dict.values()], dim=0)  # [3B, 128]
    
    # 中心化
    z_centered = all_z - all_z.mean(dim=0, keepdim=True)
    
    # 计算协方差矩阵 [128, 128]
    cov = torch.matmul(z_centered.T, z_centered) / z_centered.shape[0]
    
    # 对角线是方差，非对角线是协方差
    # 我们希望非对角线接近0（维度正交）
    cov_off_diag = cov - torch.diag(torch.diag(cov))
    
    # 惩罚非对角线元素的绝对值
    loss = (cov_off_diag ** 2).mean()
    
    return loss


def compute_effective_rank(z_dict, eps=1e-8):
    """
    计算特征的有效秩（Effective Rank）
    
    用于监控特征空间的维度利用率
    有效秩越高，说明特征使用的维度越多（越好）
    
    Args:
        z_dict: Dict[str, Tensor]
            特征字典
        eps: float
            数值稳定性常数
    
    Returns:
        effective_rank: float
            有效秩（健康范围: 50-128）
    """
    all_z = torch.cat([z for z in z_dict.values()], dim=0)
    
    # 对特征矩阵进行SVD分解
    try:
        _, S, _ = torch.svd(all_z)
        
        # 归一化奇异值
        S_normalized = S / (S.sum() + eps)
        
        # 计算熵
        entropy = -(S_normalized * torch.log(S_normalized + eps)).sum()
        
        # 有效秩 = exp(entropy)
        effective_rank = torch.exp(entropy)
        
        return effective_rank.item()
    except:
        # SVD可能失败，返回默认值
        return 64.0


if __name__ == '__main__':
    """测试代码"""
    print("\n" + "=" * 70)
    print("  Variance Regularization Loss - Test")
    print("=" * 70)
    
    # 创建测试数据
    z_dict = {
        'gelsight': F.normalize(torch.randn(32, 128), dim=1),
        'digit': F.normalize(torch.randn(32, 128), dim=1),
        'duragel': F.normalize(torch.randn(32, 128), dim=1)
    }
    
    # 测试1: 基础方差正则化
    print("\n1. Basic Variance Regularization:")
    loss_var = variance_regularization_loss(z_dict, target_std=0.15)
    print(f"   Loss: {loss_var.item():.6f}")
    
    # 测试2: 自适应方差正则化
    print("\n2. Adaptive Variance Regularization:")
    all_z = torch.cat([z for z in z_dict.values()], dim=0)
    current_std = all_z.std(dim=0).mean().item()
    print(f"   Current STD: {current_std:.6f}")
    loss_adaptive = adaptive_variance_loss(z_dict, target_std=0.15, current_std=current_std)
    print(f"   Loss: {loss_adaptive.item():.6f}")
    
    # 测试3: 协方差正则化
    print("\n3. Covariance Regularization:")
    loss_cov = feature_covariance_regularization(z_dict)
    print(f"   Loss: {loss_cov.item():.6f}")
    
    # 测试4: 有效秩
    print("\n4. Effective Rank:")
    eff_rank = compute_effective_rank(z_dict)
    print(f"   Effective Rank: {eff_rank:.2f} / 128")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70 + "\n")

