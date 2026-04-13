"""
domain_probe.py - 独立域探针（Independent Domain Probe）

功能：
- 提供独立的域分类器，用于评估encoder的真实域不变性
- 避免"自证正确"陷阱（用训练中的判别器评估）
- 实现"冻结encoder + 训练新分类器"的评估范式

版本: v1.0
日期: 2025-12-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class IndependentDomainProbe(nn.Module):
    """
    独立域探针
    
    用于评估encoder是否真正学到域不变性：
    - 冻结encoder，提取所有特征
    - 训练一个新的分类器（probe）来预测域标签
    - 如果probe准确率高（>40%），说明特征中仍包含域信息
    - 如果probe准确率低（≈33%），说明特征真正域不变
    
    v2改进: 支持线性探针模式（更严格的评估）
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_sensors: int = 3,
        dropout: float = 0.0,
        use_linear_probe: bool = False  # 新增：是否使用线性探针
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sensors = num_sensors
        self.use_linear_probe = use_linear_probe
        
        if use_linear_probe:
            # 线性探针：更严格的评估，更难拟合域信息
            self.probe = nn.Linear(input_dim, num_sensors)
        else:
            # 原始MLP探针
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_sensors)
            )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 表征向量 [B, input_dim]
        
        Returns:
            logits: 域分类logits [B, num_sensors]
        """
        return self.probe(z)


def train_independent_probe(
    z_dict: Dict[str, torch.Tensor],
    probe: IndependentDomainProbe,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 20,
    batch_size: int = 256,
    early_stop_patience: int = 3,
    early_stop_min_delta: float = 1e-4,
    shuffle: bool = True,
    device: torch.device = None
) -> Dict[str, float]:
    """
    训练独立域探针
    
    Args:
        z_dict: 特征字典 {'gelsight': [B,D], 'digit': [B,D], 'duragel': [B,D]}
        probe: 独立探针模型
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
    
    Returns:
        训练统计信息
    """
    if device is None:
        device = next(probe.parameters()).device
    
    probe.train()
    
    # 准备数据（按域拼接，标签固定为 0/1/2）
    z_all = torch.cat([z_dict['gelsight'], z_dict['digit'], z_dict['duragel']], dim=0).to(device)
    per_domain = z_dict['gelsight'].shape[0]
    labels = torch.cat([
        torch.zeros(per_domain, dtype=torch.long, device=device),
        torch.ones(per_domain, dtype=torch.long, device=device),
        torch.full((per_domain,), 2, dtype=torch.long, device=device)
    ], dim=0)
    
    # mini-batch 训练（更接近标准probe范式）
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(z_all, labels)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        drop_last=False
    )
    
    best_epoch_loss = None
    patience_left = early_stop_patience
    epoch_losses = []
    
    for _ in range(num_epochs):
        total_loss = 0.0
        total_count = 0
        
        for z_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(z_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_count = y_batch.numel()
            total_loss += loss.item() * batch_count
            total_count += batch_count
        
        epoch_loss = total_loss / max(total_count, 1)
        epoch_losses.append(epoch_loss)
        
        if best_epoch_loss is None or (best_epoch_loss - epoch_loss) > early_stop_min_delta:
            best_epoch_loss = epoch_loss
            patience_left = early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    
    return {
        'train_loss': float(epoch_losses[-1]) if epoch_losses else 0.0,
        'best_train_loss': float(best_epoch_loss) if best_epoch_loss is not None else 0.0,
        'epochs_ran': len(epoch_losses)
    }


@torch.no_grad()
def evaluate_independent_probe(
    z_dict: Dict[str, torch.Tensor],
    probe: IndependentDomainProbe
) -> Dict[str, float]:
    """
    评估独立域探针
    
    Args:
        z_dict: 特征字典
        probe: 训练好的探针
    
    Returns:
        评估指标字典
    """
    probe.eval()
    device = next(probe.parameters()).device
    
    # 准备数据
    z_all = torch.cat([z_dict['gelsight'], z_dict['digit'], z_dict['duragel']], dim=0).to(device)
    batch_size = z_dict['gelsight'].shape[0]
    labels = torch.cat([
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.ones(batch_size, dtype=torch.long, device=device),
        torch.full((batch_size,), 2, dtype=torch.long, device=device)
    ])
    
    # 预测
    logits = probe(z_all)
    preds = logits.argmax(dim=1)
    
    # loss（用于诊断probe是否学到了东西）
    loss = F.cross_entropy(logits, labels).item()
    
    # 计算准确率
    acc = (preds == labels).float().mean().item()
    
    # Per-class准确率
    acc_gelsight = (preds[:batch_size] == 0).float().mean().item()
    acc_digit = (preds[batch_size:2*batch_size] == 1).float().mean().item()
    acc_duragel = (preds[2*batch_size:] == 2).float().mean().item()
    
    # 标准差
    acc_std = torch.tensor([acc_gelsight, acc_digit, acc_duragel]).std().item()
    
    # 预测分布
    pred_counts = torch.bincount(preds, minlength=3).float()
    pred_ratios = pred_counts / pred_counts.sum()
    
    # 熵（基于预测分布，而非softmax概率）
    probs = F.softmax(logits, dim=1).mean(dim=0)
    entropy_softmax = -(probs * torch.log(probs + 1e-8)).sum().item()
    
    # 基于pred_ratios的熵（更准确反映argmax结果）
    pred_ratios_safe = pred_ratios + 1e-8
    entropy_pred = -(pred_ratios_safe * torch.log(pred_ratios_safe)).sum().item()
    
    # 增强的崩溃检测（照搬discriminator逻辑）
    # 1. 单类预测占比>80%
    max_pred_ratio = pred_ratios.max().item()
    collapse_by_ratio = max_pred_ratio > 0.8
    
    # 2. 标准差过小
    collapse_by_std = acc_std < 0.01
    
    # 3. 熵过低（使用pred_ratios的熵）
    collapse_by_entropy = entropy_pred < 0.5
    
    # 综合判断
    is_collapsed = collapse_by_ratio or collapse_by_std or collapse_by_entropy
    
    # 崩溃比率（0-1，越高越崩溃）
    collapse_ratio = 0.0
    if collapse_by_ratio:
        collapse_ratio = max(collapse_ratio, max_pred_ratio)
    if collapse_by_std:
        collapse_ratio = max(collapse_ratio, 1.0 - acc_std * 10)  # std<0.01 -> ratio>0.9
    if collapse_by_entropy:
        collapse_ratio = max(collapse_ratio, 1.0 - entropy_pred / 1.099)  # max_entropy=log(3)
    
    return {
        'probe_loss': loss,
        'probe_acc': acc,
        'probe_acc_gelsight': acc_gelsight,
        'probe_acc_digit': acc_digit,
        'probe_acc_duragel': acc_duragel,
        'probe_acc_std': acc_std,
        'probe_pred_ratio_0': pred_ratios[0].item(),
        'probe_pred_ratio_1': pred_ratios[1].item(),
        'probe_pred_ratio_2': pred_ratios[2].item(),
        'probe_entropy': entropy_pred,  # 使用pred_ratios的熵
        'probe_entropy_softmax': entropy_softmax,  # 保留softmax熵用于对比
        'probe_is_collapsed': is_collapsed,
        'probe_collapse_ratio': collapse_ratio,  # 新增：崩溃程度
        'probe_collapse_by_ratio': collapse_by_ratio,  # 新增：单类预测检测
        'probe_collapse_by_std': collapse_by_std,  # 新增：标准差检测
        'probe_collapse_by_entropy': collapse_by_entropy,  # 新增：熵检测
        'probe_max_pred_ratio': max_pred_ratio  # 新增：最大预测比例
    }
