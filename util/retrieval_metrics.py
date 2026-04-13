"""
retrieval_metrics.py - 跨传感器检索指标计算

功能：
- 计算Recall@K指标（K=1,5,10）
- 直接反映跨传感器泛化能力
- 用于验证域不变性的实际效果

版本: v1.0
日期: 2025-12-15
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


def compute_retrieval_metrics(
    z_dict: Dict[str, torch.Tensor],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算跨传感器检索指标
    
    Args:
        z_dict: 特征字典 {'gelsight': [B,D], 'digit': [B,D], 'duragel': [B,D]}
        k_values: Recall@K的K值列表
    
    Returns:
        指标字典，包含：
        - recall@1_avg: 平均Recall@1
        - recall@5_avg: 平均Recall@5
        - recall@10_avg: 平均Recall@10
        - recall@1_gelsight_to_digit: GelSight→DIGIT的Recall@1
        - ... (所有6个方向的Recall@K)
    """
    device = z_dict['gelsight'].device
    
    # 归一化特征
    z_gelsight = F.normalize(z_dict['gelsight'], dim=1)
    z_digit = F.normalize(z_dict['digit'], dim=1)
    z_duragel = F.normalize(z_dict['duragel'], dim=1)
    
    batch_size = z_gelsight.shape[0]
    
    # 6个检索方向
    retrieval_pairs = [
        ('gelsight', 'digit', z_gelsight, z_digit),
        ('gelsight', 'duragel', z_gelsight, z_duragel),
        ('digit', 'gelsight', z_digit, z_gelsight),
        ('digit', 'duragel', z_digit, z_duragel),
        ('duragel', 'gelsight', z_duragel, z_gelsight),
        ('duragel', 'digit', z_duragel, z_digit)
    ]
    
    results = {}
    
    for source_name, target_name, z_source, z_target in retrieval_pairs:
        # 计算相似度矩阵 [B, B]
        sim_matrix = torch.mm(z_source, z_target.t())
        
        # 对于每个query，找到最相似的K个target
        for k in k_values:
            # 获取top-k索引
            _, topk_indices = torch.topk(sim_matrix, k=min(k, batch_size), dim=1)
            
            # 计算Recall@K（正确的target索引就是query索引）
            correct_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            hits = (topk_indices == correct_indices).any(dim=1).float()
            recall_at_k = hits.mean().item()
            
            # 保存结果
            key = f'recall@{k}_{source_name}_to_{target_name}'
            results[key] = recall_at_k
    
    # 计算平均Recall@K
    for k in k_values:
        recall_values = [
            results[f'recall@{k}_{src}_to_{tgt}']
            for src, tgt, _, _ in retrieval_pairs
        ]
        results[f'recall@{k}_avg'] = np.mean(recall_values)
    
    return results


def compute_similarity_matrix(
    z_source: torch.Tensor,
    z_target: torch.Tensor
) -> torch.Tensor:
    """
    计算归一化的相似度矩阵
    
    Args:
        z_source: 源域特征 [B, D]
        z_target: 目标域特征 [B, D]
    
    Returns:
        相似度矩阵 [B, B]
    """
    z_source = F.normalize(z_source, dim=1)
    z_target = F.normalize(z_target, dim=1)
    return torch.mm(z_source, z_target.t())


def compute_mean_reciprocal_rank(
    z_dict: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    计算平均倒数排名（MRR）
    
    Args:
        z_dict: 特征字典
    
    Returns:
        MRR指标字典
    """
    device = z_dict['gelsight'].device
    
    z_gelsight = F.normalize(z_dict['gelsight'], dim=1)
    z_digit = F.normalize(z_dict['digit'], dim=1)
    z_duragel = F.normalize(z_dict['duragel'], dim=1)
    
    batch_size = z_gelsight.shape[0]
    
    retrieval_pairs = [
        ('gelsight', 'digit', z_gelsight, z_digit),
        ('gelsight', 'duragel', z_gelsight, z_duragel),
        ('digit', 'gelsight', z_digit, z_gelsight),
        ('digit', 'duragel', z_digit, z_duragel),
        ('duragel', 'gelsight', z_duragel, z_gelsight),
        ('duragel', 'digit', z_duragel, z_digit)
    ]
    
    results = {}
    
    for source_name, target_name, z_source, z_target in retrieval_pairs:
        sim_matrix = torch.mm(z_source, z_target.t())
        
        # 对每个query，找到正确target的排名
        correct_indices = torch.arange(batch_size, device=device)
        
        # 获取排序后的索引
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        
        # 找到正确索引的位置（排名）
        ranks = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            rank = (sorted_indices[i] == correct_indices[i]).nonzero(as_tuple=True)[0].item()
            ranks[i] = rank + 1  # 排名从1开始
        
        # 计算MRR
        mrr = (1.0 / ranks).mean().item()
        
        key = f'mrr_{source_name}_to_{target_name}'
        results[key] = mrr
    
    # 平均MRR
    mrr_values = [results[f'mrr_{src}_to_{tgt}'] for src, tgt, _, _ in retrieval_pairs]
    results['mrr_avg'] = np.mean(mrr_values)
    
    return results
