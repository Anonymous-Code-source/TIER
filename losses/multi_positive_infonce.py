"""
losses/multi_positive_infonce.py
Multi-Positive InfoNCE Loss - 跨传感器对齐损失

理论基础:
    I(Z_a; Z_b) ≥ E[log(p(z_b|z_a) / p(z_b))]
    
    对于 3 个传感器，每个样本有 2 个正样本
"""

import torch
import torch.nn.functional as F


def multi_positive_infonce_loss(
    z_dict,
    temperature=0.07,
    use_semi_hard_negatives=True,
    similarity_window=(0.2, 0.8),
    object_ids=None,
    frame_ids=None,
    exclude_same_object_negatives=False,
    prioritize_same_object_diff_frame_negatives=False,
    same_object_negative_keep_ratio=0.25,
    eps=1e-8,
    reduction="mean",
):
    """
    Multi-Positive InfoNCE Loss
    
    Args:
        z_dict: Dict[str, Tensor]
            键: 'gelsight', 'digit', 'duragel'
            值: [batch_size, 128] L2-normalized 表征
        temperature: float
            温度参数（默认 0.07）
        use_semi_hard_negatives: bool
            是否使用半难负样本过滤
        similarity_window: tuple
            半难负样本的相似度窗口 (min, max)
        object_ids: Optional[List[str] or List[int]]
            当前batch的对象ID列表，长度应为 batch_size
        frame_ids: Optional[List[int]]
            当前batch的基准帧ID列表，长度应为 batch_size
        exclude_same_object_negatives: bool
            是否将“同object但不同index”的样本从负样本中移除
        prioritize_same_object_diff_frame_negatives: bool
            是否优先保留“同object但不同frame”的负样本（用于强化接触位点可分性）
        same_object_negative_keep_ratio: float
            在优先保留同object异frame负样本时，额外保留的非优先负样本比例
        eps: float
            数值稳定性常数
    
    Returns:
        loss:
            - reduction="mean": Tensor (scalar)，Multi-Positive InfoNCE 损失
            - reduction="none": Tensor([B])，每个样本的平均损失（对 3 个 anchor 取平均）
    
    示例:
        >>> z_dict = {
        ...     'gelsight': torch.randn(8, 128),
        ...     'digit': torch.randn(8, 128),
        ...     'duragel': torch.randn(8, 128)
        ... }
        >>> # L2 归一化
        >>> for k in z_dict:
        ...     z_dict[k] = F.normalize(z_dict[k], dim=1, p=2)
        >>> loss = multi_positive_infonce_loss(z_dict, temperature=0.07)
        >>> print(loss)  # 预期: 0.5-2.0
    """
    sensors = ['gelsight', 'digit', 'duragel']
    batch_size = z_dict['gelsight'].shape[0]
    feature_dim = z_dict['gelsight'].shape[1]
    device = z_dict['gelsight'].device
    
    # 验证输入
    for sensor in sensors:
        assert sensor in z_dict, f"Missing sensor: {sensor}"
        assert z_dict[sensor].shape == (batch_size, feature_dim), \
            f"Expected shape ({batch_size}, {feature_dim}), got {z_dict[sensor].shape}"
    
    # 拼接所有传感器的表征 [3B, 128]
    z_all = torch.cat([z_dict[s] for s in sensors], dim=0)

    same_object_mask = None
    if exclude_same_object_negatives and object_ids is not None and len(object_ids) == batch_size:
        obj_list = [str(x) for x in object_ids]
        same_object_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                same_object_mask[i, j] = (obj_list[i] == obj_list[j])

    same_object_diff_frame_mask = None
    if (
        prioritize_same_object_diff_frame_negatives and
        object_ids is not None and len(object_ids) == batch_size and
        frame_ids is not None and len(frame_ids) == batch_size
    ):
        obj_list = [str(x) for x in object_ids]
        frame_list = [str(x) for x in frame_ids]
        same_object_diff_frame_mask = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                same_object_diff_frame_mask[i, j] = (
                    (obj_list[i] == obj_list[j]) and (frame_list[i] != frame_list[j])
                )
    
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unknown reduction: {reduction}. Must be 'mean' or 'none'.")

    total_loss = 0.0
    per_sample_loss_sum = torch.zeros(batch_size, device=device)
    num_pairs = 0
    
    # 对每个传感器作为 anchor
    for anchor_idx, anchor_sensor in enumerate(sensors):
        z_anchor = z_dict[anchor_sensor]  # [B, 128]
        
        # 计算相似度矩阵 [B, 3B]
        sim_matrix = torch.matmul(z_anchor, z_all.T) / temperature
        
        # ========================================
        # 构建正样本掩码
        # ========================================
        positive_mask = torch.zeros(batch_size, 3 * batch_size, 
                                   dtype=torch.bool, device=device)
        
        for pos_idx, pos_sensor in enumerate(sensors):
            if pos_sensor != anchor_sensor:
                # 同一 batch index 的其他传感器是正样本
                start = pos_idx * batch_size
                for i in range(batch_size):
                    positive_mask[i, start + i] = True
        
        # ========================================
        # 构建负样本掩码
        # ========================================
        negative_mask = torch.ones(batch_size, 3 * batch_size, 
                                   dtype=torch.bool, device=device)
        
        # 排除自己（对角线）
        for i in range(batch_size):
            negative_mask[i, anchor_idx * batch_size + i] = False
        
        # 排除正样本
        negative_mask = negative_mask & (~positive_mask)

        # 可选：排除同 object 的负样本，减少语义假负样本
        if same_object_mask is not None:
            for sensor_block in range(len(sensors)):
                start = sensor_block * batch_size
                end = start + batch_size
                negative_mask[:, start:end] = negative_mask[:, start:end] & (~same_object_mask)

        # 记录“基线负样本池”，若后续过滤过严导致空集，可回退。
        base_negative_mask = negative_mask.clone()

        # ========================================
        # 半难负样本过滤（可选）
        # ========================================
        if use_semi_hard_negatives:
            # 计算未缩放的相似度
            sim_values = torch.matmul(z_anchor, z_all.T)
            
            # 选择相似度在 [0.2, 0.8] 范围内的负样本
            hard_mask = (sim_values >= similarity_window[0]) & \
                       (sim_values <= similarity_window[1])
            
            # 更新负样本掩码
            negative_mask = negative_mask & hard_mask

        # 优先保留“同 object 不同 frame”负样本，用于加强接触位点可分性。
        if same_object_diff_frame_mask is not None:
            priority_mask = torch.zeros_like(negative_mask)
            for sensor_block in range(len(sensors)):
                start = sensor_block * batch_size
                end = start + batch_size
                priority_mask[:, start:end] = same_object_diff_frame_mask

            priority_neg = negative_mask & priority_mask
            if priority_neg.any():
                non_priority_neg = negative_mask & (~priority_mask)
                keep_ratio = max(0.0, min(1.0, float(same_object_negative_keep_ratio)))
                if keep_ratio <= 0.0:
                    sampled_non_priority = torch.zeros_like(non_priority_neg)
                elif keep_ratio >= 1.0:
                    sampled_non_priority = non_priority_neg
                else:
                    sampled_non_priority = non_priority_neg & (
                        torch.rand(non_priority_neg.shape, device=device) < keep_ratio
                    )
                negative_mask = priority_neg | sampled_non_priority

        # 防御式回退：保证每个anchor至少有一个负样本。
        row_has_neg = negative_mask.any(dim=1)
        if not row_has_neg.all():
            negative_mask[~row_has_neg] = base_negative_mask[~row_has_neg]
        
        # ========================================
        # 计算 InfoNCE 损失
        # ========================================
        # 正样本项: sum(exp(sim_pos))
        positive_sim = torch.exp(sim_matrix) * positive_mask
        numerator = positive_sim.sum(dim=1)  # [B]
        
        # 负样本项: sum(exp(sim_neg))
        negative_sim = torch.exp(sim_matrix) * negative_mask
        
        # 分母: numerator + sum(exp(sim_neg))
        denominator = numerator + negative_sim.sum(dim=1)
        
        # 损失: -log(numerator / denominator)
        loss = -torch.log(numerator / (denominator + eps))
        
        total_loss += loss.mean()
        per_sample_loss_sum += loss
        num_pairs += 1
    
    # 平均所有 anchor
    if reduction == "none":
        return per_sample_loss_sum / max(num_pairs, 1)
    return total_loss / max(num_pairs, 1)


def compute_similarity_statistics(z_dict):
    """
    计算相似度统计（用于监控）
    
    Args:
        z_dict: Dict[str, Tensor]
            {'gelsight': [B, 128], 'digit': [B, 128], 'duragel': [B, 128]}
    
    Returns:
        stats: Dict[str, float]
            {
                'same_contact_sim_mean': 同接触点平均相似度,
                'same_contact_sim_std': 同接触点相似度标准差,
                'diff_contact_sim_mean': 不同接触点平均相似度,
                'diff_contact_sim_std': 不同接触点相似度标准差
            }
    """
    sensors = ['gelsight', 'digit', 'duragel']
    batch_size = z_dict['gelsight'].shape[0]
    
    # 同接触点相似度（正样本）
    same_contact_sims = []
    for i, s1 in enumerate(sensors):
        for s2 in sensors[i+1:]:
            sim = (z_dict[s1] * z_dict[s2]).sum(dim=1)  # [B]
            same_contact_sims.append(sim)
    
    same_contact_sims = torch.cat(same_contact_sims)
    
    # 不同接触点相似度（负样本）
    diff_contact_sims = []
    for s in sensors:
        z = z_dict[s]  # [B, 128]
        # 计算 batch 内不同样本的相似度
        sim_matrix = torch.matmul(z, z.T)  # [B, B]
        # 提取非对角线元素
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        diff_sims = sim_matrix[mask]
        diff_contact_sims.append(diff_sims)
    
    diff_contact_sims = torch.cat(diff_contact_sims)
    
    stats = {
        'same_contact_sim_mean': same_contact_sims.mean().item(),
        'same_contact_sim_std': same_contact_sims.std().item(),
        'diff_contact_sim_mean': diff_contact_sims.mean().item(),
        'diff_contact_sim_std': diff_contact_sims.std().item(),
    }
    
    return stats
