"""
Stage2_engine.py - Stage-2 跨传感器无感表征学习训练引擎

功能:
- train_one_epoch(): 单个 epoch 的训练循环
- validate(): 验证集评估
- 支持 Multi-Positive InfoNCE Loss + InfoMax-Share Loss
- 支持监控指标（域探针准确率、相似度统计、表征标准差）

版本: v1.0
日期: 2025-11-01
"""

import math
import sys
import time
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

import util.misc as misc
import util.lr_sched as lr_sched

from losses.multi_positive_infonce import (
    multi_positive_infonce_loss,
    compute_similarity_statistics
)
from losses.infomax_share import (
    compute_domain_probe_accuracy,
    compute_domain_metrics_detailed,
    compute_representation_std
)
from losses.variance_regularization import (
    adaptive_variance_loss,
    compute_effective_rank
)

def _linear_schedule(start_value, end_value, progress):
    progress = max(0.0, min(1.0, progress))
    return start_value + (end_value - start_value) * progress


def _get_similarity_window(args, epoch):
    if getattr(args, 'dynamic_similarity_window', False):
        total_epochs = max(1, int(getattr(args, 'epochs', 1)))
        progress = epoch / max(total_epochs - 1, 1)
        low = _linear_schedule(args.similarity_window_low_start, args.similarity_window_low_end, progress)
        high = _linear_schedule(args.similarity_window_high_start, args.similarity_window_high_end, progress)
        return (low, high)
    return (args.similarity_window_low, args.similarity_window_high)


def _get_sensor_token_drop_prob(args, epoch):
    total_epochs = max(1, int(getattr(args, 'epochs', 1)))
    progress = epoch / max(total_epochs - 1, 1)
    return _linear_schedule(
        getattr(args, 'sensor_token_drop_start', 0.0),
        getattr(args, 'sensor_token_drop_end', 0.0),
        progress
    )


def _get_lambda_match(args, epoch):
    base_lambda = float(getattr(args, 'lambda_match', 0.0))
    if base_lambda <= 0:
        return 0.0
    warmup_ratio = float(getattr(args, 'match_warmup_ratio', 0.0))
    rampup_ratio = float(getattr(args, 'match_rampup_ratio', 0.0))
    total_epochs = max(1, int(getattr(args, 'epochs', 1)))
    warmup_epochs = int(total_epochs * warmup_ratio)
    rampup_epochs = int(total_epochs * rampup_ratio)
    if epoch < warmup_epochs:
        return 0.0
    if rampup_epochs <= 0:
        return base_lambda
    if epoch < warmup_epochs + rampup_epochs:
        progress = (epoch - warmup_epochs) / rampup_epochs
        return _linear_schedule(0.0, base_lambda, progress)
    return base_lambda


def _resolve_adversarial_feature_target(args, epoch=None):
    """
    解析当前epoch应使用的对抗分支。

    默认使用 args.adversarial_feature_target。
    若启用 phase2_adv_target_switch，则在指定epoch进度后切到 phase2_adv_target_late。
    """
    base_target = getattr(args, 'adversarial_feature_target', 'domain')
    if getattr(args, 'stage', None) != '2b':
        return base_target
    if not bool(getattr(args, 'phase2_adv_target_switch', False)):
        return base_target

    total_epochs = max(1, int(getattr(args, 'epochs', 1)))
    if epoch is None:
        epoch = int(getattr(args, 'current_epoch', 0))
    progress = float(epoch) / max(total_epochs - 1, 1)
    switch_ratio = float(getattr(args, 'phase2_adv_target_switch_ratio', 0.5))
    late_target = getattr(args, 'phase2_adv_target_late', 'contact_residual')
    if progress >= switch_ratio:
        return late_target
    return base_target


def _get_im_contact_ratio(args, epoch):
    """
    Dual-target InfoMax混合系数:
    loss_im = r * loss_im_contact + (1-r) * loss_im_residual
    """
    start = float(getattr(args, 'im_contact_ratio_start', 0.8))
    end = float(getattr(args, 'im_contact_ratio_end', 0.4))
    total_epochs = max(1, int(getattr(args, 'epochs', 1)))
    progress = float(epoch) / max(total_epochs - 1, 1)
    return _linear_schedule(start, end, progress)


def _build_digit_focus_weights(
    domain_labels: torch.Tensor,
    num_sensors: int = 3,
    use_digit_focus: bool = False,
    digit_focus_weight: float = 1.0,
):
    """
    构建逐样本域损失重加权向量：
    默认 [1, 1, 1]，启用后为 [1, digit_focus_weight, 1]。
    """
    if not use_digit_focus:
        return None
    focus_w = float(digit_focus_weight)
    if (not math.isfinite(focus_w)) or abs(focus_w - 1.0) < 1e-8:
        return None
    class_w = torch.ones((int(num_sensors),), device=domain_labels.device, dtype=torch.float32)
    if int(num_sensors) > 1:
        class_w[1] = focus_w
    return class_w[domain_labels]


def _combine_sample_weights(base_weights=None, extra_weights=None):
    """
    融合两路样本权重。shape不一致时回退到base_weights，保证稳定。
    """
    if base_weights is None:
        return extra_weights
    if extra_weights is None:
        return base_weights
    if base_weights.numel() != extra_weights.numel():
        return base_weights
    return base_weights * extra_weights


def _init_dynamic_loss_balance_state(args):
    """
    初始化辅助损失EMA动态平衡器。
    """
    return {
        'enabled': bool(getattr(args, 'enable_dynamic_loss_balance', False)),
        'ema_decay': float(getattr(args, 'loss_balance_ema_decay', 0.98)),
        'eps': float(getattr(args, 'loss_balance_eps', 1e-6)),
        'min_factor': float(getattr(args, 'loss_balance_min_factor', 0.2)),
        'max_factor': float(getattr(args, 'loss_balance_max_factor', 5.0)),
        'warmup_steps': int(getattr(args, 'loss_balance_warmup_steps', 20)),
        'step': 0,
        'ema': {}
    }


def _compute_dynamic_balance_factors(balance_state, raw_losses):
    """
    使用loss_mp作为参考量纲，对辅助项进行EMA自适应缩放。
    """
    factors = {key: 1.0 for key in raw_losses.keys()}
    if not balance_state.get('enabled', False):
        return factors

    ema = balance_state['ema']
    ema_decay = balance_state['ema_decay']
    eps = balance_state['eps']
    for key, value in raw_losses.items():
        v = abs(float(value))
        if not math.isfinite(v):
            v = 0.0
        if key not in ema:
            ema[key] = v
        else:
            ema[key] = ema_decay * ema[key] + (1.0 - ema_decay) * v

    step = int(balance_state.get('step', 0))
    balance_state['step'] = step + 1
    if step < int(balance_state.get('warmup_steps', 0)):
        return factors

    ref = max(float(ema.get('loss_mp', 0.0)), eps)
    min_factor = float(balance_state.get('min_factor', 0.2))
    max_factor = float(balance_state.get('max_factor', 5.0))
    for key in raw_losses.keys():
        if key == 'loss_mp':
            continue
        denom = max(float(ema.get(key, 0.0)), eps)
        ratio = ref / denom
        ratio = max(min_factor, min(max_factor, ratio))
        factors[key] = ratio
    return factors


def _build_sensor_type_tensor(sensor_id, batch_size, device, drop_prob=0.0):
    sensor_type = torch.full((batch_size,), sensor_id, dtype=torch.long, device=device)
    if drop_prob > 0.0 and batch_size > 0:
        drop_mask = torch.bernoulli(torch.full((batch_size,), drop_prob, device=device)).bool()
        # -1 作为 unknown token（映射到最后一个 sensor token）
        sensor_type = sensor_type.masked_fill(drop_mask, -1)
    return sensor_type


def _prepare_stage2_video_input(x: torch.Tensor, num_frames: int = 3) -> torch.Tensor:
    """
    Stage-2 输入规范化:
    - DataLoader输出: [B, T, C, H, W]
    - 模型期望输入: [B, C, T, H, W]
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape={getattr(x, 'shape', None)}")
    t = max(1, min(int(num_frames), int(x.shape[1])))
    x = x[:, :t, :, :, :]
    return x.permute(0, 2, 1, 3, 4).contiguous()


def _compute_matching_loss(
    z_dict,
    match_head,
    num_negatives=1,
    object_ids=None,
    hard_negative_topk=16,
    require_diff_object=True,
):
    if match_head is None:
        return torch.tensor(0.0, device=next(iter(z_dict.values())).device)

    sensors = ['gelsight', 'digit', 'duragel']
    pairs = [('gelsight', 'digit'), ('gelsight', 'duragel'), ('digit', 'duragel')]
    device = next(iter(z_dict.values())).device
    bsize = z_dict[sensors[0]].shape[0]
    if bsize <= 1:
        return torch.tensor(0.0, device=device)

    bce = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_pairs = 0

    diff_object_mask = None
    if object_ids is not None and len(object_ids) == bsize:
        obj_list = [str(x) for x in object_ids]
        diff_object_mask = torch.zeros((bsize, bsize), dtype=torch.bool, device=device)
        for i in range(bsize):
            for j in range(bsize):
                diff_object_mask[i, j] = (obj_list[i] != obj_list[j])

    for s1, s2 in pairs:
        z1 = z_dict[s1]
        z2 = z_dict[s2]
        sim_matrix = torch.matmul(z1, z2.T)

        pos_vec = z1 * z2
        pos_logits = match_head(pos_vec).squeeze(1)
        pos_labels = torch.ones_like(pos_logits)

        neg_logits_list = []
        topk = max(1, int(hard_negative_topk))
        neg_per_anchor = max(1, int(num_negatives))
        for i in range(bsize):
            candidate_mask = torch.ones(bsize, dtype=torch.bool, device=device)
            candidate_mask[i] = False  # 排除同index正样本
            if require_diff_object and diff_object_mask is not None:
                candidate_mask = candidate_mask & diff_object_mask[i]
            if not candidate_mask.any():
                candidate_mask = torch.ones(bsize, dtype=torch.bool, device=device)
                candidate_mask[i] = False
            if not candidate_mask.any():
                continue

            candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            candidate_sim = sim_matrix[i, candidate_idx]
            k = min(topk, candidate_idx.numel())
            hard_idx_local = torch.topk(candidate_sim, k=k, largest=True).indices
            hard_idx = candidate_idx[hard_idx_local]

            num_pick = min(neg_per_anchor, hard_idx.numel())
            perm = torch.randperm(hard_idx.numel(), device=device)[:num_pick]
            picked_idx = hard_idx[perm]

            anchor = z1[i].unsqueeze(0).expand(num_pick, -1)
            neg_vec = anchor * z2[picked_idx]
            neg_logits_list.append(match_head(neg_vec).squeeze(1))

        if len(neg_logits_list) == 0:
            continue

        neg_logits = torch.cat(neg_logits_list, dim=0)
        neg_labels = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        total_loss += bce(logits, labels)
        num_pairs += 1

    return total_loss / max(1, num_pairs)


def _compute_rowwise_pair_loss(
    z_contact_dict,
    pair_loss_type='cosine',
    pair_confidence=None,
    use_pair_confidence_weighting=True,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
):
    """
    Row-wise 跨传感器配对拉近损失。
    假设同一 batch index 对应同一接触（来自时间对齐后的三传感器采样）。

    Args:
        z_contact_dict: Dict[str, Tensor], 每个传感器 [B, D]
        pair_loss_type: 'cosine' or 'mse'

    Args:
        pair_confidence: Optional[Tensor/array], [B] 对齐置信度
        use_pair_confidence_weighting: 是否使用置信度加权
        confidence_floor: 置信度下限，避免权重过小
        confidence_power: 权重幂指数
        confidence_min: 仅保留置信度>=该阈值的样本，过滤低质量配对

    Returns:
        Tensor(scalar)
    """
    sensors = ['gelsight', 'digit', 'duragel']
    pairs = [('gelsight', 'digit'), ('gelsight', 'duragel'), ('digit', 'duragel')]
    device = z_contact_dict[sensors[0]].device
    losses = []
    confidence = None
    weights = None
    valid_mask = None

    if pair_confidence is not None:
        if isinstance(pair_confidence, torch.Tensor):
            confidence = pair_confidence.to(device=device, dtype=torch.float32).view(-1)
        else:
            confidence = torch.as_tensor(pair_confidence, dtype=torch.float32, device=device).view(-1)
        if confidence_min > 0:
            valid_mask = confidence >= float(confidence_min)
            if valid_mask.sum().item() == 0:
                return torch.tensor(0.0, device=device)

    if use_pair_confidence_weighting and confidence is not None:
        weights = torch.clamp(confidence, min=float(confidence_floor), max=1.0)
        weights = weights.pow(float(confidence_power))

    for s1, s2 in pairs:
        x = z_contact_dict[s1]
        y = z_contact_dict[s2]
        if x.shape[0] == 0:
            continue

        if pair_loss_type == 'cosine':
            per_sample = 1.0 - F.cosine_similarity(x, y, dim=1)
        elif pair_loss_type == 'mse':
            per_sample = ((x - y) ** 2).mean(dim=1)
        else:
            raise ValueError(f"Unsupported pair_loss_type: {pair_loss_type}")

        local_per_sample = per_sample
        local_weights = weights
        if valid_mask is not None and valid_mask.numel() == per_sample.numel():
            if valid_mask.sum().item() == 0:
                continue
            local_per_sample = per_sample[valid_mask]
            if local_weights is not None:
                local_weights = local_weights[valid_mask]

        if local_weights is not None and local_weights.numel() == local_per_sample.numel():
            denom = torch.clamp(local_weights.sum(), min=1e-6)
            pair_loss = (local_per_sample * local_weights).sum() / denom
        else:
            pair_loss = local_per_sample.mean()
        losses.append(pair_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _compute_rowwise_pair_hard_tail_loss(
    z_contact_dict,
    pair_loss_type='cosine',
    pair_confidence=None,
    use_pair_confidence_weighting=True,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
    tail_ratio=0.2,
    tail_min_samples=16,
):
    """
    Row-wise hard-tail配对损失：
    仅优化每个传感器对中“最难”的top样本，优先压缩长尾错配。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    pairs = [('gelsight', 'digit'), ('gelsight', 'duragel'), ('digit', 'duragel')]
    device = z_contact_dict[sensors[0]].device
    losses = []
    confidence = None
    weights = None
    valid_mask = None
    tail_ratio = max(1e-6, min(1.0, float(tail_ratio)))
    tail_min_samples = max(1, int(tail_min_samples))

    if pair_confidence is not None:
        if isinstance(pair_confidence, torch.Tensor):
            confidence = pair_confidence.to(device=device, dtype=torch.float32).view(-1)
        else:
            confidence = torch.as_tensor(pair_confidence, dtype=torch.float32, device=device).view(-1)
        if confidence_min > 0:
            valid_mask = confidence >= float(confidence_min)
            if valid_mask.sum().item() == 0:
                return torch.tensor(0.0, device=device)

    if use_pair_confidence_weighting and confidence is not None:
        weights = torch.clamp(confidence, min=float(confidence_floor), max=1.0)
        weights = weights.pow(float(confidence_power))

    for s1, s2 in pairs:
        x = z_contact_dict[s1]
        y = z_contact_dict[s2]
        if x.shape[0] == 0:
            continue

        if pair_loss_type == 'cosine':
            per_sample = 1.0 - F.cosine_similarity(x, y, dim=1)
        elif pair_loss_type == 'mse':
            per_sample = ((x - y) ** 2).mean(dim=1)
        else:
            raise ValueError(f"Unsupported pair_loss_type: {pair_loss_type}")

        local_per_sample = per_sample
        local_weights = weights
        if valid_mask is not None and valid_mask.numel() == per_sample.numel():
            if valid_mask.sum().item() == 0:
                continue
            local_per_sample = per_sample[valid_mask]
            if local_weights is not None:
                local_weights = local_weights[valid_mask]

        num_samples = int(local_per_sample.numel())
        if num_samples == 0:
            continue
        k = max(tail_min_samples, int(math.ceil(num_samples * tail_ratio)))
        k = min(k, num_samples)
        if local_weights is not None and local_weights.numel() == local_per_sample.numel():
            # 用“加权难度”排序，避免低置信噪声样本占据tail。
            hard_score = local_per_sample * local_weights
        else:
            hard_score = local_per_sample
        hard_score = torch.nan_to_num(hard_score, nan=0.0, posinf=1e6, neginf=0.0)
        hard_idx = torch.topk(hard_score, k=k, largest=True).indices
        tail_loss = local_per_sample[hard_idx]
        tail_weights = (
            local_weights[hard_idx]
            if local_weights is not None and local_weights.numel() == local_per_sample.numel()
            else None
        )
        losses.append(_weighted_mean(tail_loss, weights=tail_weights))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _compute_residual_compact_loss(
    z_contact_dict,
    pair_confidence=None,
    use_pair_confidence_weighting=True,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
):
    """
    同contact内残差紧致损失：
    对每个row，惩罚三传感器相对row中心的残差幅值。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    device = z_contact_dict[sensors[0]].device
    z_stack = torch.stack([z_contact_dict[s] for s in sensors], dim=0)  # [3, B, D]
    if z_stack.shape[1] == 0:
        return torch.tensor(0.0, device=device)

    row_center = z_stack.mean(dim=0, keepdim=True)  # [1, B, D]
    residual = z_stack - row_center                 # [3, B, D]
    per_sensor_norm = torch.norm(residual, p=2, dim=2)  # [3, B]
    per_row = per_sensor_norm.mean(dim=0)  # [B]

    conf_weights, conf_valid = _build_pair_confidence_weights(
        pair_confidence,
        device=device,
        confidence_floor=confidence_floor,
        confidence_power=confidence_power,
        confidence_min=confidence_min
    )
    if not use_pair_confidence_weighting:
        conf_weights = None
    return _weighted_mean(per_row, weights=conf_weights, valid_mask=conf_valid)


def _build_pair_confidence_weights(
    pair_confidence,
    device,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
):
    """
    将 pair_confidence 转成权重与有效样本掩码。
    Returns:
        weights: Optional[Tensor[B]]
        valid_mask: Optional[BoolTensor[B]]
    """
    if pair_confidence is None:
        return None, None
    if isinstance(pair_confidence, torch.Tensor):
        conf = pair_confidence.to(device=device, dtype=torch.float32).view(-1)
    else:
        conf = torch.as_tensor(pair_confidence, dtype=torch.float32, device=device).view(-1)
    if conf.numel() == 0:
        return None, None
    finite_mask = torch.isfinite(conf)
    # 防御性处理：将NaN/Inf清理到有限区间，避免污染损失。
    conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
    valid_mask = None
    if (not bool(finite_mask.all().item())) or confidence_min > 0:
        valid_mask = finite_mask
        if confidence_min > 0:
            valid_mask = valid_mask & (conf >= float(confidence_min))
    weights = torch.clamp(conf, min=float(confidence_floor), max=1.0).pow(float(confidence_power))
    return weights, valid_mask


def _weighted_mean(values, weights=None, valid_mask=None, eps=1e-6):
    """
    对向量 values 做可选掩码/加权平均。
    """
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    x = values
    w = weights
    if valid_mask is not None:
        if valid_mask.numel() != x.numel():
            if not getattr(_weighted_mean, '_warned_mask_mismatch', False):
                if misc.is_main_process():
                    print(
                        '⚠️  [_weighted_mean] valid_mask length mismatch, '
                        'fallback to unmasked mean (check pair_confidence pipeline).'
                    )
                _weighted_mean._warned_mask_mismatch = True
            return x.mean()
        if valid_mask.sum().item() == 0:
            return torch.tensor(0.0, device=x.device)
        x = x[valid_mask]
        if w is not None and w.numel() == valid_mask.numel():
            w = w[valid_mask]
    if x.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    if w is not None and w.numel() == x.numel():
        denom = torch.clamp(w.sum(), min=eps)
        return (x * w).sum() / denom
    return x.mean()


def _infomax_kl_per_sample_from_log_probs(log_probs, num_sensors=3, eps=1e-8):
    """
    逐样本 KL(p(s|z) || Uniform) 向量，shape=[N]。
    """
    probs = log_probs.exp().clamp_min(eps)
    log_uniform = -math.log(float(num_sensors))
    return (probs * (log_probs - log_uniform)).sum(dim=1)


def _compute_inter_contact_separation_loss(
    z_contact_dict,
    object_ids=None,
    frame_ids=None,
    pair_confidence=None,
    use_confidence_weighting=False,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
    sim_margin=0.15,
    hard_topk=512,
):
    """
    接触团间分离损失（L_inter）：
    - 先把同一row的三传感器特征求中心得到 contact centroid
    - 再约束不同contact之间的相似度不超过 margin

    loss = mean( relu(sim(c_i, c_j) - sim_margin) ), i != j
    仅对 hardest top-k (最高相似) 负对计算，可提升稳定性。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    device = z_contact_dict[sensors[0]].device
    batch_size = z_contact_dict[sensors[0]].shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    # [3, B, D] -> [B, D]
    z_stack = torch.stack([z_contact_dict[s] for s in sensors], dim=0)
    centroids = F.normalize(z_stack.mean(dim=0), dim=1)
    sim_matrix = torch.matmul(centroids, centroids.T)

    # 默认只排除对角线；若有pair_id则排除同contact重复项
    valid_negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    if object_ids is not None or frame_ids is not None:
        pair_ids = _build_pair_ids(object_ids, frame_ids, batch_size)
        same_pair = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            pi = pair_ids[i]
            for j in range(batch_size):
                if i != j and pair_ids[j] == pi:
                    same_pair[i, j] = True
        valid_negative_mask = valid_negative_mask & (~same_pair)

    conf_weights, conf_valid = _build_pair_confidence_weights(
        pair_confidence,
        device=device,
        confidence_floor=confidence_floor,
        confidence_power=confidence_power,
        confidence_min=confidence_min,
    )
    pair_weights_matrix = None
    if conf_valid is not None and conf_valid.numel() == batch_size:
        valid_negative_mask = valid_negative_mask & conf_valid.unsqueeze(0) & conf_valid.unsqueeze(1)
    if use_confidence_weighting and conf_weights is not None and conf_weights.numel() == batch_size:
        pair_weights_matrix = torch.sqrt(conf_weights.unsqueeze(0) * conf_weights.unsqueeze(1))

    valid_sims = sim_matrix[valid_negative_mask]
    valid_pair_weights = pair_weights_matrix[valid_negative_mask] if pair_weights_matrix is not None else None
    if valid_sims.numel() == 0:
        return torch.tensor(0.0, device=device)

    topk = int(hard_topk)
    if topk > 0 and valid_sims.numel() > topk:
        topk_idx = torch.topk(valid_sims, k=topk, largest=True).indices
        valid_sims = valid_sims[topk_idx]
        if valid_pair_weights is not None:
            valid_pair_weights = valid_pair_weights[topk_idx]

    hinge = F.relu(valid_sims - float(sim_margin))
    loss_inter = _weighted_mean(hinge, weights=valid_pair_weights)
    return loss_inter


def _compute_inter_contact_separation_loss_per_sensor(
    z_contact_dict,
    object_ids=None,
    frame_ids=None,
    pair_confidence=None,
    use_confidence_weighting=False,
    confidence_floor=0.05,
    confidence_power=1.0,
    confidence_min=0.0,
    sim_margin=0.15,
    hard_topk=512,
):
    """
    传感器内团间分离损失（L_inter_sensor）：
    - 对每个传感器分支分别做“不同contact相似度不超过margin”
    - 再对3个传感器求平均，避免“每个传感器各自成岛”而缺少团间约束
    """
    sensors = ['gelsight', 'digit', 'duragel']
    device = z_contact_dict[sensors[0]].device
    batch_size = z_contact_dict[sensors[0]].shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    # 默认只排除对角线；若有pair_id则排除同contact重复项
    valid_negative_mask_base = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    if object_ids is not None or frame_ids is not None:
        pair_ids = _build_pair_ids(object_ids, frame_ids, batch_size)
        same_pair = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            pi = pair_ids[i]
            for j in range(batch_size):
                if i != j and pair_ids[j] == pi:
                    same_pair[i, j] = True
        valid_negative_mask_base = valid_negative_mask_base & (~same_pair)

    conf_weights, conf_valid = _build_pair_confidence_weights(
        pair_confidence,
        device=device,
        confidence_floor=confidence_floor,
        confidence_power=confidence_power,
        confidence_min=confidence_min,
    )
    pair_weights_matrix = None
    if use_confidence_weighting and conf_weights is not None and conf_weights.numel() == batch_size:
        pair_weights_matrix = torch.sqrt(conf_weights.unsqueeze(0) * conf_weights.unsqueeze(1))

    per_sensor_losses = []
    for sensor in sensors:
        feats = F.normalize(z_contact_dict[sensor], dim=1)
        sim_matrix = torch.matmul(feats, feats.T)

        valid_negative_mask = valid_negative_mask_base
        if conf_valid is not None and conf_valid.numel() == batch_size:
            valid_negative_mask = valid_negative_mask & conf_valid.unsqueeze(0) & conf_valid.unsqueeze(1)

        valid_sims = sim_matrix[valid_negative_mask]
        valid_pair_weights = (
            pair_weights_matrix[valid_negative_mask] if pair_weights_matrix is not None else None
        )
        if valid_sims.numel() == 0:
            continue

        topk = int(hard_topk)
        if topk > 0 and valid_sims.numel() > topk:
            topk_idx = torch.topk(valid_sims, k=topk, largest=True).indices
            valid_sims = valid_sims[topk_idx]
            if valid_pair_weights is not None:
                valid_pair_weights = valid_pair_weights[topk_idx]

        hinge = F.relu(valid_sims - float(sim_margin))
        per_sensor_losses.append(_weighted_mean(hinge, weights=valid_pair_weights))

    if len(per_sensor_losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(per_sensor_losses).mean()


def _compute_decoupling_loss(z_contact_dict, z_domain_dict, eps=1e-6, detach_domain=False):
    """
    通过跨协方差惩罚推动 z_contact 与 z_domain 解耦。

    Args:
        z_contact_dict: Dict[str, Tensor], 各传感器 contact 表征
        z_domain_dict: Dict[str, Tensor], 各传感器 domain 表征
        eps: 数值稳定常数

    Returns:
        Tensor(scalar): 解耦损失，越小越好
    """
    z_contact = torch.cat([z_contact_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0)
    z_domain = torch.cat([z_domain_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0)
    if detach_domain:
        # 单向解耦：仅让 z_contact 远离 z_domain，避免反向破坏域分支表征
        z_domain = z_domain.detach()

    if z_contact.shape[0] <= 1:
        return torch.tensor(0.0, device=z_contact.device)

    zc = z_contact - z_contact.mean(dim=0, keepdim=True)
    zd = z_domain - z_domain.mean(dim=0, keepdim=True)
    zc = zc / (zc.std(dim=0, keepdim=True) + eps)
    zd = zd / (zd.std(dim=0, keepdim=True) + eps)

    cross_corr = torch.matmul(zc.transpose(0, 1), zd) / zc.shape[0]
    return (cross_corr ** 2).mean()


def _compute_sensor_coral_loss(z_contact_dict, include_mean=True, eps=1e-6):
    """
    在 z_contact 上做跨传感器分布对齐（CORAL）。
    目标：直接削弱“按传感器成团”的统计偏差。

    Args:
        z_contact_dict: Dict[str, Tensor], 各传感器 contact 表征 [B, D]
        include_mean: 是否加入均值对齐项
        eps: 数值稳定常数

    Returns:
        Tensor(scalar): 越小表示跨传感器分布越一致
    """
    sensors = ['gelsight', 'digit', 'duragel']
    pairs = [('gelsight', 'digit'), ('gelsight', 'duragel'), ('digit', 'duragel')]
    device = z_contact_dict[sensors[0]].device
    losses = []

    for s1, s2 in pairs:
        x = z_contact_dict[s1].float()
        y = z_contact_dict[s2].float()
        if x.shape[0] <= 1 or y.shape[0] <= 1:
            continue

        # 协方差对齐
        xc = x - x.mean(dim=0, keepdim=True)
        yc = y - y.mean(dim=0, keepdim=True)
        cov_x = torch.matmul(xc.transpose(0, 1), xc) / max(x.shape[0] - 1, 1)
        cov_y = torch.matmul(yc.transpose(0, 1), yc) / max(y.shape[0] - 1, 1)
        loss_cov = ((cov_x - cov_y) ** 2).mean()

        if include_mean:
            loss_mean = ((x.mean(dim=0) - y.mean(dim=0)) ** 2).mean()
            losses.append(loss_cov + loss_mean)
        else:
            losses.append(loss_cov)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _compute_contact_residual_coral_loss(z_contact_dict, include_mean=False):
    """
    在同一接触条件下做跨传感器对齐：
    先减去每个pair的跨传感器均值，再在残差空间做CORAL。
    """
    # CORAL残差分支使用未归一化残差，避免数值量级被压平导致损失接近0。
    residual_dict = _compute_contact_residual_dict(z_contact_dict, normalize=False)
    return _compute_sensor_coral_loss(residual_dict, include_mean=include_mean)


def _compute_pair_link_length_tensor(z_contact_dict, eps=1e-8):
    """
    计算同一pair内三传感器两两距离均值，返回 [B]。
    """
    zg = z_contact_dict['gelsight']
    zd = z_contact_dict['digit']
    zu = z_contact_dict['duragel']
    d_gd = torch.sqrt(torch.clamp(((zg - zd) ** 2).sum(dim=1), min=eps))
    d_gu = torch.sqrt(torch.clamp(((zg - zu) ** 2).sum(dim=1), min=eps))
    d_du = torch.sqrt(torch.clamp(((zd - zu) ** 2).sum(dim=1), min=eps))
    return (d_gd + d_gu + d_du) / 3.0


def _build_pair_ids(object_ids, frame_ids, batch_size):
    """
    构建每个样本的pair_id（object|frame），用于条件化评估与分组split。
    """
    if object_ids is None:
        obj_list = [f'obj_{i}' for i in range(batch_size)]
    else:
        if isinstance(object_ids, torch.Tensor):
            obj_list = [str(int(x)) for x in object_ids.detach().cpu().tolist()]
        else:
            obj_list = [str(x) for x in list(object_ids)]
        if len(obj_list) != batch_size:
            obj_list = (obj_list + [obj_list[-1] if len(obj_list) > 0 else 'obj'])[:batch_size]

    if frame_ids is None:
        frm_list = [0 for _ in range(batch_size)]
    else:
        if isinstance(frame_ids, torch.Tensor):
            frm_list = [int(x) for x in frame_ids.detach().cpu().tolist()]
        else:
            frm_list = [int(x) for x in list(frame_ids)]
        if len(frm_list) != batch_size:
            frm_list = (frm_list + [frm_list[-1] if len(frm_list) > 0 else 0])[:batch_size]

    return [f'{obj_list[i]}|{frm_list[i]}' for i in range(batch_size)]


def _build_group_splits(pair_ids: List[str], train_ratio: float, num_splits: int, seed_base: int):
    """
    按pair_id分组切分 train/test，确保同一接触pair不会同时落入训练和测试。
    """
    import numpy as np

    pair_ids_arr = np.asarray(pair_ids)
    unique_pairs = np.unique(pair_ids_arr)
    n_pairs = unique_pairs.shape[0]
    if n_pairs <= 1:
        # 退化情形：无法做分组切分，回退到样本切分
        all_idx = np.arange(pair_ids_arr.shape[0])
        split = max(1, min(int(len(all_idx) * train_ratio), len(all_idx) - 1))
        return [(all_idx[:split], all_idx[split:])]

    splits = []
    for split_id in range(max(1, int(num_splits))):
        rng = np.random.default_rng(seed_base + split_id)
        perm_pairs = unique_pairs.copy()
        rng.shuffle(perm_pairs)
        n_train_pairs = max(1, min(int(n_pairs * train_ratio), n_pairs - 1))
        train_pairs = set(perm_pairs[:n_train_pairs].tolist())
        train_mask = np.array([pid in train_pairs for pid in pair_ids_arr], dtype=bool)
        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(~train_mask)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))

    if len(splits) == 0:
        all_idx = np.arange(pair_ids_arr.shape[0])
        split = max(1, min(int(len(all_idx) * train_ratio), len(all_idx) - 1))
        splits = [(all_idx[:split], all_idx[split:])]
    return splits


def _gather_variable_length_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    DDP下聚合变长Tensor（第一维可变），单卡下原样返回。
    """
    if not misc.is_dist_avail_and_initialized():
        return t

    import torch.distributed as dist

    world_size = dist.get_world_size()
    backend = dist.get_backend()
    moved_from_cpu = False
    original_device = t.device
    # NCCL不支持CPU tensor通信；若输入在CPU，临时搬到当前GPU聚合，再搬回CPU。
    if backend == 'nccl' and (not t.is_cuda):
        t = t.to(torch.device('cuda', torch.cuda.current_device()))
        moved_from_cpu = True

    local_n = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
    ns = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(ns, local_n)
    ns = [int(x.item()) for x in ns]
    max_n = max(ns)

    if t.shape[0] < max_n:
        pad_shape = (max_n - t.shape[0],) + tuple(t.shape[1:])
        pad = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
        t = torch.cat([t, pad], dim=0)

    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    gathered = [gathered[i][:ns[i]] for i in range(world_size)]
    out = torch.cat(gathered, dim=0)
    if moved_from_cpu:
        out = out.to(original_device)
    return out


def _evaluate_conditional_probe_grouped(
    z_dict: Dict[str, torch.Tensor],
    pair_ids: List[str],
    probe_config: Dict[str, object],
    device: torch.device,
    train_ratio: float,
    num_splits: int,
    seed_base: int,
    probe_train_epochs: int,
    probe_batch_size: int,
    probe_early_stop_patience: int,
    probe_early_stop_min_delta: float,
    probe_lr: float,
    probe_weight_decay: float,
) -> Dict[str, float]:
    """
    条件probe：按pair_id分组切分（同一接触pair不会同时进train/test）。
    返回:
      {
        'acc', 'acc_std', 'used_splits',
        'acc_gelsight', 'acc_digit', 'acc_duragel',
        'acc_gelsight_std', 'acc_digit_std', 'acc_duragel_std'
      }
    """
    def _empty():
        return {
            'acc': 0.0,
            'acc_std': 0.0,
            'used_splits': 0,
            'acc_gelsight': 0.0,
            'acc_digit': 0.0,
            'acc_duragel': 0.0,
            'acc_gelsight_std': 0.0,
            'acc_digit_std': 0.0,
            'acc_duragel_std': 0.0,
        }

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    def _std(xs: List[float]) -> float:
        if len(xs) <= 1:
            return 0.0
        mu = _mean(xs)
        return float((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

    if len(pair_ids) == 0:
        return _empty()
    n = z_dict['gelsight'].shape[0]
    if n <= 1 or len(pair_ids) != n:
        return _empty()

    from model.domain_probe import (
        IndependentDomainProbe,
        train_independent_probe,
        evaluate_independent_probe
    )

    splits = _build_group_splits(
        pair_ids=pair_ids,
        train_ratio=train_ratio,
        num_splits=num_splits,
        seed_base=seed_base
    )
    if len(splits) == 0:
        return _empty()

    accs: List[float] = []
    accs_gelsight: List[float] = []
    accs_digit: List[float] = []
    accs_duragel: List[float] = []
    for split_id, (train_idx_np, test_idx_np) in enumerate(splits):
        if len(train_idx_np) == 0 or len(test_idx_np) == 0:
            continue
        index_device = z_dict['gelsight'].device
        train_idx = torch.as_tensor(train_idx_np, device=index_device, dtype=torch.long)
        test_idx = torch.as_tensor(test_idx_np, device=index_device, dtype=torch.long)

        train_z = {
            'gelsight': z_dict['gelsight'][train_idx],
            'digit': z_dict['digit'][train_idx],
            'duragel': z_dict['duragel'][train_idx],
        }
        test_z = {
            'gelsight': z_dict['gelsight'][test_idx],
            'digit': z_dict['digit'][test_idx],
            'duragel': z_dict['duragel'][test_idx],
        }

        probe = IndependentDomainProbe(**probe_config).to(device)
        probe_optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=probe_lr,
            weight_decay=probe_weight_decay
        )

        with torch.enable_grad():
            train_independent_probe(
                train_z,
                probe,
                probe_optimizer,
                num_epochs=probe_train_epochs,
                batch_size=probe_batch_size,
                early_stop_patience=probe_early_stop_patience,
                early_stop_min_delta=probe_early_stop_min_delta,
                shuffle=True
            )
        probe_metrics = evaluate_independent_probe(test_z, probe)
        accs.append(float(probe_metrics.get('probe_acc', 0.0)))
        accs_gelsight.append(float(probe_metrics.get('probe_acc_gelsight', 0.0)))
        accs_digit.append(float(probe_metrics.get('probe_acc_digit', 0.0)))
        accs_duragel.append(float(probe_metrics.get('probe_acc_duragel', 0.0)))

    if len(accs) == 0:
        return _empty()
    return {
        'acc': _mean(accs),
        'acc_std': _std(accs),
        'used_splits': len(accs),
        'acc_gelsight': _mean(accs_gelsight),
        'acc_digit': _mean(accs_digit),
        'acc_duragel': _mean(accs_duragel),
        'acc_gelsight_std': _std(accs_gelsight),
        'acc_digit_std': _std(accs_digit),
        'acc_duragel_std': _std(accs_duragel),
    }


def _quantile_from_sorted(sorted_tensor: torch.Tensor, q: float):
    if sorted_tensor.numel() == 0:
        return 0.0
    if sorted_tensor.numel() == 1:
        return float(sorted_tensor[0].item())
    q = float(max(0.0, min(1.0, q)))
    idx = int(round(q * (sorted_tensor.numel() - 1)))
    return float(sorted_tensor[idx].item())

def _compute_contact_residual_dict(z_contact_dict, eps=1e-6, normalize=False):
    """
    条件去域特征：去除同一接触(同batch index)的跨传感器共享均值，仅保留传感器残差。

    该分支用于域对抗时，可减少对“接触语义主成分”的误伤。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    z_stack = torch.stack([z_contact_dict[s] for s in sensors], dim=0)  # [3, B, D]
    row_mean = z_stack.mean(dim=0, keepdim=False)  # [B, D]
    residual = {}
    for sensor in sensors:
        res = z_contact_dict[sensor] - row_mean
        if normalize:
            res = F.normalize(res, dim=1, eps=eps)
        residual[sensor] = res
    return residual


def _compute_sensor_centroid_distance(z_dict):
    """
    传感器质心平均距离（越小表示全局传感器偏移越弱）。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    centroids = []
    for sensor in sensors:
        z = z_dict.get(sensor, None)
        if z is None or z.numel() == 0:
            continue
        centroids.append(z.mean(dim=0))
    if len(centroids) < 2:
        return None
    centroid_tensor = torch.stack(centroids, dim=0)
    pairwise = torch.pdist(centroid_tensor, p=2)
    if pairwise.numel() == 0:
        return None
    return float(pairwise.mean().item())


def _compute_cross_sensor_knn_mix_rate(
    z_dict,
    k=20,
    max_samples_per_sensor=512,
    chunk_size=256,
    seed=0,
):
    """
    计算跨传感器kNN混合率（越高越好）。
    仅用于验证监控，默认对每个传感器子采样，避免O(N^2)开销过大。
    """
    sensors = ['gelsight', 'digit', 'duragel']
    features = []
    labels = []
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))

    for sensor_id, sensor in enumerate(sensors):
        z = z_dict.get(sensor, None)
        if z is None or z.numel() == 0:
            continue
        z_cpu = z.detach().float().cpu()
        if max_samples_per_sensor is not None and max_samples_per_sensor > 0 and z_cpu.shape[0] > max_samples_per_sensor:
            idx = torch.randperm(z_cpu.shape[0], generator=generator)[:max_samples_per_sensor]
            z_cpu = z_cpu[idx]
        features.append(z_cpu)
        labels.append(torch.full((z_cpu.shape[0],), sensor_id, dtype=torch.long))

    if len(features) < 2:
        return None

    x = torch.cat(features, dim=0)  # [N, D]
    y = torch.cat(labels, dim=0)    # [N]
    n = x.shape[0]
    if n <= 1:
        return None

    k_eff = min(max(1, int(k)), n - 1)
    x = F.normalize(x, dim=1)
    mixed_sum = 0.0
    mixed_count = 0

    for start in range(0, n, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), n)
        q = x[start:end]  # [C, D]
        sims = torch.matmul(q, x.T)  # [C, N]
        local_diag = torch.arange(end - start, dtype=torch.long)
        global_diag = torch.arange(start, end, dtype=torch.long)
        sims[local_diag, global_diag] = -1e9
        topk_idx = torch.topk(sims, k=k_eff, dim=1, largest=True).indices  # [C, k]
        q_labels = y[start:end].unsqueeze(1)  # [C, 1]
        neigh_labels = y[topk_idx]  # [C, k]
        mixed = (neigh_labels != q_labels).float()
        mixed_sum += float(mixed.sum().item())
        mixed_count += int(mixed.numel())

    if mixed_count <= 0:
        return None
    return float(mixed_sum / mixed_count)


def _get_adversarial_feature_dict(z_contact_dict, z_domain_dict, target='domain'):
    """
    选择域对抗/InfoMax作用的特征分支。

    - domain: 对 z_domain 施加去域（旧行为）
    - contact: 对 z_contact 施加去域（最终输出分支）
    - contact_residual: 对 z_contact 的“去均值残差”施加去域（条件去域）
    """
    if target == 'contact':
        return z_contact_dict
    if target == 'contact_residual':
        return _compute_contact_residual_dict(z_contact_dict, normalize=False)
    if target == 'domain':
        return z_domain_dict
    raise ValueError(f"Unsupported adversarial_feature_target: {target}")

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args=None,
    wandb_log=True
):
    """
    训练一个 epoch（Stage-2 跨传感器对齐）
    
    Args:
        model: Stage2Model 实例
        data_loader: Stage-2 配对数据加载器
        optimizer: 优化器
        device: 训练设备 (cuda/cpu)
        epoch: 当前 epoch 编号
        loss_scaler: 混合精度训练的损失缩放器
        args: 训练参数配置
        wandb_log: 是否使用 Wandb 记录
        
    Returns:
        dict: 训练统计信息
    """
    model.train(True)
    
    # 🔧 FIX: 设置DistributedSampler的epoch（PyTorch分布式训练必需）
    # 确保每个epoch使用不同的随机种子，避免DataLoader挂起
    if hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
        if misc.is_main_process():
            print(f'✅ Set sampler epoch to {epoch}')
    
    # 初始化 metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_mp', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im_contact', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im_residual', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_var', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_match', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_pair', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_pair_tail', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_inter', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_inter_sensor', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_residual_compact', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_decouple', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_domain_cls', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_coral', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_coral_residual', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('pair_conf', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('bal_im', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_match', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_pair', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_pair_tail', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_inter', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_inter_sensor', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_residual_compact', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_decouple', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_domain_cls', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_coral', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    metric_logger.add_meter('bal_coral_residual', misc.SmoothedValue(window_size=20, fmt='{global_avg:.3f}'))
    
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    
    accum_iter = getattr(args, 'accum_iter', 1)
    
    # 🆕 支持字典形式的optimizer（对抗式训练）
    if isinstance(optimizer, dict):
        optimizer['encoder'].zero_grad()
        optimizer['discriminator'].zero_grad()
    else:
        optimizer.zero_grad()
    
    # 获取训练配置
    lambda_im = getattr(args, 'lambda_im', 0.0)  # InfoMax-Share 权重
    im_loss_scale = float(getattr(args, 'im_loss_scale', 1.0))
    im_dual_target = bool(getattr(args, 'im_dual_target', False))
    im_contact_ratio = _get_im_contact_ratio(args, epoch) if im_dual_target else 1.0
    temperature = getattr(args, 'temperature', 0.07)
    use_semi_hard = getattr(args, 'use_semi_hard_negatives', True)
    similarity_window = _get_similarity_window(args, epoch)
    use_sensor_token = getattr(args, 'stage2_use_sensor_token', False)
    sensor_token_drop_prob = _get_sensor_token_drop_prob(args, epoch) if use_sensor_token else 0.0
    lambda_match = _get_lambda_match(args, epoch)
    match_negatives_per_pair = getattr(args, 'match_negatives_per_pair', 1)
    match_hard_negative_topk = getattr(args, 'matching_hard_negative_topk', 16)
    enable_matching_in_phase2 = getattr(args, 'enable_matching_in_phase2', False)
    lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
    lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
    pair_tail_ratio = float(getattr(args, 'pair_tail_ratio', 0.2))
    pair_tail_min_samples = int(getattr(args, 'pair_tail_min_samples', 16))
    lambda_inter = float(getattr(args, 'lambda_inter', 0.0))
    lambda_inter_sensor = float(getattr(args, 'lambda_inter_sensor', 0.0))
    lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
    inter_sim_margin = float(getattr(args, 'inter_sim_margin', 0.15))
    inter_hard_topk = int(getattr(args, 'inter_hard_topk', 512))
    pair_loss_type = getattr(args, 'pair_loss_type', 'cosine')
    use_pair_confidence_weighting = bool(getattr(args, 'use_pair_confidence_weighting', True))
    pair_confidence_floor = float(getattr(args, 'pair_confidence_floor', 0.05))
    pair_confidence_power = float(getattr(args, 'pair_confidence_power', 1.0))
    pair_confidence_min = float(getattr(args, 'pair_confidence_min', 0.0))
    use_pair_confidence_for_domain_and_inter = bool(
        getattr(args, 'use_pair_confidence_for_domain_and_inter', False)
    )
    lambda_decouple = float(getattr(args, 'lambda_decouple', 0.0))
    lambda_domain_cls = float(getattr(args, 'lambda_domain_cls', 0.0))
    lambda_coral = float(getattr(args, 'lambda_coral', 0.0))
    lambda_coral_residual = float(getattr(args, 'lambda_coral_residual', 0.0))
    coral_include_mean = bool(getattr(args, 'coral_include_mean', True))
    coral_residual_include_mean = bool(getattr(args, 'coral_residual_include_mean', False))
    decouple_detach_domain = bool(getattr(args, 'decouple_detach_domain', True))
    domain_cls_detach_encoder = bool(getattr(args, 'domain_cls_detach_encoder', True))
    exclude_same_object_negatives = bool(getattr(args, 'exclude_same_object_negatives', False))
    prioritize_same_object_diff_frame_negatives = bool(
        getattr(args, 'prioritize_same_object_diff_frame_negatives', False)
    )
    same_object_negative_keep_ratio = float(getattr(args, 'same_object_negative_keep_ratio', 0.25))
    if prioritize_same_object_diff_frame_negatives and exclude_same_object_negatives:
        if misc.is_main_process():
            print(
                "⚠️  prioritize_same_object_diff_frame_negatives=True 与 "
                "exclude_same_object_negatives=True 冲突，自动关闭后者。"
            )
        exclude_same_object_negatives = False
    adversarial_feature_target = _resolve_adversarial_feature_target(args, epoch)
    use_digit_focus_reweight = bool(getattr(args, 'use_digit_focus_reweight', False))
    digit_focus_weight = float(getattr(args, 'digit_focus_weight', 1.0))
    disc_update_every = int(getattr(args, 'disc_update_every', 1))
    disc_steps_per_iter = int(getattr(args, 'disc_steps_per_iter', 1))
    stage2_num_frames = int(getattr(args, 'stage2_num_frames', 3))
    if disc_update_every < 1:
        disc_update_every = 1
    if disc_steps_per_iter < 1:
        disc_steps_per_iter = 1
    dynamic_balance_state = _init_dynamic_loss_balance_state(args)
    
    print(f'🎯 Stage-2 Training Configuration:')
    print(f'   Temperature: {temperature}')
    print(f'   Lambda_im: {lambda_im}')
    print(f'   IM loss scale: {im_loss_scale}')
    print(
        f'   IM dual target: {im_dual_target} '
        f'(contact_ratio={im_contact_ratio:.3f})'
    )
    print(f'   Semi-hard negatives: {use_semi_hard}')
    print(f'   Similarity window: {similarity_window[0]:.2f}-{similarity_window[1]:.2f}')
    print(f'   Matching loss: {lambda_match}')
    print(f'   Row-wise pair loss (lambda_pair): {lambda_pair} (type={pair_loss_type})')
    print(
        f'   Row-wise hard-tail pair loss (lambda_pair_tail): {lambda_pair_tail} '
        f'(ratio={pair_tail_ratio:.2f}, min_samples={pair_tail_min_samples})'
    )
    print(
        f'   Inter-contact separation loss (lambda_inter): {lambda_inter} '
        f'(margin={inter_sim_margin:.3f}, hard_topk={inter_hard_topk})'
    )
    print(
        f'   Inter-contact separation per sensor (lambda_inter_sensor): {lambda_inter_sensor} '
        f'(margin={inter_sim_margin:.3f}, hard_topk={inter_hard_topk})'
    )
    print(f'   Residual compact loss (lambda_residual_compact): {lambda_residual_compact}')
    print(
        f'   Pair confidence weighting: {use_pair_confidence_weighting} '
        f'(floor={pair_confidence_floor:.2f}, power={pair_confidence_power:.2f}, '
        f'min={pair_confidence_min:.2f})'
    )
    print(
        f'   Confidence-guided domain/inter: {use_pair_confidence_for_domain_and_inter}'
    )
    print(f'   Decoupling loss (lambda_decouple): {lambda_decouple}')
    print(f'   Decoupling single-direction (detach domain): {decouple_detach_domain}')
    print(f'   Domain supervision loss (lambda_domain_cls): {lambda_domain_cls}')
    print(f'   CORAL alignment loss (lambda_coral): {lambda_coral} (include_mean={coral_include_mean})')
    print(
        f'   Conditional CORAL loss (lambda_coral_residual): {lambda_coral_residual} '
        f'(include_mean={coral_residual_include_mean})'
    )
    print(f'   Domain cls detach encoder: {domain_cls_detach_encoder}')
    print(f'   Exclude same-object negatives: {exclude_same_object_negatives}')
    print(
        f'   Prioritize same-object diff-frame negatives: '
        f'{prioritize_same_object_diff_frame_negatives} (keep_ratio={same_object_negative_keep_ratio:.2f})'
    )
    print(
        f'   Adversarial target: {adversarial_feature_target} '
        f'(base={getattr(args, "adversarial_feature_target", "domain")}, '
        f'switch={getattr(args, "phase2_adv_target_switch", False)}, '
        f'ratio={getattr(args, "phase2_adv_target_switch_ratio", 0.5):.2f})'
    )
    print(
        f'   DIGIT focus reweight: {use_digit_focus_reweight} '
        f'(digit_weight={digit_focus_weight:.3f})'
    )
    print(
        f'   Input frame policy: stage2_num_frames={stage2_num_frames} '
        f'([B,T,C,H,W] -> [B,C,T,H,W])'
    )
    print(f'   Discriminator update every: {disc_update_every} iter')
    print(f'   Discriminator steps per iter: {disc_steps_per_iter}')
    print(
        f'   Dynamic loss balance: {dynamic_balance_state["enabled"]} '
        f'(ema={dynamic_balance_state["ema_decay"]:.3f}, '
        f'factor={dynamic_balance_state["min_factor"]:.2f}~{dynamic_balance_state["max_factor"]:.2f}, '
        f'warmup_steps={dynamic_balance_state["warmup_steps"]})'
    )
    print(f'   Sensor token: {use_sensor_token} (drop_prob={sensor_token_drop_prob:.2f})')
    print(f'   Gradient accumulation: {accum_iter}')
    
    # 用于累积监控指标
    total_domain_acc = 0.0
    total_repr_std = 0.0
    total_same_sim = 0.0
    total_diff_sim = 0.0
    metric_count = 0
    
    # 详细域判别器指标累积
    total_domain_acc_gelsight = 0.0
    total_domain_acc_digit = 0.0
    total_domain_acc_duragel = 0.0
    total_domain_acc_std = 0.0
    total_domain_pred_ratio_0 = 0.0
    total_domain_pred_ratio_1 = 0.0
    total_domain_pred_ratio_2 = 0.0
    total_domain_entropy = 0.0
    domain_collapse_count = 0
    probe_collapse_score = 0.0
    
    # Group-DRO 权重初始化
    use_group_dro = getattr(args, 'use_group_dro', False)
    group_weight_max_observed = 0.0
    group_weight_min_observed = 1.0
    group_weight_saturation_steps = 0
    group_weight_saturation_warn = getattr(args, 'group_weight_saturation_warn', 80)
    group_weight_saturation_margin = getattr(args, 'group_weight_saturation_margin', 0.02)

    if use_group_dro:
        group_weights = torch.full((3,), 1 / 3, device=device)
        group_dro_eta = getattr(args, 'group_dro_eta', 0.01)
        eta_cap = getattr(args, 'group_dro_eta_cap', 0.05)
        disable_eta_cap = getattr(args, 'group_dro_disable_eta_cap', False)
        # 标准化loss下的大eta会导致指数更新失控，可按需限制或放开
        if getattr(args, 'group_dro_normalize_losses', False):
            if disable_eta_cap:
                print(f'   Group-DRO: η cap disabled, using raw η={group_dro_eta}')
            elif group_dro_eta > eta_cap:
                print(f'   ⚠️  Group-DRO η={group_dro_eta} 偏大，自动截断为 {eta_cap}')
                group_dro_eta = eta_cap
        print(f'   Group-DRO: Enabled (η={group_dro_eta})')
        group_dro_min_weight = getattr(args, 'group_dro_min_weight', 0.0)
        group_dro_max_weight = getattr(args, 'group_dro_max_weight', 1.0)
        group_weight_max_observed = group_weights.max().item()
        group_weight_min_observed = group_weights.min().item()
    
    # 用于累积组损失
    total_loss_fine = 0.0
    total_loss_indoor = 0.0
    total_loss_outdoor = 0.0
    group_count = 0
    
    # ====================================================================
    # 训练循环
    # ====================================================================
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 学习率调度
        if data_iter_step % accum_iter == 0:
            # 🆕 支持字典形式的optimizer
            if isinstance(optimizer, dict):
                lr_sched.adjust_learning_rate(
                    optimizer['encoder'],
                    data_iter_step / len(data_loader) + epoch,
                    args
                )
            else:
                lr_sched.adjust_learning_rate(
                    optimizer,
                    data_iter_step / len(data_loader) + epoch,
                    args
                )
        
        # 将数据移到设备
        # batch: {'gelsight': [B,4,3,224,224], 'digit': [...], 'duragel': [...]}
        for sensor in ['gelsight', 'digit', 'duragel']:
            batch[sensor] = batch[sensor].to(device, non_blocking=True)
        
        # ================================================================
        # 前向传播（混合精度）
        # ================================================================
        with torch.cuda.amp.autocast():
            # 提取表征
            z_contact_dict = {}
            z_domain_dict = {}
            h_dict = {}
            sensor_id_map = {'gelsight': 0, 'digit': 1, 'duragel': 2}
            for sensor in ['gelsight', 'digit', 'duragel']:
                # 统一Stage-2输入策略: [B,T,C,H,W] -> [B,C,T,H,W]
                imgs = _prepare_stage2_video_input(
                    batch[sensor],
                    num_frames=stage2_num_frames
                )
                if use_sensor_token:
                    sensor_type = _build_sensor_type_tensor(
                        sensor_id_map[sensor],
                        imgs.shape[0],
                        device,
                        drop_prob=sensor_token_drop_prob
                    )
                else:
                    sensor_type = None
                outputs = model(
                    imgs,
                    sensor_type=sensor_type,
                    return_all=True,
                    return_encoder_output=True
                )
                z_contact_dict[sensor] = outputs['z_contact']
                z_domain_dict[sensor] = outputs['z_domain']
                h_dict[sensor] = outputs['h']

            z_adv_dict = _get_adversarial_feature_dict(
                z_contact_dict,
                z_domain_dict,
                target=adversarial_feature_target
            )
        
        object_ids = batch.get('object', None)
        frame_ids = batch.get('frame', None)
        pair_confidence = batch.get('pair_confidence', None)
        if isinstance(pair_confidence, torch.Tensor):
            pair_confidence = pair_confidence.to(device=device, dtype=torch.float32).view(-1)
            pair_confidence_mean = pair_confidence.mean().item()
        else:
            pair_confidence_mean = 1.0
        domain_sample_weights = None
        domain_valid_mask = None
        if use_pair_confidence_for_domain_and_inter:
            conf_w, conf_valid = _build_pair_confidence_weights(
                pair_confidence,
                device=device,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min,
            )
            if conf_w is not None:
                domain_sample_weights = conf_w.repeat(3)
            if conf_valid is not None:
                domain_valid_mask = conf_valid.repeat(3)

        # 准备域标签/对抗分支特征（在autocast外）
        use_adversarial = getattr(args, 'use_adversarial_training', False)
        num_sensors = int(getattr(args, 'num_sensors', 3))
        batch_size = z_domain_dict['gelsight'].shape[0]
        domain_labels = torch.cat([
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.long, device=device),
            torch.full((batch_size,), 2, dtype=torch.long, device=device)
        ])
        digit_focus_weights = _build_digit_focus_weights(
            domain_labels=domain_labels,
            num_sensors=num_sensors,
            use_digit_focus=use_digit_focus_reweight,
            digit_focus_weight=digit_focus_weight,
        )
        domain_loss_weights = _combine_sample_weights(
            domain_sample_weights,
            digit_focus_weights
        )
        z_all_for_adversarial = None
        z_all_contact_for_adversarial = None
        z_all_residual_for_adversarial = None
        if lambda_im > 0 and model.discriminator is not None:
            if im_dual_target and args.stage == '2b':
                z_all_contact_for_adversarial = torch.cat(
                    [z_contact_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
                z_residual_dict = _compute_contact_residual_dict(z_contact_dict, normalize=False)
                z_all_residual_for_adversarial = torch.cat(
                    [z_residual_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
            else:
                z_all_for_adversarial = torch.cat(
                    [z_adv_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )

        # 关键修复1：先更新判别器，再计算encoder对抗损失（避免“旧判别器梯度”）
        if (
            (z_all_for_adversarial is not None or (
                z_all_contact_for_adversarial is not None and
                z_all_residual_for_adversarial is not None
            )) and
            use_adversarial and
            args.stage == '2b' and
            isinstance(optimizer, dict)
        ):
            if (data_iter_step + 1) % disc_update_every == 0:
                for _ in range(disc_steps_per_iter):
                    with torch.cuda.amp.autocast():
                        if (
                            im_dual_target and
                            z_all_contact_for_adversarial is not None and
                            z_all_residual_for_adversarial is not None
                        ):
                            # dual-target: 先偏向contact，后期转向residual
                            log_probs_disc_contact = model.discriminator(
                                z_all_contact_for_adversarial.detach()
                            )
                            per_sample_nll_contact = F.nll_loss(
                                log_probs_disc_contact,
                                domain_labels,
                                reduction='none'
                            )
                            loss_disc_contact = _weighted_mean(
                                per_sample_nll_contact,
                                weights=domain_loss_weights,
                                valid_mask=domain_valid_mask
                            )

                            log_probs_disc_residual = model.discriminator(
                                z_all_residual_for_adversarial.detach()
                            )
                            per_sample_nll_residual = F.nll_loss(
                                log_probs_disc_residual,
                                domain_labels,
                                reduction='none'
                            )
                            loss_disc_residual = _weighted_mean(
                                per_sample_nll_residual,
                                weights=domain_loss_weights,
                                valid_mask=domain_valid_mask
                            )
                            loss_discriminator = (
                                float(im_contact_ratio) * loss_disc_contact +
                                (1.0 - float(im_contact_ratio)) * loss_disc_residual
                            )
                        else:
                            # 判别器目标：最小化域分类误差（输入detach，避免更新encoder）
                            log_probs_disc = model.discriminator(z_all_for_adversarial.detach())
                            per_sample_nll = F.nll_loss(
                                log_probs_disc,
                                domain_labels,
                                reduction='none'
                            )
                            loss_discriminator = _weighted_mean(
                                per_sample_nll,
                                weights=domain_loss_weights,
                                valid_mask=domain_valid_mask
                            )
                    optimizer['discriminator'].zero_grad()
                    loss_scaler._scaler.scale(loss_discriminator).backward()
                    loss_scaler._scaler.unscale_(optimizer['discriminator'])
                    loss_scaler._scaler.step(optimizer['discriminator'])
                    loss_scaler._scaler.update()
        
        # 继续在autocast内计算损失
        with torch.cuda.amp.autocast():
            
            # ================================================================
            # 计算损失
            # ================================================================

            # Group-DRO: 按组计算损失（使用完整batch作为负样本池）
            if use_group_dro and 'group_label' in batch:
                group_label = batch['group_label'].to(device).long()  # [B]

                # ✅ 先计算每个样本的对比损失（负样本池=完整batch），再按组聚合
                per_sample_loss = multi_positive_infonce_loss(
                    z_contact_dict,
                    temperature=temperature,
                    use_semi_hard_negatives=use_semi_hard,
                    similarity_window=similarity_window,
                    object_ids=object_ids,
                    frame_ids=frame_ids,
                    exclude_same_object_negatives=exclude_same_object_negatives,
                    prioritize_same_object_diff_frame_negatives=prioritize_same_object_diff_frame_negatives,
                    same_object_negative_keep_ratio=same_object_negative_keep_ratio,
                    reduction="none",
                )  # [B]

                loss_fine_batch = None
                loss_indoor_batch = None
                loss_outdoor_batch = None

                for group_id in range(3):
                    group_mask = (group_label == group_id)
                    if group_mask.any():
                        loss_group = per_sample_loss[group_mask].mean()
                        if group_id == 0:  # Fine
                            loss_fine_batch = loss_group
                        elif group_id == 1:  # Indoor
                            loss_indoor_batch = loss_group
                        else:  # Outdoor
                            loss_outdoor_batch = loss_group

                # 累积组损失（只统计出现的组，避免污染phase2_combined）
                if loss_fine_batch is not None:
                    total_loss_fine += loss_fine_batch.item()
                if loss_indoor_batch is not None:
                    total_loss_indoor += loss_indoor_batch.item()
                if loss_outdoor_batch is not None:
                    total_loss_outdoor += loss_outdoor_batch.item()
                group_count += 1

                # 只对出现的组计算加权loss
                group_losses_list = []
                group_weights_list = []
                if loss_fine_batch is not None:
                    group_losses_list.append(loss_fine_batch)
                    group_weights_list.append(group_weights[0])
                if loss_indoor_batch is not None:
                    group_losses_list.append(loss_indoor_batch)
                    group_weights_list.append(group_weights[1])
                if loss_outdoor_batch is not None:
                    group_losses_list.append(loss_outdoor_batch)
                    group_weights_list.append(group_weights[2])

                if len(group_losses_list) > 0:
                    group_losses_tensor = torch.stack(group_losses_list)
                    group_weights_tensor = torch.stack(group_weights_list)
                    group_weights_tensor = group_weights_tensor / group_weights_tensor.sum()
                    loss_mp = (group_weights_tensor * group_losses_tensor).sum()
                else:
                    loss_mp = multi_positive_infonce_loss(
                        z_contact_dict,
                        temperature=temperature,
                        use_semi_hard_negatives=use_semi_hard,
                        similarity_window=similarity_window,
                        object_ids=object_ids,
                        frame_ids=frame_ids,
                        exclude_same_object_negatives=exclude_same_object_negatives,
                        prioritize_same_object_diff_frame_negatives=prioritize_same_object_diff_frame_negatives,
                        same_object_negative_keep_ratio=same_object_negative_keep_ratio,
                    )

                # 更新Group-DRO权重（只对出现的组）
                with torch.no_grad():
                    valid_groups = []
                    valid_losses = []

                    if loss_fine_batch is not None:
                        valid_groups.append(0)
                        valid_losses.append(loss_fine_batch.item())
                    if loss_indoor_batch is not None:
                        valid_groups.append(1)
                        valid_losses.append(loss_indoor_batch.item())
                    if loss_outdoor_batch is not None:
                        valid_groups.append(2)
                        valid_losses.append(loss_outdoor_batch.item())

                    if len(valid_groups) > 0:
                        valid_losses_tensor = torch.tensor(valid_losses, device=device)
                        valid_weights = group_weights[valid_groups]

                        center_losses = getattr(args, 'group_dro_center_losses', False)
                        normalize_losses = getattr(args, 'group_dro_normalize_losses', False)
                        if normalize_losses and args.stage == '2b':
                            losses_mean = valid_losses_tensor.mean()
                            losses_std = valid_losses_tensor.std() + 1e-8
                            losses_normalized = (valid_losses_tensor - losses_mean) / losses_std
                            valid_weights = valid_weights * torch.exp(group_dro_eta * losses_normalized)
                        elif center_losses:
                            losses_centered = valid_losses_tensor - valid_losses_tensor.mean()
                            valid_weights = valid_weights * torch.exp(group_dro_eta * losses_centered)
                        else:
                            valid_weights = valid_weights * torch.exp(group_dro_eta * valid_losses_tensor)

                        valid_weights = torch.clamp(valid_weights, min=1e-8)
                        valid_weights = valid_weights / valid_weights.sum()
                        if use_group_dro:
                            min_w = group_dro_min_weight
                            max_w = group_dro_max_weight
                            if min_w > 0.0 or max_w < 1.0:
                                valid_weights = torch.clamp(valid_weights, min=min_w, max=max_w)
                                valid_weights = torch.clamp(valid_weights, min=1e-8)
                                valid_weights = valid_weights / valid_weights.sum()
                        group_weights[valid_groups] = valid_weights
                        # 记录Group-DRO权重动态，便于监控和发出警告
                        current_max = group_weights.max().item()
                        current_min = group_weights.min().item()
                        group_weight_max_observed = max(group_weight_max_observed, current_max)
                        group_weight_min_observed = min(group_weight_min_observed, current_min)
                        saturated_upper = (current_max >= max(group_dro_max_weight - group_weight_saturation_margin, 0.0))
                        saturated_lower = (current_min <= group_dro_min_weight + group_weight_saturation_margin)
                        if saturated_upper or saturated_lower:
                            group_weight_saturation_steps += 1
                        else:
                            group_weight_saturation_steps = max(group_weight_saturation_steps - 1, 0)
            else:
                # 标准InfoNCE (无Group-DRO)
                loss_mp = multi_positive_infonce_loss(
                    z_contact_dict,
                    temperature=temperature,
                    use_semi_hard_negatives=use_semi_hard,
                    similarity_window=similarity_window,
                    object_ids=object_ids,
                    frame_ids=frame_ids,
                    exclude_same_object_negatives=exclude_same_object_negatives,
                    prioritize_same_object_diff_frame_negatives=prioritize_same_object_diff_frame_negatives,
                    same_object_negative_keep_ratio=same_object_negative_keep_ratio,
                )

                # 记录统一的损失（无分组）
                loss_fine_batch = loss_mp.item()
                loss_indoor_batch = loss_mp.item()
                loss_outdoor_batch = loss_mp.item()
            
            # (2) 🆕 域不变性损失（支持对抗式训练）
            loss_im_contact = torch.tensor(0.0, device=device)
            loss_im_residual = torch.tensor(0.0, device=device)
            if lambda_im > 0 and model.discriminator is not None:
                if use_adversarial and args.stage == '2b' and isinstance(optimizer, dict):
                    # Step 2: Encoder对抗损失（在已更新的判别器上计算）
                    # 冻结判别器参数，避免encoder反向时污染判别器梯度
                    discriminator_params = list(model.discriminator.parameters())
                    for p in discriminator_params:
                        p.requires_grad_(False)
                    if (
                        im_dual_target and
                        z_all_contact_for_adversarial is not None and
                        z_all_residual_for_adversarial is not None
                    ):
                        log_probs_encoder_contact = model.discriminator(
                            z_all_contact_for_adversarial
                        )
                        per_sample_kl_contact = _infomax_kl_per_sample_from_log_probs(
                            log_probs_encoder_contact,
                            num_sensors=num_sensors
                        )
                        loss_im_contact = _weighted_mean(
                            per_sample_kl_contact,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )

                        log_probs_encoder_residual = model.discriminator(
                            z_all_residual_for_adversarial
                        )
                        per_sample_kl_residual = _infomax_kl_per_sample_from_log_probs(
                            log_probs_encoder_residual,
                            num_sensors=num_sensors
                        )
                        loss_im_residual = _weighted_mean(
                            per_sample_kl_residual,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )
                        loss_im = (
                            float(im_contact_ratio) * loss_im_contact +
                            (1.0 - float(im_contact_ratio)) * loss_im_residual
                        )
                    else:
                        log_probs_encoder = model.discriminator(z_all_for_adversarial)  # 不detach，梯度回到encoder
                        per_sample_kl = _infomax_kl_per_sample_from_log_probs(
                            log_probs_encoder,
                            num_sensors=num_sensors
                        )
                        loss_im = _weighted_mean(
                            per_sample_kl,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )
                        if adversarial_feature_target == 'contact':
                            loss_im_contact = loss_im
                        elif adversarial_feature_target == 'contact_residual':
                            loss_im_residual = loss_im
                    for p in discriminator_params:
                        p.requires_grad_(True)
                else:
                    # 原版 InfoMax（Phase-1 或兼容模式）
                    if im_dual_target and args.stage == '2b':
                        z_all_contact = torch.cat(
                            [z_contact_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                        )
                        z_residual_eval_dict = _compute_contact_residual_dict(z_contact_dict, normalize=False)
                        z_all_residual = torch.cat(
                            [z_residual_eval_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                        )
                        log_probs_contact = model.discriminator(z_all_contact)
                        per_sample_kl_contact = _infomax_kl_per_sample_from_log_probs(
                            log_probs_contact,
                            num_sensors=num_sensors
                        )
                        loss_im_contact = _weighted_mean(
                            per_sample_kl_contact,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )
                        log_probs_residual = model.discriminator(z_all_residual)
                        per_sample_kl_residual = _infomax_kl_per_sample_from_log_probs(
                            log_probs_residual,
                            num_sensors=num_sensors
                        )
                        loss_im_residual = _weighted_mean(
                            per_sample_kl_residual,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )
                        loss_im = (
                            float(im_contact_ratio) * loss_im_contact +
                            (1.0 - float(im_contact_ratio)) * loss_im_residual
                        )
                    else:
                        z_all = torch.cat([z_adv_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0)
                        log_probs = model.discriminator(z_all)
                        per_sample_kl = _infomax_kl_per_sample_from_log_probs(
                            log_probs,
                            num_sensors=num_sensors
                        )
                        loss_im = _weighted_mean(
                            per_sample_kl,
                            weights=domain_loss_weights,
                            valid_mask=domain_valid_mask
                        )
                        if adversarial_feature_target == 'contact':
                            loss_im_contact = loss_im
                        elif adversarial_feature_target == 'contact_residual':
                            loss_im_residual = loss_im
            else:
                loss_im = torch.tensor(0.0, device=device)

            # (2.5) z_domain 域监督损失（关键修复2）
            if (
                lambda_domain_cls > 0 and
                hasattr(model, 'domain_classifier') and
                model.domain_classifier is not None
            ):
                if domain_cls_detach_encoder:
                    # 域监督只训练域分支，不把“域信息”写回共享encoder
                    h_domain_all = torch.cat(
                        [h_dict[s].detach() for s in ['gelsight', 'digit', 'duragel']], dim=0
                    )
                    z_domain_for_cls = model.domain_projection_head(h_domain_all)
                else:
                    z_domain_for_cls = torch.cat(
                        [z_domain_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                    )
                log_probs_domain_cls = model.domain_classifier(z_domain_for_cls)
                per_sample_domain_cls = F.nll_loss(
                    log_probs_domain_cls,
                    domain_labels,
                    reduction='none'
                )
                loss_domain_cls = _weighted_mean(
                    per_sample_domain_cls,
                    weights=domain_loss_weights,
                    valid_mask=domain_valid_mask
                )
            else:
                loss_domain_cls = torch.tensor(0.0, device=device)

            # (2.6) Cross-sensor matching 辅助损失
            is_phase2_style = (
                getattr(args, 'stage', None) == '2b' and
                getattr(args, 'use_adversarial_training', False)
            )
            matching_allowed = not (is_phase2_style and not enable_matching_in_phase2)
            if lambda_match > 0 and getattr(model, 'match_head', None) is not None and matching_allowed:
                object_ids = batch.get('object', None)
                loss_match = _compute_matching_loss(
                    z_contact_dict,
                    model.match_head,
                    num_negatives=match_negatives_per_pair,
                    object_ids=object_ids,
                    hard_negative_topk=match_hard_negative_topk,
                    require_diff_object=True,
                )
            else:
                loss_match = torch.tensor(0.0, device=device)

            # (2.7) Row-wise 跨传感器配对拉近损失（同index同接触）
            if lambda_pair > 0:
                loss_pair = _compute_rowwise_pair_loss(
                    z_contact_dict,
                    pair_loss_type=pair_loss_type,
                    pair_confidence=pair_confidence,
                    use_pair_confidence_weighting=use_pair_confidence_weighting,
                    confidence_floor=pair_confidence_floor,
                    confidence_power=pair_confidence_power,
                    confidence_min=pair_confidence_min,
                )
            else:
                loss_pair = torch.tensor(0.0, device=device)

            # (2.8) Row-wise hard-tail配对损失（仅优化最难样本）
            if lambda_pair_tail > 0:
                loss_pair_tail = _compute_rowwise_pair_hard_tail_loss(
                    z_contact_dict,
                    pair_loss_type=pair_loss_type,
                    pair_confidence=pair_confidence,
                    use_pair_confidence_weighting=use_pair_confidence_weighting,
                    confidence_floor=pair_confidence_floor,
                    confidence_power=pair_confidence_power,
                    confidence_min=pair_confidence_min,
                    tail_ratio=pair_tail_ratio,
                    tail_min_samples=pair_tail_min_samples,
                )
            else:
                loss_pair_tail = torch.tensor(0.0, device=device)

            # (2.9) 团间分离损失（不同contact要拉开）
            if lambda_inter > 0:
                loss_inter = _compute_inter_contact_separation_loss(
                    z_contact_dict,
                    object_ids=object_ids,
                    frame_ids=frame_ids,
                    pair_confidence=pair_confidence,
                    use_confidence_weighting=use_pair_confidence_for_domain_and_inter,
                    confidence_floor=pair_confidence_floor,
                    confidence_power=pair_confidence_power,
                    confidence_min=pair_confidence_min if use_pair_confidence_for_domain_and_inter else 0.0,
                    sim_margin=inter_sim_margin,
                    hard_topk=inter_hard_topk,
                )
            else:
                loss_inter = torch.tensor(0.0, device=device)

            # (2.9b) 每个传感器分支内做团间分离，抑制“按传感器成岛”
            if lambda_inter_sensor > 0:
                loss_inter_sensor = _compute_inter_contact_separation_loss_per_sensor(
                    z_contact_dict,
                    object_ids=object_ids,
                    frame_ids=frame_ids,
                    pair_confidence=pair_confidence,
                    use_confidence_weighting=use_pair_confidence_for_domain_and_inter,
                    confidence_floor=pair_confidence_floor,
                    confidence_power=pair_confidence_power,
                    confidence_min=pair_confidence_min if use_pair_confidence_for_domain_and_inter else 0.0,
                    sim_margin=inter_sim_margin,
                    hard_topk=inter_hard_topk,
                )
            else:
                loss_inter_sensor = torch.tensor(0.0, device=device)

            # (2.10) 条件区域紧致（压缩同contact内跨传感器残差）
            if lambda_residual_compact > 0:
                loss_residual_compact = _compute_residual_compact_loss(
                    z_contact_dict,
                    pair_confidence=pair_confidence,
                    use_pair_confidence_weighting=use_pair_confidence_weighting,
                    confidence_floor=pair_confidence_floor,
                    confidence_power=pair_confidence_power,
                    confidence_min=pair_confidence_min,
                )
            else:
                loss_residual_compact = torch.tensor(0.0, device=device)
            
            # (3) 方差正则化损失 (防止特征坍缩)
            use_variance_reg = getattr(args, 'use_variance_reg', False)
            lambda_var = getattr(args, 'lambda_var', 0.1)
            target_std = getattr(args, 'target_std', 0.15)
            
            if use_variance_reg:
                # 计算当前标准差用于自适应权重
                current_std = compute_representation_std(z_contact_dict)
                loss_var = adaptive_variance_loss(
                    z_contact_dict,
                    target_std=target_std,
                    current_std=current_std
                )
            else:
                loss_var = torch.tensor(0.0, device=device)

            # (3.5) Contact/Domain 解耦损失
            if lambda_decouple > 0:
                loss_decouple = _compute_decoupling_loss(
                    z_contact_dict,
                    z_domain_dict,
                    detach_domain=decouple_detach_domain
                )
            else:
                loss_decouple = torch.tensor(0.0, device=device)

            # (3.7) 跨传感器CORAL对齐（直接约束z_contact去传感器化）
            if lambda_coral > 0:
                loss_coral = _compute_sensor_coral_loss(
                    z_contact_dict,
                    include_mean=coral_include_mean
                )
            else:
                loss_coral = torch.tensor(0.0, device=device)

            # (3.8) 条件CORAL（同pair残差对齐，避免压缩接触主语义）
            if lambda_coral_residual > 0:
                loss_coral_residual = _compute_contact_residual_coral_loss(
                    z_contact_dict,
                    include_mean=coral_residual_include_mean
                )
            else:
                loss_coral_residual = torch.tensor(0.0, device=device)

            # (3.9) EMA量纲平衡：动态调整辅助项相对强度
            balance_factors = _compute_dynamic_balance_factors(
                dynamic_balance_state,
                {
                    'loss_mp': loss_mp.detach().item(),
                    'loss_im': loss_im.detach().item() if isinstance(loss_im, torch.Tensor) else 0.0,
                    'loss_match': loss_match.detach().item() if isinstance(loss_match, torch.Tensor) else 0.0,
                    'loss_pair': loss_pair.detach().item() if isinstance(loss_pair, torch.Tensor) else 0.0,
                    'loss_pair_tail': (
                        loss_pair_tail.detach().item() if isinstance(loss_pair_tail, torch.Tensor) else 0.0
                    ),
                    'loss_inter': loss_inter.detach().item() if isinstance(loss_inter, torch.Tensor) else 0.0,
                    'loss_inter_sensor': (
                        loss_inter_sensor.detach().item()
                        if isinstance(loss_inter_sensor, torch.Tensor)
                        else 0.0
                    ),
                    'loss_residual_compact': (
                        loss_residual_compact.detach().item()
                        if isinstance(loss_residual_compact, torch.Tensor)
                        else 0.0
                    ),
                    'loss_decouple': loss_decouple.detach().item() if isinstance(loss_decouple, torch.Tensor) else 0.0,
                    'loss_domain_cls': loss_domain_cls.detach().item() if isinstance(loss_domain_cls, torch.Tensor) else 0.0,
                    'loss_coral': loss_coral.detach().item() if isinstance(loss_coral, torch.Tensor) else 0.0,
                    'loss_coral_residual': (
                        loss_coral_residual.detach().item()
                        if isinstance(loss_coral_residual, torch.Tensor)
                        else 0.0
                    ),
                }
            )
            balance_im = float(balance_factors.get('loss_im', 1.0))
            balance_match = float(balance_factors.get('loss_match', 1.0))
            balance_pair = float(balance_factors.get('loss_pair', 1.0))
            balance_pair_tail = float(balance_factors.get('loss_pair_tail', 1.0))
            balance_inter = float(balance_factors.get('loss_inter', 1.0))
            balance_inter_sensor = float(balance_factors.get('loss_inter_sensor', 1.0))
            balance_residual_compact = float(balance_factors.get('loss_residual_compact', 1.0))
            balance_decouple = float(balance_factors.get('loss_decouple', 1.0))
            balance_domain_cls = float(balance_factors.get('loss_domain_cls', 1.0))
            balance_coral = float(balance_factors.get('loss_coral', 1.0))
            balance_coral_residual = float(balance_factors.get('loss_coral_residual', 1.0))
            
            # (4) 总损失
            loss = (
                loss_mp +
                lambda_match * balance_match * loss_match +
                lambda_pair * balance_pair * loss_pair +
                lambda_pair_tail * balance_pair_tail * loss_pair_tail +
                lambda_inter * balance_inter * loss_inter +
                lambda_inter_sensor * balance_inter_sensor * loss_inter_sensor +
                lambda_residual_compact * balance_residual_compact * loss_residual_compact +
                lambda_im * im_loss_scale * balance_im * loss_im +
                lambda_domain_cls * balance_domain_cls * loss_domain_cls +
                lambda_var * loss_var +
                lambda_decouple * balance_decouple * loss_decouple +
                lambda_coral * balance_coral * loss_coral +
                lambda_coral_residual * balance_coral_residual * loss_coral_residual
            )
        
        # 检查损失有效性
        if not math.isfinite(loss.item()):
            print(f"⚠️  Warning: loss is {loss.item()}, skipping this batch")
            continue
        
        loss_value = loss.item()
        loss_mp_value = loss_mp.item()
        loss_im_value = loss_im.item() if isinstance(loss_im, torch.Tensor) else 0.0
        loss_im_contact_value = (
            loss_im_contact.item() if isinstance(loss_im_contact, torch.Tensor) else 0.0
        )
        loss_im_residual_value = (
            loss_im_residual.item() if isinstance(loss_im_residual, torch.Tensor) else 0.0
        )
        loss_im_scaled_value = lambda_im * im_loss_scale * balance_im * loss_im_value
        loss_var_value = loss_var.item() if isinstance(loss_var, torch.Tensor) else 0.0
        loss_match_value = loss_match.item() if isinstance(loss_match, torch.Tensor) else 0.0
        loss_pair_value = loss_pair.item() if isinstance(loss_pair, torch.Tensor) else 0.0
        loss_pair_tail_value = loss_pair_tail.item() if isinstance(loss_pair_tail, torch.Tensor) else 0.0
        loss_inter_value = loss_inter.item() if isinstance(loss_inter, torch.Tensor) else 0.0
        loss_inter_sensor_value = (
            loss_inter_sensor.item() if isinstance(loss_inter_sensor, torch.Tensor) else 0.0
        )
        loss_residual_compact_value = (
            loss_residual_compact.item() if isinstance(loss_residual_compact, torch.Tensor) else 0.0
        )
        loss_decouple_value = loss_decouple.item() if isinstance(loss_decouple, torch.Tensor) else 0.0
        loss_coral_value = loss_coral.item() if isinstance(loss_coral, torch.Tensor) else 0.0
        loss_coral_residual_value = (
            loss_coral_residual.item() if isinstance(loss_coral_residual, torch.Tensor) else 0.0
        )
        loss_domain_cls_value = (
            loss_domain_cls.item() if isinstance(loss_domain_cls, torch.Tensor) else 0.0
        )
        
        # ================================================================
        # 反向传播和优化
        # ================================================================
        loss = loss / accum_iter
        
        # 🆕 支持字典形式的optimizer
        if isinstance(optimizer, dict):
            # 对抗式训练：只更新encoder
            encoder_params = []
            for param_group in optimizer['encoder'].param_groups:
                encoder_params.extend(param_group['params'])
            loss_scaler(
                loss,
                optimizer['encoder'],
                parameters=encoder_params,
                update_grad=(data_iter_step + 1) % accum_iter == 0
            )
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer['encoder'].zero_grad()
        else:
            # 标准训练
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(data_iter_step + 1) % accum_iter == 0
            )
            
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        
        # ================================================================
        # 计算监控指标（不需要梯度）
        # ================================================================
        with torch.no_grad():
            # 相似度统计
            stats = compute_similarity_statistics(z_contact_dict)
            
            # 域探针准确率（详细版）
            if model.discriminator is not None:
                domain_metrics = compute_domain_metrics_detailed(z_adv_dict, model.discriminator)
                domain_acc = domain_metrics['domain_acc']
                
                # 累积详细域判别器指标
                total_domain_acc_gelsight += domain_metrics['domain_acc_gelsight']
                total_domain_acc_digit += domain_metrics['domain_acc_digit']
                total_domain_acc_duragel += domain_metrics['domain_acc_duragel']
                total_domain_acc_std += domain_metrics['domain_acc_std']
                total_domain_pred_ratio_0 += domain_metrics['domain_pred_ratio_0']
                total_domain_pred_ratio_1 += domain_metrics['domain_pred_ratio_1']
                total_domain_pred_ratio_2 += domain_metrics['domain_pred_ratio_2']
                total_domain_entropy += domain_metrics['domain_entropy']
                if domain_metrics['domain_is_collapsed']:
                    domain_collapse_count += 1
            else:
                domain_acc = 0.0
            
            # 表征标准差（防止塌缩）
            repr_std = compute_representation_std(z_contact_dict)
            
            # 累积指标
            total_domain_acc += domain_acc
            total_repr_std += repr_std
            total_same_sim += stats['same_contact_sim_mean']
            total_diff_sim += stats['diff_contact_sim_mean']
            metric_count += 1
        
        # 同步所有进程（分布式训练）
        torch.cuda.synchronize()
        
        # 更新 metric logger
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mp=loss_mp_value)
        metric_logger.update(loss_im=loss_im_value)
        metric_logger.update(loss_im_contact=loss_im_contact_value)
        metric_logger.update(loss_im_residual=loss_im_residual_value)
        metric_logger.update(loss_var=loss_var_value)
        metric_logger.update(loss_match=loss_match_value)
        metric_logger.update(loss_pair=loss_pair_value)
        metric_logger.update(loss_pair_tail=loss_pair_tail_value)
        metric_logger.update(loss_inter=loss_inter_value)
        metric_logger.update(loss_inter_sensor=loss_inter_sensor_value)
        metric_logger.update(loss_residual_compact=loss_residual_compact_value)
        metric_logger.update(loss_decouple=loss_decouple_value)
        metric_logger.update(loss_domain_cls=loss_domain_cls_value)
        metric_logger.update(loss_coral=loss_coral_value)
        metric_logger.update(loss_coral_residual=loss_coral_residual_value)
        metric_logger.update(pair_conf=pair_confidence_mean)
        metric_logger.update(bal_im=balance_im)
        metric_logger.update(bal_match=balance_match)
        metric_logger.update(bal_pair=balance_pair)
        metric_logger.update(bal_pair_tail=balance_pair_tail)
        metric_logger.update(bal_inter=balance_inter)
        metric_logger.update(bal_inter_sensor=balance_inter_sensor)
        metric_logger.update(bal_residual_compact=balance_residual_compact)
        metric_logger.update(bal_decouple=balance_decouple)
        metric_logger.update(bal_domain_cls=balance_domain_cls)
        metric_logger.update(bal_coral=balance_coral)
        metric_logger.update(bal_coral_residual=balance_coral_residual)
        
        # 🆕 支持字典形式的optimizer
        if isinstance(optimizer, dict):
            lr = optimizer['encoder'].param_groups[0]["lr"]
        else:
            lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # ================================================================
        # Wandb 日志（可选）
        # ================================================================
        if wandb_log and data_iter_step % print_freq == 0:
            try:
                import wandb
                
                # 基础指标
                sim_gap = stats['same_contact_sim_mean'] - stats['diff_contact_sim_mean']
                log_dict = {
                    'train/loss': loss_value,
                    'train/loss_mp': loss_mp_value,
                    'train/loss_im': loss_im_value,
                    'train/loss_im_contact': loss_im_contact_value,
                    'train/loss_im_residual': loss_im_residual_value,
                    'train/loss_im_scaled': loss_im_scaled_value,
                    'train/loss_var': loss_var_value,
                    'train/loss_match': loss_match_value,
                    'train/loss_pair': loss_pair_value,
                    'train/loss_pair_tail': loss_pair_tail_value,
                    'train/loss_inter': loss_inter_value,
                    'train/loss_inter_sensor': loss_inter_sensor_value,
                    'train/loss_residual_compact': loss_residual_compact_value,
                    'train/loss_decouple': loss_decouple_value,
                    'train/loss_domain_cls': loss_domain_cls_value,
                    'train/loss_coral': loss_coral_value,
                    'train/loss_coral_residual': loss_coral_residual_value,
                    'train/pair_confidence_mean': pair_confidence_mean,
                    'train/im_loss_scale': im_loss_scale,
                    'train/im_contact_ratio': float(im_contact_ratio),
                    'train/digit_focus_weight': float(digit_focus_weight),
                    'train/disc_steps_per_iter': disc_steps_per_iter,
                    'train/adversarial_target_code': {
                        'domain': 0.0,
                        'contact': 1.0,
                        'contact_residual': 2.0
                    }.get(adversarial_feature_target, -1.0),
                    'train/balance_factor_im': balance_im,
                    'train/balance_factor_match': balance_match,
                    'train/balance_factor_pair': balance_pair,
                    'train/balance_factor_pair_tail': balance_pair_tail,
                    'train/balance_factor_inter': balance_inter,
                    'train/balance_factor_inter_sensor': balance_inter_sensor,
                    'train/balance_factor_residual_compact': balance_residual_compact,
                    'train/balance_factor_decouple': balance_decouple,
                    'train/balance_factor_domain_cls': balance_domain_cls,
                    'train/balance_factor_coral': balance_coral,
                    'train/balance_factor_coral_residual': balance_coral_residual,
                    'train/lr': lr,
                    'train/domain_acc': domain_acc,
                    'train/repr_std': repr_std,
                    'train/same_contact_sim': stats['same_contact_sim_mean'],
                    'train/diff_contact_sim': stats['diff_contact_sim_mean'],
                    'train/sim_gap': sim_gap,  # 关键指标：判别性
                    'epoch': epoch,
                    'iter': data_iter_step
                }
                
                # 添加encoder学习率
                if isinstance(optimizer, dict):
                    log_dict['train/lr_encoder'] = optimizer['encoder'].param_groups[0]['lr']
                    log_dict['train/lr_discriminator'] = optimizer['discriminator'].param_groups[0]['lr']
                elif len(optimizer.param_groups) > 1:
                    log_dict['train/lr_encoder'] = optimizer.param_groups[1]['lr']
                
                # 添加梯度范数
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                log_dict['train/grad_norm'] = total_norm
                
                # 🔧 FIX 5: Group-DRO相关指标（只对出现的组计算gap）
                if use_group_dro:
                    # 收集出现组的loss
                    batch_group_losses = []
                    if isinstance(loss_fine_batch, torch.Tensor):
                        batch_group_losses.append(loss_fine_batch if isinstance(loss_fine_batch, torch.Tensor) else loss_fine_batch)
                    if isinstance(loss_indoor_batch, torch.Tensor):
                        batch_group_losses.append(loss_indoor_batch if isinstance(loss_indoor_batch, torch.Tensor) else loss_indoor_batch)
                    if isinstance(loss_outdoor_batch, torch.Tensor):
                        batch_group_losses.append(loss_outdoor_batch if isinstance(loss_outdoor_batch, torch.Tensor) else loss_outdoor_batch)
                    
                    # 只对出现的组计算gap
                    if len(batch_group_losses) > 1:
                        batch_gap = max(batch_group_losses) - min(batch_group_losses)
                    else:
                        batch_gap = 0.0
                    
                    log_dict.update({
                        'train/loss_fine': loss_fine_batch if isinstance(loss_fine_batch, torch.Tensor) else 0.0,
                        'train/loss_indoor': loss_indoor_batch if isinstance(loss_indoor_batch, torch.Tensor) else 0.0,
                        'train/loss_outdoor': loss_outdoor_batch if isinstance(loss_outdoor_batch, torch.Tensor) else 0.0,
                        'train/group_weight_fine': group_weights[0].item(),
                        'train/group_weight_indoor': group_weights[1].item(),
                        'train/group_weight_outdoor': group_weights[2].item(),
                        'train/group_weight_max_observed': group_weight_max_observed,
                        'train/group_weight_min_observed': group_weight_min_observed,
                        'train/group_loss_gap': batch_gap
                    })
                
                wandb.log(log_dict)
            except ImportError:
                pass  # Wandb 未安装
    
    # ====================================================================
    # Epoch 结束统计
    # ====================================================================
    metric_logger.synchronize_between_processes()
    
    # 计算平均监控指标
    avg_domain_acc = total_domain_acc / max(metric_count, 1)
    avg_repr_std = total_repr_std / max(metric_count, 1)
    avg_same_sim = total_same_sim / max(metric_count, 1)
    avg_diff_sim = total_diff_sim / max(metric_count, 1)
    
    # 计算详细域判别器指标的平均值
    avg_domain_acc_gelsight = total_domain_acc_gelsight / max(metric_count, 1)
    avg_domain_acc_digit = total_domain_acc_digit / max(metric_count, 1)
    avg_domain_acc_duragel = total_domain_acc_duragel / max(metric_count, 1)
    avg_domain_acc_std = total_domain_acc_std / max(metric_count, 1)
    avg_domain_pred_ratio_0 = total_domain_pred_ratio_0 / max(metric_count, 1)
    avg_domain_pred_ratio_1 = total_domain_pred_ratio_1 / max(metric_count, 1)
    avg_domain_pred_ratio_2 = total_domain_pred_ratio_2 / max(metric_count, 1)
    avg_domain_entropy = total_domain_entropy / max(metric_count, 1)
    domain_collapse_ratio = domain_collapse_count / max(metric_count, 1)
    
    avg_sim_gap = avg_same_sim - avg_diff_sim
    
    print(f"\n📊 Epoch {epoch} Training Summary:")
    print(f"   Averaged stats: {metric_logger}")
    print(f"   Domain probe accuracy: {avg_domain_acc*100:.2f}%")
    print(f"   Representation std: {avg_repr_std:.4f}")
    print(f"   Same contact similarity: {avg_same_sim:.4f}")
    print(f"   Diff contact similarity: {avg_diff_sim:.4f}")
    print(f"   Similarity gap (discriminability): {avg_sim_gap:.4f}")
    if dynamic_balance_state['enabled']:
        print(
            f"   [Dynamic Balance] im={metric_logger.meters['bal_im'].global_avg:.3f}, "
            f"pair={metric_logger.meters['bal_pair'].global_avg:.3f}, "
            f"pair_tail={metric_logger.meters['bal_pair_tail'].global_avg:.3f}, "
            f"inter={metric_logger.meters['bal_inter'].global_avg:.3f}, "
            f"inter_sensor={metric_logger.meters['bal_inter_sensor'].global_avg:.3f}, "
            f"residual_compact={metric_logger.meters['bal_residual_compact'].global_avg:.3f}, "
            f"domain_cls={metric_logger.meters['bal_domain_cls'].global_avg:.3f}, "
            f"coral_res={metric_logger.meters['bal_coral_residual'].global_avg:.3f}"
        )
    
    # 打印详细域判别器指标
    if model.discriminator is not None:
        print(f"   [Domain Detailed] GelSight: {avg_domain_acc_gelsight*100:.2f}% | DIGIT: {avg_domain_acc_digit*100:.2f}% | DuraGel: {avg_domain_acc_duragel*100:.2f}%")
        print(f"   [Domain Detailed] Std: {avg_domain_acc_std:.4f} | Entropy: {avg_domain_entropy:.4f} (max=1.099)")
        print(f"   [Domain Detailed] Pred Ratios: [{avg_domain_pred_ratio_0:.3f}, {avg_domain_pred_ratio_1:.3f}, {avg_domain_pred_ratio_2:.3f}]")
        if domain_collapse_ratio > 0.1:
            print(f"   🚨 [WARNING] Domain collapse detected in {domain_collapse_ratio*100:.1f}% of batches!")
    
    # 🔧 FIX 4: Group-DRO统计（只对出现的组计算gap）
    if use_group_dro and group_count > 0:
        # 收集出现组的平均loss
        group_losses_for_gap = []
        avg_loss_fine = total_loss_fine / group_count if total_loss_fine > 0 else 0.0
        avg_loss_indoor = total_loss_indoor / group_count if total_loss_indoor > 0 else 0.0
        avg_loss_outdoor = total_loss_outdoor / group_count if total_loss_outdoor > 0 else 0.0
        
        if total_loss_fine > 0:
            group_losses_for_gap.append(avg_loss_fine)
        if total_loss_indoor > 0:
            group_losses_for_gap.append(avg_loss_indoor)
        if total_loss_outdoor > 0:
            group_losses_for_gap.append(avg_loss_outdoor)
        
        # 只对出现的组计算gap
        if len(group_losses_for_gap) > 1:
            group_loss_gap = max(group_losses_for_gap) - min(group_losses_for_gap)
        else:
            group_loss_gap = 0.0
        
        print(f"   [Group-DRO] Fine: {avg_loss_fine:.4f} | Indoor: {avg_loss_indoor:.4f} | Outdoor: {avg_loss_outdoor:.4f}")
        print(f"   [Group-DRO] Loss Gap: {group_loss_gap:.4f} (from {len(group_losses_for_gap)} groups)")
        print(f"   [Group-DRO] Weights - Fine: {group_weights[0]:.3f} | Indoor: {group_weights[1]:.3f} | Outdoor: {group_weights[2]:.3f}")
        if group_weight_saturation_steps >= group_weight_saturation_warn:
            print(f"   ⚠️  Group-DRO权重已在边界附近停留 {group_weight_saturation_steps} steps，请考虑提高 eta 或放宽 min/max")
    
    # 返回统计信息
    stats = {
        'loss': metric_logger.meters['loss'].global_avg,
        'loss_mp': metric_logger.meters['loss_mp'].global_avg,
        'loss_im': metric_logger.meters['loss_im'].global_avg,
        'loss_im_contact': metric_logger.meters['loss_im_contact'].global_avg,
        'loss_im_residual': metric_logger.meters['loss_im_residual'].global_avg,
        'loss_pair': metric_logger.meters['loss_pair'].global_avg,
        'loss_pair_tail': metric_logger.meters['loss_pair_tail'].global_avg,
        'loss_inter': metric_logger.meters['loss_inter'].global_avg,
        'loss_inter_sensor': metric_logger.meters['loss_inter_sensor'].global_avg,
        'loss_residual_compact': metric_logger.meters['loss_residual_compact'].global_avg,
        'loss_decouple': metric_logger.meters['loss_decouple'].global_avg,
        'loss_domain_cls': metric_logger.meters['loss_domain_cls'].global_avg,
        'loss_coral': metric_logger.meters['loss_coral'].global_avg,
        'loss_coral_residual': metric_logger.meters['loss_coral_residual'].global_avg,
        'pair_confidence': metric_logger.meters['pair_conf'].global_avg,
        'balance_im': metric_logger.meters['bal_im'].global_avg,
        'balance_match': metric_logger.meters['bal_match'].global_avg,
        'balance_pair': metric_logger.meters['bal_pair'].global_avg,
        'balance_pair_tail': metric_logger.meters['bal_pair_tail'].global_avg,
        'balance_inter': metric_logger.meters['bal_inter'].global_avg,
        'balance_inter_sensor': metric_logger.meters['bal_inter_sensor'].global_avg,
        'balance_residual_compact': metric_logger.meters['bal_residual_compact'].global_avg,
        'balance_decouple': metric_logger.meters['bal_decouple'].global_avg,
        'balance_domain_cls': metric_logger.meters['bal_domain_cls'].global_avg,
        'balance_coral': metric_logger.meters['bal_coral'].global_avg,
        'balance_coral_residual': metric_logger.meters['bal_coral_residual'].global_avg,
        'lr': metric_logger.meters['lr'].value,
        'domain_acc': avg_domain_acc,
        'repr_std': avg_repr_std,
        'same_contact_sim': avg_same_sim,
        'diff_contact_sim': avg_diff_sim,
        # 详细域判别器指标
        'domain_acc_gelsight': avg_domain_acc_gelsight,
        'domain_acc_digit': avg_domain_acc_digit,
        'domain_acc_duragel': avg_domain_acc_duragel,
        'domain_acc_std': avg_domain_acc_std,
        'domain_pred_ratio_0': avg_domain_pred_ratio_0,
        'domain_pred_ratio_1': avg_domain_pred_ratio_1,
        'domain_pred_ratio_2': avg_domain_pred_ratio_2,
        'domain_entropy': avg_domain_entropy,
        'domain_collapse_ratio': domain_collapse_ratio
    }
    
    # 添加Group-DRO统计到返回字典
    if use_group_dro and group_count > 0:
        stats.update({
            'loss_fine': avg_loss_fine,
            'loss_indoor': avg_loss_indoor,
            'loss_outdoor': avg_loss_outdoor,
            'group_loss_gap': group_loss_gap,
            'group_weight_fine': group_weights[0].item(),
            'group_weight_indoor': group_weights[1].item(),
            'group_weight_outdoor': group_weights[2].item(),
            'group_weight_max_observed': group_weight_max_observed,
            'group_weight_min_observed': group_weight_min_observed,
            'group_weight_saturation_steps': group_weight_saturation_steps
        })
    
    return stats


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args=None,
    wandb_log=True
):
    """
    验证集评估
    
    Args:
        model: Stage2Model 实例
        data_loader: 验证集数据加载器
        device: 设备
        args: 参数配置
        wandb_log: 是否使用 Wandb 记录
        
    Returns:
        dict: 验证统计信息
    """
    model.eval()
    
    # 🔧 FIX: 设置DistributedSampler的epoch（验证集也需要）
    # 虽然验证集不需要shuffle，但仍需设置epoch以保持一致性
    if hasattr(data_loader.sampler, 'set_epoch'):
        # 验证集使用固定的epoch编号（如0）或当前训练epoch
        # 这里使用0因为验证集不需要每次都不同
        data_loader.sampler.set_epoch(0)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_mp', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im_contact', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_im_residual', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_var', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_pair', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_pair_tail', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_inter', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_inter_sensor', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_residual_compact', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_decouple', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_domain_cls', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_coral', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('loss_coral_residual', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))
    metric_logger.add_meter('pair_conf', misc.SmoothedValue(window_size=20, fmt='{global_avg:.4f}'))

    header = 'Validation:'
    print_freq = 10
    
    # 获取配置
    lambda_im = getattr(args, 'lambda_im', 0.0)
    im_loss_scale = float(getattr(args, 'im_loss_scale', 1.0))
    im_dual_target = bool(getattr(args, 'im_dual_target', False))
    temperature = getattr(args, 'temperature', 0.07)
    use_semi_hard = getattr(args, 'use_semi_hard_negatives', True)
    lambda_pair = float(getattr(args, 'lambda_pair', 0.0))
    lambda_pair_tail = float(getattr(args, 'lambda_pair_tail', 0.0))
    pair_tail_ratio = float(getattr(args, 'pair_tail_ratio', 0.2))
    pair_tail_min_samples = int(getattr(args, 'pair_tail_min_samples', 16))
    lambda_inter = float(getattr(args, 'lambda_inter', 0.0))
    lambda_inter_sensor = float(getattr(args, 'lambda_inter_sensor', 0.0))
    lambda_residual_compact = float(getattr(args, 'lambda_residual_compact', 0.0))
    inter_sim_margin = float(getattr(args, 'inter_sim_margin', 0.15))
    inter_hard_topk = int(getattr(args, 'inter_hard_topk', 512))
    use_variance_reg = bool(getattr(args, 'use_variance_reg', False))
    lambda_var = float(getattr(args, 'lambda_var', 0.0))
    target_std = float(getattr(args, 'target_std', 0.15))
    pair_loss_type = getattr(args, 'pair_loss_type', 'cosine')
    use_pair_confidence_weighting = bool(getattr(args, 'use_pair_confidence_weighting', True))
    pair_confidence_floor = float(getattr(args, 'pair_confidence_floor', 0.05))
    pair_confidence_power = float(getattr(args, 'pair_confidence_power', 1.0))
    pair_confidence_min = float(getattr(args, 'pair_confidence_min', 0.0))
    use_pair_confidence_for_domain_and_inter = bool(
        getattr(args, 'use_pair_confidence_for_domain_and_inter', False)
    )
    lambda_decouple = float(getattr(args, 'lambda_decouple', 0.0))
    lambda_domain_cls = float(getattr(args, 'lambda_domain_cls', 0.0))
    lambda_coral = float(getattr(args, 'lambda_coral', 0.0))
    lambda_coral_residual = float(getattr(args, 'lambda_coral_residual', 0.0))
    coral_include_mean = bool(getattr(args, 'coral_include_mean', True))
    coral_residual_include_mean = bool(getattr(args, 'coral_residual_include_mean', False))
    decouple_detach_domain = bool(getattr(args, 'decouple_detach_domain', True))
    domain_cls_detach_encoder = bool(getattr(args, 'domain_cls_detach_encoder', True))
    exclude_same_object_negatives = bool(getattr(args, 'exclude_same_object_negatives', False))
    prioritize_same_object_diff_frame_negatives = bool(
        getattr(args, 'prioritize_same_object_diff_frame_negatives', False)
    )
    same_object_negative_keep_ratio = float(getattr(args, 'same_object_negative_keep_ratio', 0.25))
    if prioritize_same_object_diff_frame_negatives and exclude_same_object_negatives:
        exclude_same_object_negatives = False
    val_epoch = int(getattr(args, 'current_epoch', 0))
    im_contact_ratio = _get_im_contact_ratio(args, val_epoch) if im_dual_target else 1.0
    adversarial_feature_target = _resolve_adversarial_feature_target(args, val_epoch)
    stage2_num_frames = int(getattr(args, 'stage2_num_frames', 3))
    use_digit_focus_reweight = bool(getattr(args, 'use_digit_focus_reweight', False))
    digit_focus_weight = float(getattr(args, 'digit_focus_weight', 1.0))
    global_mix_knn_k = int(getattr(args, 'global_mix_knn_k', 20))
    global_mix_knn_max_samples = int(getattr(args, 'global_mix_knn_max_samples_per_sensor', 512))
    
    # 🆕 初始化独立 probe（如果启用）
    use_independent_probe = getattr(args, 'use_independent_probe', False)
    use_linear_probe = getattr(args, 'use_linear_probe', False)  # 新增
    probe_optimizer = None
    probe_config = None
    
    # 🔧 BUG FIX 2: 收集所有特征后再训练/评估probe（避免"边背答案边考试"）
    all_z_dicts = []  # 收集所有batch的特征
    all_pair_ids: List[str] = []
    pair_link_len_list: List[torch.Tensor] = []
    
    if use_independent_probe and args.stage == '2b':
        probe_config = {
            'input_dim': args.projection_dim,
            'hidden_dim': getattr(args, 'probe_hidden_dim', args.discriminator_hidden_dim),
            'num_sensors': args.num_sensors,
            # 为了可复现/稳定做best选择：默认禁用dropout
            'dropout': getattr(args, 'probe_dropout', 0.0),
            'use_linear_probe': use_linear_probe  # 新增
        }
        probe_type = "Linear" if use_linear_probe else "MLP"
        print(f"🆕 [Validation] Using independent {probe_type} probe for domain invariance evaluation")
        print(f"    ✅ Fix: Collect all features → multi-split train/eval → aggregate")
    
    # 累积监控指标
    total_domain_acc = 0.0
    total_repr_std = 0.0
    total_same_sim = 0.0
    total_diff_sim = 0.0
    metric_count = 0
    
    # 详细域判别器指标累积
    total_domain_acc_gelsight = 0.0
    total_domain_acc_digit = 0.0
    total_domain_acc_duragel = 0.0
    total_domain_acc_std = 0.0
    total_domain_pred_ratio_0 = 0.0
    total_domain_pred_ratio_1 = 0.0
    total_domain_pred_ratio_2 = 0.0
    total_domain_entropy = 0.0
    domain_collapse_count = 0
    probe_collapse_score: Optional[float] = None
    probe_acc_std_over_splits = 0.0
    probe_loss_mean = 0.0
    probe_eval_splits = int(getattr(args, 'probe_eval_splits', 5))
    probe_train_ratio = float(getattr(args, 'probe_train_ratio', 0.8))
    independent_probe_used = False
    conditional_probe_acc_raw = 0.0
    conditional_probe_acc_residual = 0.0
    conditional_probe_acc_raw_std = 0.0
    conditional_probe_acc_residual_std = 0.0
    conditional_probe_acc_residual_gelsight = 0.0
    conditional_probe_acc_residual_digit = 0.0
    conditional_probe_acc_residual_duragel = 0.0
    conditional_probe_used_splits = 0
    pair_link_len_mean = 0.0
    pair_link_len_median = 0.0
    pair_link_len_p90 = 0.0
    pair_link_len_p95 = 0.0
    sensor_centroid_dist = None
    cross_sensor_knn_mix_rate = None
    
    use_group_dro = getattr(args, 'use_group_dro', False)
    total_loss_fine = 0.0
    total_loss_indoor = 0.0
    total_loss_outdoor = 0.0
    group_batch_count = 0
    avg_loss_fine = None
    avg_loss_indoor = None
    avg_loss_outdoor = None
    group_loss_gap = None
    
    # ====================================================================
    # 验证循环
    # ====================================================================
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        
        # 将数据移到设备
        for sensor in ['gelsight', 'digit', 'duragel']:
            batch[sensor] = batch[sensor].to(device, non_blocking=True)
        group_label = None
        if use_group_dro and 'group_label' in batch:
            group_label = batch['group_label'].to(device, non_blocking=True).long()
        
        # 前向传播
        z_contact_dict = {}
        z_domain_dict = {}
        h_dict = {}
        for sensor in ['gelsight', 'digit', 'duragel']:
            # 统一Stage-2输入策略: [B,T,C,H,W] -> [B,C,T,H,W]
            imgs = _prepare_stage2_video_input(
                batch[sensor],
                num_frames=stage2_num_frames
            )
            outputs = model(
                imgs,
                sensor_type=None,
                return_all=True,
                return_encoder_output=True
            )
            z_contact_dict[sensor] = outputs['z_contact']
            z_domain_dict[sensor] = outputs['z_domain']
            h_dict[sensor] = outputs['h']

        z_adv_dict = _get_adversarial_feature_dict(
            z_contact_dict,
            z_domain_dict,
            target=adversarial_feature_target
        )
        batch_size = z_domain_dict['gelsight'].shape[0]
        object_ids = batch.get('object', None)
        frame_ids = batch.get('frame', None)
        pair_confidence = batch.get('pair_confidence', None)
        if isinstance(pair_confidence, torch.Tensor):
            pair_confidence = pair_confidence.to(device=device, dtype=torch.float32).view(-1)
            pair_confidence_mean = pair_confidence.mean().item()
        else:
            pair_confidence_mean = 1.0
        domain_sample_weights = None
        domain_valid_mask = None
        if use_pair_confidence_for_domain_and_inter:
            conf_w, conf_valid = _build_pair_confidence_weights(
                pair_confidence,
                device=device,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min,
            )
            if conf_w is not None:
                domain_sample_weights = conf_w.repeat(3)
            if conf_valid is not None:
                domain_valid_mask = conf_valid.repeat(3)
        pair_ids_batch = _build_pair_ids(object_ids, frame_ids, batch_size)
        all_pair_ids.extend(pair_ids_batch)
        domain_labels = torch.cat([
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.long, device=device),
            torch.full((batch_size,), 2, dtype=torch.long, device=device)
        ])
        num_sensors = int(getattr(args, 'num_sensors', 3))
        digit_focus_weights = _build_digit_focus_weights(
            domain_labels=domain_labels,
            num_sensors=num_sensors,
            use_digit_focus=use_digit_focus_reweight,
            digit_focus_weight=digit_focus_weight,
        )
        domain_loss_weights = _combine_sample_weights(
            domain_sample_weights,
            digit_focus_weights
        )
        
        # 计算损失
        loss_mp = multi_positive_infonce_loss(
            z_contact_dict,
            temperature=temperature,
            use_semi_hard_negatives=use_semi_hard,
            object_ids=object_ids,
            frame_ids=frame_ids,
            exclude_same_object_negatives=exclude_same_object_negatives,
            prioritize_same_object_diff_frame_negatives=prioritize_same_object_diff_frame_negatives,
            same_object_negative_keep_ratio=same_object_negative_keep_ratio,
        )
        
        loss_im_contact = torch.tensor(0.0, device=device)
        loss_im_residual = torch.tensor(0.0, device=device)
        if lambda_im > 0 and model.discriminator is not None:
            if im_dual_target and args.stage == '2b':
                z_all_contact = torch.cat(
                    [z_contact_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
                z_residual_eval_dict = _compute_contact_residual_dict(z_contact_dict, normalize=False)
                z_all_residual = torch.cat(
                    [z_residual_eval_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
                log_probs_contact = model.discriminator(z_all_contact)
                per_sample_kl_contact = _infomax_kl_per_sample_from_log_probs(
                    log_probs_contact,
                    num_sensors=num_sensors
                )
                loss_im_contact = _weighted_mean(
                    per_sample_kl_contact,
                    weights=domain_loss_weights,
                    valid_mask=domain_valid_mask
                )
                log_probs_residual = model.discriminator(z_all_residual)
                per_sample_kl_residual = _infomax_kl_per_sample_from_log_probs(
                    log_probs_residual,
                    num_sensors=num_sensors
                )
                loss_im_residual = _weighted_mean(
                    per_sample_kl_residual,
                    weights=domain_loss_weights,
                    valid_mask=domain_valid_mask
                )
                loss_im = (
                    float(im_contact_ratio) * loss_im_contact +
                    (1.0 - float(im_contact_ratio)) * loss_im_residual
                )
            else:
                z_all = torch.cat([z_adv_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0)
                log_probs = model.discriminator(z_all)
                per_sample_kl = _infomax_kl_per_sample_from_log_probs(
                    log_probs,
                    num_sensors=num_sensors
                )
                loss_im = _weighted_mean(
                    per_sample_kl,
                    weights=domain_loss_weights,
                    valid_mask=domain_valid_mask
                )
                if adversarial_feature_target == 'contact':
                    loss_im_contact = loss_im
                elif adversarial_feature_target == 'contact_residual':
                    loss_im_residual = loss_im
        else:
            loss_im = torch.tensor(0.0, device=device)

        if lambda_pair > 0:
            loss_pair = _compute_rowwise_pair_loss(
                z_contact_dict,
                pair_loss_type=pair_loss_type,
                pair_confidence=pair_confidence,
                use_pair_confidence_weighting=use_pair_confidence_weighting,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min,
            )
        else:
            loss_pair = torch.tensor(0.0, device=device)

        if lambda_pair_tail > 0:
            loss_pair_tail = _compute_rowwise_pair_hard_tail_loss(
                z_contact_dict,
                pair_loss_type=pair_loss_type,
                pair_confidence=pair_confidence,
                use_pair_confidence_weighting=use_pair_confidence_weighting,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min,
                tail_ratio=pair_tail_ratio,
                tail_min_samples=pair_tail_min_samples,
            )
        else:
            loss_pair_tail = torch.tensor(0.0, device=device)

        if lambda_inter > 0:
            loss_inter = _compute_inter_contact_separation_loss(
                z_contact_dict,
                object_ids=object_ids,
                frame_ids=frame_ids,
                pair_confidence=pair_confidence,
                use_confidence_weighting=use_pair_confidence_for_domain_and_inter,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min if use_pair_confidence_for_domain_and_inter else 0.0,
                sim_margin=inter_sim_margin,
                hard_topk=inter_hard_topk,
            )
        else:
            loss_inter = torch.tensor(0.0, device=device)

        if lambda_inter_sensor > 0:
            loss_inter_sensor = _compute_inter_contact_separation_loss_per_sensor(
                z_contact_dict,
                object_ids=object_ids,
                frame_ids=frame_ids,
                pair_confidence=pair_confidence,
                use_confidence_weighting=use_pair_confidence_for_domain_and_inter,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min if use_pair_confidence_for_domain_and_inter else 0.0,
                sim_margin=inter_sim_margin,
                hard_topk=inter_hard_topk,
            )
        else:
            loss_inter_sensor = torch.tensor(0.0, device=device)

        if lambda_residual_compact > 0:
            loss_residual_compact = _compute_residual_compact_loss(
                z_contact_dict,
                pair_confidence=pair_confidence,
                use_pair_confidence_weighting=use_pair_confidence_weighting,
                confidence_floor=pair_confidence_floor,
                confidence_power=pair_confidence_power,
                confidence_min=pair_confidence_min,
            )
        else:
            loss_residual_compact = torch.tensor(0.0, device=device)

        if use_variance_reg and lambda_var > 0:
            current_std = compute_representation_std(z_contact_dict)
            loss_var = adaptive_variance_loss(
                z_contact_dict,
                target_std=target_std,
                current_std=current_std
            )
        else:
            loss_var = torch.tensor(0.0, device=device)

        if lambda_decouple > 0:
            loss_decouple = _compute_decoupling_loss(
                z_contact_dict,
                z_domain_dict,
                detach_domain=decouple_detach_domain
            )
        else:
            loss_decouple = torch.tensor(0.0, device=device)

        if lambda_coral > 0:
            loss_coral = _compute_sensor_coral_loss(
                z_contact_dict,
                include_mean=coral_include_mean
            )
        else:
            loss_coral = torch.tensor(0.0, device=device)

        if lambda_coral_residual > 0:
            loss_coral_residual = _compute_contact_residual_coral_loss(
                z_contact_dict,
                include_mean=coral_residual_include_mean
            )
        else:
            loss_coral_residual = torch.tensor(0.0, device=device)

        if (
            lambda_domain_cls > 0 and
            hasattr(model, 'domain_classifier') and
            model.domain_classifier is not None
        ):
            if domain_cls_detach_encoder:
                h_domain_all = torch.cat(
                    [h_dict[s].detach() for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
                z_domain_for_cls = model.domain_projection_head(h_domain_all)
            else:
                z_domain_for_cls = torch.cat(
                    [z_domain_dict[s] for s in ['gelsight', 'digit', 'duragel']], dim=0
                )
            log_probs_domain_cls = model.domain_classifier(z_domain_for_cls)
            per_sample_domain_cls = F.nll_loss(
                log_probs_domain_cls,
                domain_labels,
                reduction='none'
            )
            loss_domain_cls = _weighted_mean(
                per_sample_domain_cls,
                weights=domain_loss_weights,
                valid_mask=domain_valid_mask
            )
        else:
            loss_domain_cls = torch.tensor(0.0, device=device)

        loss = (
            loss_mp +
            lambda_im * im_loss_scale * loss_im +
            lambda_pair * loss_pair +
            lambda_pair_tail * loss_pair_tail +
            lambda_inter * loss_inter +
            lambda_inter_sensor * loss_inter_sensor +
            lambda_residual_compact * loss_residual_compact +
            lambda_var * loss_var +
            lambda_decouple * loss_decouple +
            lambda_domain_cls * loss_domain_cls +
            lambda_coral * loss_coral +
            lambda_coral_residual * loss_coral_residual
        )
        
        # 计算监控指标
        stats = compute_similarity_statistics(z_contact_dict)
        pair_link_len_list.append(_compute_pair_link_length_tensor(z_contact_dict).detach())
        
        # 🔧 BUG FIX 2: 收集特征（稍后统一训练/评估probe）
        if use_independent_probe and probe_config is not None:
            # 只收集特征，不在这里训练/评估
            all_z_dicts.append({
                'gelsight': z_contact_dict['gelsight'].detach().float().cpu(),
                'digit': z_contact_dict['digit'].detach().float().cpu(),
                'duragel': z_contact_dict['duragel'].detach().float().cpu()
            })
            domain_acc = 0.0  # 临时占位，稍后计算
        elif model.discriminator is not None:
            # 原版评估（Phase-1 或兼容模式）
            domain_metrics = compute_domain_metrics_detailed(z_adv_dict, model.discriminator)
            domain_acc = domain_metrics['domain_acc']
            
            # 累积详细域判别器指标
            total_domain_acc_gelsight += domain_metrics['domain_acc_gelsight']
            total_domain_acc_digit += domain_metrics['domain_acc_digit']
            total_domain_acc_duragel += domain_metrics['domain_acc_duragel']
            total_domain_acc_std += domain_metrics['domain_acc_std']
            total_domain_pred_ratio_0 += domain_metrics['domain_pred_ratio_0']
            total_domain_pred_ratio_1 += domain_metrics['domain_pred_ratio_1']
            total_domain_pred_ratio_2 += domain_metrics['domain_pred_ratio_2']
            total_domain_entropy += domain_metrics['domain_entropy']
            if domain_metrics['domain_is_collapsed']:
                domain_collapse_count += 1
        else:
            domain_acc = 0.0
        
        repr_std = compute_representation_std(z_contact_dict)
        
        # Group-DRO验证统计
        if group_label is not None:
            per_sample_loss = multi_positive_infonce_loss(
                z_contact_dict,
                temperature=temperature,
                use_semi_hard_negatives=use_semi_hard,
                object_ids=object_ids,
                frame_ids=frame_ids,
                exclude_same_object_negatives=exclude_same_object_negatives,
                prioritize_same_object_diff_frame_negatives=prioritize_same_object_diff_frame_negatives,
                same_object_negative_keep_ratio=same_object_negative_keep_ratio,
                reduction="none",
            )  # [B]

            loss_fine_batch = 0.0
            loss_indoor_batch = 0.0
            loss_outdoor_batch = 0.0
            for group_id in range(3):
                group_mask = (group_label == group_id)
                if group_mask.any():
                    loss_group = per_sample_loss[group_mask].mean().item()
                    if group_id == 0:
                        loss_fine_batch = loss_group
                    elif group_id == 1:
                        loss_indoor_batch = loss_group
                    else:
                        loss_outdoor_batch = loss_group
            total_loss_fine += loss_fine_batch
            total_loss_indoor += loss_indoor_batch
            total_loss_outdoor += loss_outdoor_batch
            group_batch_count += 1
        
        # 累积
        total_domain_acc += domain_acc
        total_repr_std += repr_std
        total_same_sim += stats['same_contact_sim_mean']
        total_diff_sim += stats['diff_contact_sim_mean']
        metric_count += 1
        
        # 更新 logger
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_mp=loss_mp.item())
        metric_logger.update(loss_im=loss_im.item() if isinstance(loss_im, torch.Tensor) else 0.0)
        metric_logger.update(
            loss_im_contact=loss_im_contact.item() if isinstance(loss_im_contact, torch.Tensor) else 0.0
        )
        metric_logger.update(
            loss_im_residual=loss_im_residual.item() if isinstance(loss_im_residual, torch.Tensor) else 0.0
        )
        metric_logger.update(loss_var=loss_var.item() if isinstance(loss_var, torch.Tensor) else 0.0)
        metric_logger.update(loss_pair=loss_pair.item() if isinstance(loss_pair, torch.Tensor) else 0.0)
        metric_logger.update(
            loss_pair_tail=loss_pair_tail.item() if isinstance(loss_pair_tail, torch.Tensor) else 0.0
        )
        metric_logger.update(loss_inter=loss_inter.item() if isinstance(loss_inter, torch.Tensor) else 0.0)
        metric_logger.update(
            loss_inter_sensor=(
                loss_inter_sensor.item() if isinstance(loss_inter_sensor, torch.Tensor) else 0.0
            )
        )
        metric_logger.update(
            loss_residual_compact=(
                loss_residual_compact.item() if isinstance(loss_residual_compact, torch.Tensor) else 0.0
            )
        )
        metric_logger.update(loss_decouple=loss_decouple.item() if isinstance(loss_decouple, torch.Tensor) else 0.0)
        metric_logger.update(loss_coral=loss_coral.item() if isinstance(loss_coral, torch.Tensor) else 0.0)
        metric_logger.update(
            loss_coral_residual=loss_coral_residual.item() if isinstance(loss_coral_residual, torch.Tensor) else 0.0
        )
        metric_logger.update(
            loss_domain_cls=loss_domain_cls.item() if isinstance(loss_domain_cls, torch.Tensor) else 0.0
        )
        metric_logger.update(pair_conf=pair_confidence_mean)
    
    # ====================================================================
    # 🔧 BUG FIX 2: 在收集完所有特征后，训练和评估独立probe
    # ====================================================================
    if use_independent_probe and probe_config is not None and len(all_z_dicts) > 0:
        independent_probe_used = True
        print(f"\n🔬 Training independent probe on collected features...")
        print(f"   Total local batches collected: {len(all_z_dicts)}")
        
        # 合并所有特征（当前rank）
        all_gelsight = torch.cat([z['gelsight'] for z in all_z_dicts], dim=0)
        all_digit = torch.cat([z['digit'] for z in all_z_dicts], dim=0)
        all_duragel = torch.cat([z['duragel'] for z in all_z_dicts], dim=0)
        
        # DDP下聚合所有rank特征与pair_ids
        all_gelsight = _gather_variable_length_tensor(all_gelsight)
        all_digit = _gather_variable_length_tensor(all_digit)
        all_duragel = _gather_variable_length_tensor(all_duragel)
        if misc.is_dist_avail_and_initialized():
            import torch.distributed as dist
            gathered_pair_ids = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_pair_ids, list(all_pair_ids))
            all_pair_ids = []
            for item in gathered_pair_ids:
                if item:
                    all_pair_ids.extend(item)
        
        total_samples = all_gelsight.shape[0]
        print(f"   Total samples per sensor (all ranks): {total_samples}")
        
        # 多次split，取均值/方差，提高best选择稳定性
        train_ratio = probe_train_ratio
        num_splits = max(1, int(probe_eval_splits))
        probe_eval_splits = num_splits
        
        probe_train_epochs = int(getattr(args, 'probe_train_epochs', 20))
        probe_batch_size = int(getattr(args, 'probe_batch_size', 256))
        probe_early_stop_patience = int(getattr(args, 'probe_early_stop_patience', 3))
        probe_early_stop_min_delta = float(getattr(args, 'probe_early_stop_min_delta', 1e-4))
        probe_lr = float(getattr(args, 'probe_lr', 1e-3))
        probe_weight_decay = float(getattr(args, 'probe_weight_decay', 0.0))
        
        seed_base = int(getattr(args, 'probe_seed_base', getattr(args, 'random_seed', 0)))
        
        probe_acc_mean = 0.333
        probe_acc_std_over_splits = 0.0
        probe_loss_mean = 0.0
        domain_acc_gelsight_mean = 0.0
        domain_acc_digit_mean = 0.0
        domain_acc_duragel_mean = 0.0
        domain_acc_std_mean = 0.0
        pred_ratio_0_mean = 1.0 / 3.0
        pred_ratio_1_mean = 1.0 / 3.0
        pred_ratio_2_mean = 1.0 / 3.0
        entropy_mean = 1.099
        collapse_fraction = 0.0
        collapse_score = 0.0
        
        if misc.is_main_process():
            split_metrics = []
            from model.domain_probe import IndependentDomainProbe, train_independent_probe, evaluate_independent_probe
            
            train_size = int(total_samples * train_ratio)
            train_size = max(1, min(train_size, total_samples - 1))
            test_size = total_samples - train_size
            print(f"   Split: train={train_size}, test={test_size} (ratio={train_ratio:.2f}), splits={num_splits}")
            
            for split_id in range(num_splits):
                # 每个split从零初始化probe，保证“独立评估”的可比性
                domain_probe = IndependentDomainProbe(**probe_config).to(device)
                probe_optimizer = torch.optim.Adam(
                    domain_probe.parameters(),
                    lr=probe_lr,
                    weight_decay=probe_weight_decay
                )
                
                index_device = all_gelsight.device
                if all_gelsight.is_cuda:
                    generator = torch.Generator(device=index_device)
                else:
                    generator = torch.Generator()
                generator.manual_seed(seed_base + split_id)
                indices = torch.randperm(total_samples, generator=generator, device=index_device)
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                
                train_z_dict = {
                    'gelsight': all_gelsight[train_indices],
                    'digit': all_digit[train_indices],
                    'duragel': all_duragel[train_indices]
                }
                test_z_dict = {
                    'gelsight': all_gelsight[test_indices],
                    'digit': all_digit[test_indices],
                    'duragel': all_duragel[test_indices]
                }
                
                with torch.enable_grad():
                    train_stats = train_independent_probe(
                        train_z_dict,
                        domain_probe,
                        probe_optimizer,
                        num_epochs=probe_train_epochs,
                        batch_size=probe_batch_size,
                        early_stop_patience=probe_early_stop_patience,
                        early_stop_min_delta=probe_early_stop_min_delta,
                        shuffle=True
                    )
                
                probe_metrics = evaluate_independent_probe(test_z_dict, domain_probe)
                probe_metrics['train_loss'] = train_stats.get('best_train_loss', train_stats.get('train_loss', 0.0))
                split_metrics.append(probe_metrics)
            
            # 聚合统计（用于best选择的标量指标）
            def _mean(key: str, default: float = 0.0) -> float:
                vals = [float(m.get(key, default)) for m in split_metrics]
                return float(sum(vals) / max(len(vals), 1))
            
            def _std(key: str, default: float = 0.0) -> float:
                vals = [float(m.get(key, default)) for m in split_metrics]
                if len(vals) <= 1:
                    return 0.0
                mu = sum(vals) / len(vals)
                return float((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)
            
            probe_acc_mean = _mean('probe_acc', 0.333)
            probe_acc_std_over_splits = _std('probe_acc', 0.333)
            probe_loss_mean = _mean('probe_loss', 0.0)
            domain_acc_gelsight_mean = _mean('probe_acc_gelsight', 0.0)
            domain_acc_digit_mean = _mean('probe_acc_digit', 0.0)
            domain_acc_duragel_mean = _mean('probe_acc_duragel', 0.0)
            domain_acc_std_mean = _mean('probe_acc_std', 0.0)
            pred_ratio_0_mean = _mean('probe_pred_ratio_0', 1.0 / 3.0)
            pred_ratio_1_mean = _mean('probe_pred_ratio_1', 1.0 / 3.0)
            pred_ratio_2_mean = _mean('probe_pred_ratio_2', 1.0 / 3.0)
            entropy_mean = _mean('probe_entropy', 1.099)
            collapse_fraction = _mean('probe_is_collapsed', 0.0)
            collapse_score = _mean('probe_collapse_ratio', 0.0)
            probe_collapse_score = float(collapse_score)

            # 条件probe（同pair分组切分）:
            # - raw: 直接在z_contact上测“在接触条件下还能否识别sensor”
            # - residual: 在z-contact中心化残差上测，衡量局部传感器痕迹
            z_all_dict = {
                'gelsight': all_gelsight,
                'digit': all_digit,
                'duragel': all_duragel
            }
            sensor_centroid_dist = _compute_sensor_centroid_distance(z_all_dict)
            cross_sensor_knn_mix_rate = _compute_cross_sensor_knn_mix_rate(
                z_all_dict,
                k=global_mix_knn_k,
                max_samples_per_sensor=global_mix_knn_max_samples,
                seed=seed_base + 2027,
            )
            if len(all_pair_ids) == total_samples:
                raw_probe_metrics = _evaluate_conditional_probe_grouped(
                    z_all_dict,
                    all_pair_ids,
                    probe_config,
                    device,
                    train_ratio,
                    num_splits,
                    seed_base,
                    probe_train_epochs,
                    probe_batch_size,
                    probe_early_stop_patience,
                    probe_early_stop_min_delta,
                    probe_lr,
                    probe_weight_decay
                )
                z_residual_dict = _compute_contact_residual_dict(z_all_dict, normalize=False)
                residual_probe_metrics = _evaluate_conditional_probe_grouped(
                    z_residual_dict,
                    all_pair_ids,
                    probe_config,
                    device,
                    train_ratio,
                    num_splits,
                    seed_base + 997,
                    probe_train_epochs,
                    probe_batch_size,
                    probe_early_stop_patience,
                    probe_early_stop_min_delta,
                    probe_lr,
                    probe_weight_decay
                )
                conditional_probe_acc_raw = float(raw_probe_metrics.get('acc', 0.0))
                conditional_probe_acc_raw_std = float(raw_probe_metrics.get('acc_std', 0.0))
                used_raw = int(raw_probe_metrics.get('used_splits', 0))

                conditional_probe_acc_residual = float(residual_probe_metrics.get('acc', 0.0))
                conditional_probe_acc_residual_std = float(residual_probe_metrics.get('acc_std', 0.0))
                conditional_probe_acc_residual_gelsight = float(
                    residual_probe_metrics.get('acc_gelsight', 0.0)
                )
                conditional_probe_acc_residual_digit = float(
                    residual_probe_metrics.get('acc_digit', 0.0)
                )
                conditional_probe_acc_residual_duragel = float(
                    residual_probe_metrics.get('acc_duragel', 0.0)
                )
                used_residual = int(residual_probe_metrics.get('used_splits', 0))
                conditional_probe_used_splits = min(used_raw, used_residual)
            else:
                print(
                    f"   ⚠️  Skip conditional probe: pair_ids length mismatch "
                    f"({len(all_pair_ids)} vs {total_samples})"
                )

        # 将rank0的聚合结果广播给所有rank（避免每卡都训练probe）
        if misc.is_dist_avail_and_initialized():
            import torch.distributed as dist
            packed = torch.tensor([
                probe_acc_mean,
                probe_acc_std_over_splits,
                probe_loss_mean,
                domain_acc_gelsight_mean,
                domain_acc_digit_mean,
                domain_acc_duragel_mean,
                domain_acc_std_mean,
                pred_ratio_0_mean,
                pred_ratio_1_mean,
                pred_ratio_2_mean,
                entropy_mean,
                collapse_fraction,
                collapse_score,
                conditional_probe_acc_raw,
                conditional_probe_acc_residual,
                conditional_probe_acc_raw_std,
                conditional_probe_acc_residual_std,
                conditional_probe_acc_residual_gelsight,
                conditional_probe_acc_residual_digit,
                conditional_probe_acc_residual_duragel,
                float(conditional_probe_used_splits),
                float('nan') if sensor_centroid_dist is None else float(sensor_centroid_dist),
                float('nan') if cross_sensor_knn_mix_rate is None else float(cross_sensor_knn_mix_rate),
            ], device=device, dtype=torch.float32)
            dist.broadcast(packed, src=0)
            probe_acc_mean = float(packed[0].item())
            probe_acc_std_over_splits = float(packed[1].item())
            probe_loss_mean = float(packed[2].item())
            domain_acc_gelsight_mean = float(packed[3].item())
            domain_acc_digit_mean = float(packed[4].item())
            domain_acc_duragel_mean = float(packed[5].item())
            domain_acc_std_mean = float(packed[6].item())
            pred_ratio_0_mean = float(packed[7].item())
            pred_ratio_1_mean = float(packed[8].item())
            pred_ratio_2_mean = float(packed[9].item())
            entropy_mean = float(packed[10].item())
            collapse_fraction = float(packed[11].item())
            collapse_score = float(packed[12].item())
            conditional_probe_acc_raw = float(packed[13].item())
            conditional_probe_acc_residual = float(packed[14].item())
            conditional_probe_acc_raw_std = float(packed[15].item())
            conditional_probe_acc_residual_std = float(packed[16].item())
            conditional_probe_acc_residual_gelsight = float(packed[17].item())
            conditional_probe_acc_residual_digit = float(packed[18].item())
            conditional_probe_acc_residual_duragel = float(packed[19].item())
            conditional_probe_used_splits = int(round(float(packed[20].item())))
            sensor_centroid_dist = float(packed[21].item())
            cross_sensor_knn_mix_rate = float(packed[22].item())
            if not math.isfinite(sensor_centroid_dist):
                sensor_centroid_dist = None
            if not math.isfinite(cross_sensor_knn_mix_rate):
                cross_sensor_knn_mix_rate = None
            probe_collapse_score = float(collapse_score)
        
        # 更新累积指标（使用聚合后的结果，保证可比且稳定）
        total_domain_acc_gelsight = domain_acc_gelsight_mean * metric_count
        total_domain_acc_digit = domain_acc_digit_mean * metric_count
        total_domain_acc_duragel = domain_acc_duragel_mean * metric_count
        total_domain_acc_std = domain_acc_std_mean * metric_count
        total_domain_pred_ratio_0 = pred_ratio_0_mean * metric_count
        total_domain_pred_ratio_1 = pred_ratio_1_mean * metric_count
        total_domain_pred_ratio_2 = pred_ratio_2_mean * metric_count
        total_domain_entropy = entropy_mean * metric_count
        total_domain_acc = probe_acc_mean * metric_count
        
        # 修复：不再二值放大；collapse_ratio 表示“split中崩溃比例”
        domain_collapse_count = collapse_fraction * metric_count
        
        print(f"   ✅ Independent probe acc: {probe_acc_mean*100:.2f}% (±{probe_acc_std_over_splits*100:.2f} over {num_splits} splits)")
        print(f"      Per-sensor: GelSight={domain_acc_gelsight_mean*100:.1f}%, "
              f"DIGIT={domain_acc_digit_mean*100:.1f}%, "
              f"DuraGel={domain_acc_duragel_mean*100:.1f}%")
        print(f"      Probe loss (test): {probe_loss_mean:.4f} | Entropy: {entropy_mean:.4f} | Collapse: {collapse_fraction*100:.1f}% (score={collapse_score:.3f})")
        if conditional_probe_used_splits > 0:
            print(
                f"      Conditional probe raw: {conditional_probe_acc_raw*100:.2f}% "
                f"(±{conditional_probe_acc_raw_std*100:.2f})"
            )
            print(
                f"      Conditional probe residual: {conditional_probe_acc_residual*100:.2f}% "
                f"(±{conditional_probe_acc_residual_std*100:.2f})"
            )
            print(
                "      Conditional residual per-sensor: "
                f"GelSight={conditional_probe_acc_residual_gelsight*100:.2f}%, "
                f"DIGIT={conditional_probe_acc_residual_digit*100:.2f}%, "
                f"DuraGel={conditional_probe_acc_residual_duragel*100:.2f}%"
            )
        if sensor_centroid_dist is not None and cross_sensor_knn_mix_rate is not None:
            print(
                f"      Global mix: centroid_dist={sensor_centroid_dist:.4f}, "
                f"cross_sensor_knn_mix@{global_mix_knn_k}={cross_sensor_knn_mix_rate:.4f}"
            )

    # 高维“同位置跨传感器连线长度”分布（越小越好）
    pair_link_tensor = (
        torch.cat(pair_link_len_list, dim=0).float()
        if len(pair_link_len_list) > 0
        else torch.zeros((0,), device=device, dtype=torch.float32)
    )
    pair_link_tensor = _gather_variable_length_tensor(pair_link_tensor)
    if pair_link_tensor.numel() > 0:
        pair_link_len_mean = float(pair_link_tensor.mean().item())
        sorted_link = torch.sort(pair_link_tensor).values
        pair_link_len_median = _quantile_from_sorted(sorted_link, 0.5)
        pair_link_len_p90 = _quantile_from_sorted(sorted_link, 0.9)
        pair_link_len_p95 = _quantile_from_sorted(sorted_link, 0.95)
    
    # ====================================================================
    # 汇总结果
    # ====================================================================
    metric_logger.synchronize_between_processes()
    
    # 分布式下确保监控指标完全一致（总和/计数先 all_reduce，再统一求平均）
    if misc.is_dist_avail_and_initialized():
        import torch.distributed as dist
        
        reduce_vec = torch.tensor([
            total_domain_acc,
            total_repr_std,
            total_same_sim,
            total_diff_sim,
            total_domain_acc_gelsight,
            total_domain_acc_digit,
            total_domain_acc_duragel,
            total_domain_acc_std,
            total_domain_pred_ratio_0,
            total_domain_pred_ratio_1,
            total_domain_pred_ratio_2,
            total_domain_entropy,
            domain_collapse_count,
            metric_count,
            total_loss_fine,
            total_loss_indoor,
            total_loss_outdoor,
            group_batch_count
        ], device=device, dtype=torch.float64)
        dist.all_reduce(reduce_vec, op=dist.ReduceOp.SUM)
        
        (
            total_domain_acc,
            total_repr_std,
            total_same_sim,
            total_diff_sim,
            total_domain_acc_gelsight,
            total_domain_acc_digit,
            total_domain_acc_duragel,
            total_domain_acc_std,
            total_domain_pred_ratio_0,
            total_domain_pred_ratio_1,
            total_domain_pred_ratio_2,
            total_domain_entropy,
            domain_collapse_count,
            metric_count,
            total_loss_fine,
            total_loss_indoor,
            total_loss_outdoor,
            group_batch_count
        ) = [x.item() for x in reduce_vec]
    
    metric_count = max(int(round(metric_count)), 1)
    group_batch_count = max(int(round(group_batch_count)), 0)
    
    avg_domain_acc = total_domain_acc / max(metric_count, 1)
    avg_repr_std = total_repr_std / max(metric_count, 1)
    avg_same_sim = total_same_sim / max(metric_count, 1)
    avg_diff_sim = total_diff_sim / max(metric_count, 1)
    
    # 计算详细域判别器指标的平均值
    avg_domain_acc_gelsight = total_domain_acc_gelsight / max(metric_count, 1)
    avg_domain_acc_digit = total_domain_acc_digit / max(metric_count, 1)
    avg_domain_acc_duragel = total_domain_acc_duragel / max(metric_count, 1)
    avg_domain_acc_std = total_domain_acc_std / max(metric_count, 1)
    avg_domain_pred_ratio_0 = total_domain_pred_ratio_0 / max(metric_count, 1)
    avg_domain_pred_ratio_1 = total_domain_pred_ratio_1 / max(metric_count, 1)
    avg_domain_pred_ratio_2 = total_domain_pred_ratio_2 / max(metric_count, 1)
    avg_domain_entropy = total_domain_entropy / max(metric_count, 1)
    domain_collapse_ratio = domain_collapse_count / max(metric_count, 1)

    if probe_collapse_score is None:
        probe_collapse_score = float(domain_collapse_ratio)
    
    print(f"\n📊 Validation Summary:")
    print(f"   {metric_logger}")
    print(f"   Domain probe accuracy: {avg_domain_acc*100:.2f}%")
    print(f"   Representation std: {avg_repr_std:.4f}")
    print(f"   Same contact similarity: {avg_same_sim:.4f}")
    print(f"   Diff contact similarity: {avg_diff_sim:.4f}")
    print(
        f"   Pair link length: mean={pair_link_len_mean:.4f}, "
        f"median={pair_link_len_median:.4f}, p90={pair_link_len_p90:.4f}, p95={pair_link_len_p95:.4f}"
    )
    if sensor_centroid_dist is not None and cross_sensor_knn_mix_rate is not None:
        print(
            f"   Global mix: centroid_dist={sensor_centroid_dist:.4f}, "
            f"cross_sensor_knn_mix@{global_mix_knn_k}={cross_sensor_knn_mix_rate:.4f}"
        )
    
    if use_group_dro and group_batch_count > 0:
        avg_loss_fine = total_loss_fine / group_batch_count
        avg_loss_indoor = total_loss_indoor / group_batch_count
        avg_loss_outdoor = total_loss_outdoor / group_batch_count
        group_loss_gap = max(avg_loss_fine, avg_loss_indoor, avg_loss_outdoor) - \
                         min(avg_loss_fine, avg_loss_indoor, avg_loss_outdoor)
        print(f"   [Group-DRO] Fine: {avg_loss_fine:.4f} | Indoor: {avg_loss_indoor:.4f} | Outdoor: {avg_loss_outdoor:.4f}")
        print(f"   [Group-DRO] Loss Gap: {group_loss_gap:.4f}")
    
    # 打印详细域判别器指标
    if model.discriminator is not None:
        print(f"   [Domain Detailed] GelSight: {avg_domain_acc_gelsight*100:.2f}% | DIGIT: {avg_domain_acc_digit*100:.2f}% | DuraGel: {avg_domain_acc_duragel*100:.2f}%")
        print(f"   [Domain Detailed] Std: {avg_domain_acc_std:.4f} | Entropy: {avg_domain_entropy:.4f} (max=1.099)")
        print(f"   [Domain Detailed] Pred Ratios: [{avg_domain_pred_ratio_0:.3f}, {avg_domain_pred_ratio_1:.3f}, {avg_domain_pred_ratio_2:.3f}]")
        if domain_collapse_ratio > 0.1:
            print(f"   🚨 [WARNING] Domain collapse detected in {domain_collapse_ratio*100:.1f}% of batches!")
    
    # Wandb 日志
    if wandb_log:
        try:
            import wandb
            current_epoch = getattr(args, 'current_epoch', None)
            log_dict = {
                'val/loss': metric_logger.meters['loss'].global_avg,
                'val/loss_mp': metric_logger.meters['loss_mp'].global_avg,
                'val/loss_im': metric_logger.meters['loss_im'].global_avg,
                'val/loss_im_contact': metric_logger.meters['loss_im_contact'].global_avg,
                'val/loss_im_residual': metric_logger.meters['loss_im_residual'].global_avg,
                'val/loss_im_scaled': lambda_im * im_loss_scale * metric_logger.meters['loss_im'].global_avg,
                'val/loss_var': metric_logger.meters['loss_var'].global_avg,
                'val/loss_pair': metric_logger.meters['loss_pair'].global_avg,
                'val/loss_pair_tail': metric_logger.meters['loss_pair_tail'].global_avg,
                'val/loss_inter': metric_logger.meters['loss_inter'].global_avg,
                'val/loss_inter_sensor': metric_logger.meters['loss_inter_sensor'].global_avg,
                'val/loss_residual_compact': metric_logger.meters['loss_residual_compact'].global_avg,
                'val/loss_decouple': metric_logger.meters['loss_decouple'].global_avg,
                'val/loss_domain_cls': metric_logger.meters['loss_domain_cls'].global_avg,
                'val/loss_coral': metric_logger.meters['loss_coral'].global_avg,
                'val/loss_coral_residual': metric_logger.meters['loss_coral_residual'].global_avg,
                'val/pair_confidence_mean': metric_logger.meters['pair_conf'].global_avg,
                'val/domain_acc': avg_domain_acc,
                'val/repr_std': avg_repr_std,
                'val/same_contact_sim': avg_same_sim,
                'val/diff_contact_sim': avg_diff_sim,
                'val/pair_link_len_mean': pair_link_len_mean,
                'val/pair_link_len_median': pair_link_len_median,
                'val/pair_link_len_p90': pair_link_len_p90,
                'val/pair_link_len_p95': pair_link_len_p95,
                # 详细域判别器指标
                'val/domain_acc_gelsight': avg_domain_acc_gelsight,
                'val/domain_acc_digit': avg_domain_acc_digit,
                'val/domain_acc_duragel': avg_domain_acc_duragel,
                'val/domain_acc_std': avg_domain_acc_std,
                'val/domain_pred_ratio_0': avg_domain_pred_ratio_0,
                'val/domain_pred_ratio_1': avg_domain_pred_ratio_1,
                'val/domain_pred_ratio_2': avg_domain_pred_ratio_2,
                'val/domain_entropy': avg_domain_entropy,
                'val/domain_collapse_ratio': domain_collapse_ratio,
                'val/independent_probe_enabled': 1.0 if independent_probe_used else 0.0
            }
            if sensor_centroid_dist is not None:
                log_dict['val/sensor_centroid_dist'] = float(sensor_centroid_dist)
            if cross_sensor_knn_mix_rate is not None:
                log_dict['val/cross_sensor_knn_mix_rate'] = float(cross_sensor_knn_mix_rate)
            log_dict['val/im_contact_ratio'] = float(im_contact_ratio)
            log_dict['val/digit_focus_weight'] = float(digit_focus_weight)
            log_dict['val/adversarial_target_code'] = {
                'domain': 0.0,
                'contact': 1.0,
                'contact_residual': 2.0
            }.get(adversarial_feature_target, -1.0)
            if independent_probe_used:
                log_dict.update({
                    'val/independent_probe_acc': avg_domain_acc,
                    'val/independent_probe_acc_std': probe_acc_std_over_splits,
                    'val/independent_probe_loss': probe_loss_mean,
                    'val/independent_probe_eval_splits': float(probe_eval_splits),
                    'val/independent_probe_train_ratio': probe_train_ratio,
                    'val/independent_probe_collapse_score': float(probe_collapse_score),
                    'val/independent_probe_collapse_fraction': domain_collapse_ratio
                })
                if conditional_probe_used_splits > 0:
                    log_dict.update({
                        'val/conditional_probe_acc_raw': conditional_probe_acc_raw,
                        'val/conditional_probe_acc_residual': conditional_probe_acc_residual,
                        'val/conditional_probe_acc_raw_std': conditional_probe_acc_raw_std,
                        'val/conditional_probe_acc_residual_std': conditional_probe_acc_residual_std,
                        'val/conditional_probe_acc_residual_gelsight': conditional_probe_acc_residual_gelsight,
                        'val/conditional_probe_acc_residual_digit': conditional_probe_acc_residual_digit,
                        'val/conditional_probe_acc_residual_duragel': conditional_probe_acc_residual_duragel,
                        'val/conditional_probe_used_splits': float(conditional_probe_used_splits),
                    })
            if current_epoch is not None:
                log_dict['epoch'] = current_epoch
            if group_loss_gap is not None:
                log_dict.update({
                    'val/group_loss_gap': group_loss_gap,
                    'val/group_loss_fine': avg_loss_fine,
                    'val/group_loss_indoor': avg_loss_indoor,
                    'val/group_loss_outdoor': avg_loss_outdoor
                })
            wandb.log(log_dict)
        except ImportError:
            pass
    
    result = {
        'loss': metric_logger.meters['loss'].global_avg,
        'loss_mp': metric_logger.meters['loss_mp'].global_avg,
        'loss_im': metric_logger.meters['loss_im'].global_avg,
        'loss_im_contact': metric_logger.meters['loss_im_contact'].global_avg,
        'loss_im_residual': metric_logger.meters['loss_im_residual'].global_avg,
        'loss_var': metric_logger.meters['loss_var'].global_avg,
        'loss_pair': metric_logger.meters['loss_pair'].global_avg,
        'loss_pair_tail': metric_logger.meters['loss_pair_tail'].global_avg,
        'loss_inter': metric_logger.meters['loss_inter'].global_avg,
        'loss_inter_sensor': metric_logger.meters['loss_inter_sensor'].global_avg,
        'loss_residual_compact': metric_logger.meters['loss_residual_compact'].global_avg,
        'loss_decouple': metric_logger.meters['loss_decouple'].global_avg,
        'loss_domain_cls': metric_logger.meters['loss_domain_cls'].global_avg,
        'loss_coral': metric_logger.meters['loss_coral'].global_avg,
        'loss_coral_residual': metric_logger.meters['loss_coral_residual'].global_avg,
        'pair_confidence': metric_logger.meters['pair_conf'].global_avg,
        'domain_acc': avg_domain_acc,
        'repr_std': avg_repr_std,
        'same_contact_sim': avg_same_sim,
        'diff_contact_sim': avg_diff_sim,
        'pair_link_len_mean': pair_link_len_mean,
        'pair_link_len_median': pair_link_len_median,
        'pair_link_len_p90': pair_link_len_p90,
        'pair_link_len_p95': pair_link_len_p95,
        'sensor_centroid_dist': sensor_centroid_dist,
        'cross_sensor_knn_mix_rate': cross_sensor_knn_mix_rate,
        # 详细域判别器指标
        'domain_acc_gelsight': avg_domain_acc_gelsight,
        'domain_acc_digit': avg_domain_acc_digit,
        'domain_acc_duragel': avg_domain_acc_duragel,
        'domain_acc_std': avg_domain_acc_std,
        'domain_pred_ratio_0': avg_domain_pred_ratio_0,
        'domain_pred_ratio_1': avg_domain_pred_ratio_1,
        'domain_pred_ratio_2': avg_domain_pred_ratio_2,
        'domain_entropy': avg_domain_entropy,
        'domain_collapse_ratio': domain_collapse_ratio,
        'independent_probe_enabled': bool(independent_probe_used),
        'independent_probe_acc': avg_domain_acc if independent_probe_used else None,
        'independent_probe_acc_std': probe_acc_std_over_splits if independent_probe_used else None,
        'independent_probe_loss': probe_loss_mean if independent_probe_used else None,
        'independent_probe_eval_splits': probe_eval_splits if independent_probe_used else None,
        'independent_probe_train_ratio': probe_train_ratio if independent_probe_used else None,
        'adversarial_target_active': adversarial_feature_target,
        'conditional_probe_acc_raw': conditional_probe_acc_raw if conditional_probe_used_splits > 0 else None,
        'conditional_probe_acc_residual': (
            conditional_probe_acc_residual if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_acc_raw_std': (
            conditional_probe_acc_raw_std if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_acc_residual_std': (
            conditional_probe_acc_residual_std if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_acc_residual_gelsight': (
            conditional_probe_acc_residual_gelsight if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_acc_residual_digit': (
            conditional_probe_acc_residual_digit if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_acc_residual_duragel': (
            conditional_probe_acc_residual_duragel if conditional_probe_used_splits > 0 else None
        ),
        'conditional_probe_used_splits': conditional_probe_used_splits if conditional_probe_used_splits > 0 else 0,
        'im_contact_ratio': float(im_contact_ratio),
        'digit_focus_weight': float(digit_focus_weight),
        # 🆕 崩溃检测指标（用于lambda退火）
        'probe_is_collapsed': domain_collapse_ratio > float(getattr(args, 'probe_collapse_fraction_threshold', 0.5)),
        'probe_collapse_ratio': float(probe_collapse_score),
        'probe_collapse_fraction': domain_collapse_ratio
    }
    
    if group_loss_gap is not None:
        result.update({
            'group_loss_gap': group_loss_gap,
            'group_loss_fine': avg_loss_fine,
            'group_loss_indoor': avg_loss_indoor,
            'group_loss_outdoor': avg_loss_outdoor
        })
    
    return result
