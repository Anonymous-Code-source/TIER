#!/usr/bin/env python3
"""
Training/evaluation engine for Stage3 V3 Symbolize + Reasoner.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _prepare_stage3_video_input(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize Stage3 touch clips to Stage2/Stage1 expected layout.

    Supported:
      - [B, T, C, H, W] -> [B, C, T, H, W]
      - [B, C, T, H, W] (already normalized)
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got shape={getattr(x, 'shape', None)}")

    if int(x.shape[1]) == 3:
        return x.contiguous()

    if int(x.shape[2]) == 3:
        return x.permute(0, 2, 1, 3, 4).contiguous()

    raise ValueError(
        f"Unsupported touch layout {tuple(x.shape)}. "
        "Expected channel dim=3 at dim1 or dim2."
    )


@torch.no_grad()
def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def _new_confusion(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)


@torch.no_grad()
def _update_confusion(conf: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    n_cls = int(conf.shape[0])
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()
    valid = (targets >= 0) & (targets < n_cls) & (preds >= 0) & (preds < n_cls)
    if not bool(valid.any()):
        return
    idx = targets[valid] * n_cls + preds[valid]
    binc = torch.bincount(idx, minlength=n_cls * n_cls).to(conf.dtype)
    conf += binc.view(n_cls, n_cls)


@torch.no_grad()
def _confusion_to_acc_macro_f1(conf: torch.Tensor, eps: float = 1e-8) -> List[float]:
    total = conf.sum().item()
    acc = float(conf.diag().sum().item() / total) if total > 0 else 0.0

    f1_list: List[float] = []
    n_cls = int(conf.shape[0])
    for c in range(n_cls):
        tp = conf[c, c].item()
        support = conf[c, :].sum().item()
        if support == 0:
            continue
        fp = conf[:, c].sum().item() - tp
        fn = support - tp
        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)
        f1 = (2.0 * precision * recall) / max(precision + recall, eps)
        f1_list.append(float(f1))
    macro_f1 = float(sum(f1_list) / len(f1_list)) if len(f1_list) > 0 else 0.0
    return [acc, macro_f1]


@torch.no_grad()
def _reduce_if_distributed(t: torch.Tensor) -> torch.Tensor:
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    touch = batch["touch"].to(device, non_blocking=True)
    return {
        "touch": _prepare_stage3_video_input(touch),
        "material": batch["material"].to(device, non_blocking=True).long(),
        "hardness": batch["hardness"].to(device, non_blocking=True).long(),
        "roughness": batch["roughness"].to(device, non_blocking=True).long(),
    }


def _format_eta(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "--:--"
    total = int(round(seconds))
    mm, ss = divmod(total, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _kl_div_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


@torch.no_grad()
def _update_symbol_hist(hist: torch.Tensor, symbol_ids: torch.Tensor) -> None:
    symbol_k = int(hist.shape[0])
    ids = symbol_ids.view(-1).long()
    valid = (ids >= 0) & (ids < symbol_k)
    if not bool(valid.any()):
        return
    hist += torch.bincount(ids[valid], minlength=symbol_k).to(hist.dtype)


def _compute_symbol_health(
    symbol_hist: torch.Tensor,
    symbol_prob_sum: torch.Tensor,
    sum_sample_entropy: float,
    total_symbol_samples: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    symbol_k = int(symbol_hist.shape[0])
    total = max(float(total_symbol_samples), 1.0)
    if symbol_k <= 0:
        return {
            "active_symbol_count": 0.0,
            "symbol_usage_ratio": 0.0,
            "top1_symbol_collapse_ratio": 1.0,
            "symbol_perplexity": 1.0,
            "symbol_perplexity_ratio": 1.0,
            "mean_sample_entropy_norm": 0.0,
        }

    active_count = int((symbol_hist > 0).sum().item())
    usage_ratio = float(active_count / max(symbol_k, 1))
    top1_ratio = float(symbol_hist.max().item() / total)

    p_bar = symbol_prob_sum / total
    p_bar = p_bar / p_bar.sum().clamp_min(eps)
    entropy_bar = float(_entropy_from_probs(p_bar, eps=eps).item())
    perplexity = float(math.exp(entropy_bar))
    perplexity_ratio = float(perplexity / max(symbol_k, 1))

    mean_sample_entropy = float(sum_sample_entropy / total)
    entropy_denom = math.log(symbol_k) if symbol_k > 1 else 1.0
    mean_sample_entropy_norm = float(mean_sample_entropy / max(entropy_denom, eps))

    return {
        "active_symbol_count": float(active_count),
        "symbol_usage_ratio": usage_ratio,
        "top1_symbol_collapse_ratio": top1_ratio,
        "symbol_perplexity": perplexity,
        "symbol_perplexity_ratio": perplexity_ratio,
        "mean_sample_entropy_norm": mean_sample_entropy_norm,
    }


def _compute_reason_batch_stats(
    out: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    ratios = []
    delta_norms = []
    for task in ("material", "hardness", "roughness"):
        final_logits = out[f"logits_{task}"]
        unary_logits = out[f"logits_{task}_unary"]
        delta = final_logits - unary_logits
        ratios.append(delta.norm(dim=-1) / final_logits.norm(dim=-1).clamp_min(eps))
        delta_norms.append(delta.norm(dim=-1))

    reason_contrib_ratio = torch.stack(ratios, dim=0).mean(dim=0).mean()
    message_delta_norm = torch.stack(delta_norms, dim=0).mean(dim=0).mean()
    return {
        "reason_contrib_ratio": reason_contrib_ratio,
        "message_delta_norm": message_delta_norm,
    }


def _get_relation_stats(
    model: torch.nn.Module,
    threshold: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    core = _unwrap_model(model)
    return {
        "relation_l1": core.relation_l1(),
        "relation_abs_mean": core.relation_abs_mean(),
        "relation_sparsity": core.relation_sparsity(threshold=threshold),
    }


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    num_material: int = 0,
    num_hardness: int = 0,
    num_roughness: int = 0,
    weight_material: torch.Tensor | None = None,
    weight_hardness: torch.Tensor | None = None,
    weight_roughness: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    loss_weight_material: float = 1.0,
    loss_weight_hardness: float = 1.0,
    loss_weight_roughness: float = 1.0,
    symbol_k: int = 64,
    symbol_temperature: float = 1.0,
    lambda_conf: float = 1e-2,
    lambda_bal: float = 2e-2,
    lambda_sparse: float = 1e-4,
    lambda_iter: float = 0.0,
    relation_sparsity_threshold: float = 1e-3,
    log_freq: int = 50,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_loss_task = 0.0
    total_loss_m = 0.0
    total_loss_h = 0.0
    total_loss_r = 0.0
    total_loss_conf = 0.0
    total_loss_bal = 0.0
    total_loss_sparse = 0.0
    total_loss_iter = 0.0
    total_reason_contrib = 0.0
    total_message_delta_norm = 0.0
    total_relation_abs_mean = 0.0
    total_relation_sparsity = 0.0
    total_samples = 0

    conf_m = _new_confusion(max(num_material, 1), device)
    conf_h = _new_confusion(max(num_hardness, 1), device)
    conf_r = _new_confusion(max(num_roughness, 1), device)

    symbol_hist = torch.zeros(max(symbol_k, 1), dtype=torch.float64, device=device)
    symbol_prob_sum = torch.zeros(max(symbol_k, 1), dtype=torch.float64, device=device)
    sum_sample_entropy = torch.zeros((), dtype=torch.float64, device=device)
    total_symbol_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    start_time = time.time()

    for step, batch in enumerate(data_loader, start=1):
        b = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(b["touch"], symbol_temperature=symbol_temperature)
            loss_m = F.cross_entropy(
                out["logits_material"],
                b["material"],
                weight=weight_material,
                label_smoothing=label_smoothing,
            )
            loss_h = F.cross_entropy(
                out["logits_hardness"],
                b["hardness"],
                weight=weight_hardness,
                label_smoothing=label_smoothing,
            )
            loss_r = F.cross_entropy(
                out["logits_roughness"],
                b["roughness"],
                weight=weight_roughness,
                label_smoothing=label_smoothing,
            )
            loss_task = (
                float(loss_weight_material) * loss_m
                + float(loss_weight_hardness) * loss_h
                + float(loss_weight_roughness) * loss_r
            )

            symbol_probs = out["symbol_probs"]
            sample_entropy = _entropy_from_probs(symbol_probs)
            loss_conf = sample_entropy.mean()
            loss_bal = -_entropy_from_probs(symbol_probs.mean(dim=0))

            relation_stats = _get_relation_stats(
                model=model,
                threshold=relation_sparsity_threshold,
            )
            loss_sparse = relation_stats["relation_l1"]
            loss_iter = (
                _kl_div_from_probs(out["probs_material"], out["prev_probs_material"])
                + _kl_div_from_probs(out["probs_hardness"], out["prev_probs_hardness"])
                + _kl_div_from_probs(out["probs_roughness"], out["prev_probs_roughness"])
            )

            loss = (
                loss_task
                + float(lambda_conf) * loss_conf
                + float(lambda_bal) * loss_bal
                + float(lambda_sparse) * loss_sparse
                + float(lambda_iter) * loss_iter
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(b["material"].shape[0])
        total_samples += bs
        total_symbol_samples += bs
        total_loss += float(loss.item()) * bs
        total_loss_task += float(loss_task.item()) * bs
        total_loss_m += float(loss_m.item()) * bs
        total_loss_h += float(loss_h.item()) * bs
        total_loss_r += float(loss_r.item()) * bs
        total_loss_conf += float(loss_conf.item()) * bs
        total_loss_bal += float(loss_bal.item()) * bs
        total_loss_sparse += float(loss_sparse.item()) * bs
        total_loss_iter += float(loss_iter.item()) * bs

        reason_batch_stats = _compute_reason_batch_stats(out)
        total_reason_contrib += float(reason_batch_stats["reason_contrib_ratio"].item()) * bs
        total_message_delta_norm += float(reason_batch_stats["message_delta_norm"].item()) * bs
        total_relation_abs_mean += float(relation_stats["relation_abs_mean"].item()) * bs
        total_relation_sparsity += float(relation_stats["relation_sparsity"].item()) * bs

        pm = out["logits_material"].argmax(dim=1).detach()
        ph = out["logits_hardness"].argmax(dim=1).detach()
        pr = out["logits_roughness"].argmax(dim=1).detach()
        tm = b["material"].detach()
        th = b["hardness"].detach()
        tr = b["roughness"].detach()

        _update_confusion(conf_m, pm, tm)
        _update_confusion(conf_h, ph, th)
        _update_confusion(conf_r, pr, tr)

        symbol_probs_detached = symbol_probs.detach()
        _update_symbol_hist(symbol_hist, out["symbol_id"].detach())
        symbol_prob_sum += symbol_probs_detached.sum(dim=0).to(torch.float64)
        sum_sample_entropy += sample_entropy.detach().sum().to(torch.float64)

        if log_freq > 0 and (step % log_freq == 0 or (total_batches is not None and step == total_batches)):
            elapsed = max(time.time() - start_time, 1e-6)
            it_per_sec = step / elapsed
            eta = -1.0
            if total_batches is not None:
                eta = (total_batches - step) / max(it_per_sec, 1e-6)
            if _is_main_process():
                active_symbol_count = int((symbol_hist > 0).sum().item())
                usage_ratio = active_symbol_count / float(max(symbol_k, 1))
                print(
                    f"[Train][Epoch {epoch:03d}] step {step}/{total_batches if total_batches is not None else '?'} "
                    f"loss={total_loss / max(total_samples, 1):.4f} "
                    f"task={total_loss_task / max(total_samples, 1):.4f} "
                    f"sparse={total_loss_sparse / max(total_samples, 1):.4f} "
                    f"iter={total_loss_iter / max(total_samples, 1):.4f} "
                    f"lr_head={optimizer.param_groups[0]['lr']:.2e} "
                    f"tau={symbol_temperature:.4f} "
                    f"use={usage_ratio:.3f} "
                    f"it/s={it_per_sec:.2f} "
                    f"eta={_format_eta(eta)}",
                    flush=True,
                )

    stats_tensor = torch.tensor(
        [
            total_loss,
            total_loss_task,
            total_loss_m,
            total_loss_h,
            total_loss_r,
            total_loss_conf,
            total_loss_bal,
            total_loss_sparse,
            total_loss_iter,
            total_reason_contrib,
            total_message_delta_norm,
            total_relation_abs_mean,
            total_relation_sparsity,
            float(total_samples),
            float(total_symbol_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    _reduce_if_distributed(stats_tensor)
    _reduce_if_distributed(conf_m)
    _reduce_if_distributed(conf_h)
    _reduce_if_distributed(conf_r)
    _reduce_if_distributed(symbol_hist)
    _reduce_if_distributed(symbol_prob_sum)
    _reduce_if_distributed(sum_sample_entropy)

    (
        total_loss_g,
        total_loss_task_g,
        total_loss_m_g,
        total_loss_h_g,
        total_loss_r_g,
        total_loss_conf_g,
        total_loss_bal_g,
        total_loss_sparse_g,
        total_loss_iter_g,
        total_reason_contrib_g,
        total_message_delta_norm_g,
        total_relation_abs_mean_g,
        total_relation_sparsity_g,
        total_samples_g,
        total_symbol_samples_g,
    ) = stats_tensor.tolist()
    denom = max(float(total_samples_g), 1.0)

    acc_m, f1_m = _confusion_to_acc_macro_f1(conf_m)
    acc_h, f1_h = _confusion_to_acc_macro_f1(conf_h)
    acc_r, f1_r = _confusion_to_acc_macro_f1(conf_r)
    mean_accuracy = (acc_m + acc_h + acc_r) / 3.0
    mean_macro_f1 = (f1_m + f1_h + f1_r) / 3.0
    symbol_health = _compute_symbol_health(
        symbol_hist=symbol_hist,
        symbol_prob_sum=symbol_prob_sum,
        sum_sample_entropy=float(sum_sample_entropy.item()),
        total_symbol_samples=float(total_symbol_samples_g),
    )
    return {
        "epoch": float(epoch),
        "symbol_temperature": float(symbol_temperature),
        "loss": total_loss_g / denom,
        "loss_task": total_loss_task_g / denom,
        "loss_material": total_loss_m_g / denom,
        "loss_hardness": total_loss_h_g / denom,
        "loss_roughness": total_loss_r_g / denom,
        "loss_symbol_conf": total_loss_conf_g / denom,
        "loss_symbol_bal": total_loss_bal_g / denom,
        "loss_reason_sparse": total_loss_sparse_g / denom,
        "loss_reason_iter": total_loss_iter_g / denom,
        "relation_abs_mean": total_relation_abs_mean_g / denom,
        "relation_sparsity": total_relation_sparsity_g / denom,
        "reason_contrib_ratio": total_reason_contrib_g / denom,
        "message_delta_norm": total_message_delta_norm_g / denom,
        "acc_material": acc_m,
        "acc_hardness": acc_h,
        "acc_roughness": acc_r,
        "mean_accuracy": mean_accuracy,
        "macro_f1_material": f1_m,
        "macro_f1_hardness": f1_h,
        "macro_f1_roughness": f1_r,
        "mean_macro_f1": mean_macro_f1,
        **symbol_health,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    num_material: int = 0,
    num_hardness: int = 0,
    num_roughness: int = 0,
    weight_material: torch.Tensor | None = None,
    weight_hardness: torch.Tensor | None = None,
    weight_roughness: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    loss_weight_material: float = 1.0,
    loss_weight_hardness: float = 1.0,
    loss_weight_roughness: float = 1.0,
    symbol_k: int = 64,
    symbol_temperature: float = 1.0,
    lambda_conf: float = 1e-2,
    lambda_bal: float = 2e-2,
    lambda_sparse: float = 1e-4,
    lambda_iter: float = 0.0,
    relation_sparsity_threshold: float = 1e-3,
    log_freq: int = 20,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_loss_task = 0.0
    total_loss_m = 0.0
    total_loss_h = 0.0
    total_loss_r = 0.0
    total_loss_conf = 0.0
    total_loss_bal = 0.0
    total_loss_sparse = 0.0
    total_loss_iter = 0.0
    total_reason_contrib = 0.0
    total_message_delta_norm = 0.0
    total_relation_abs_mean = 0.0
    total_relation_sparsity = 0.0
    total_samples = 0

    conf_m = _new_confusion(max(num_material, 1), device)
    conf_h = _new_confusion(max(num_hardness, 1), device)
    conf_r = _new_confusion(max(num_roughness, 1), device)

    symbol_hist = torch.zeros(max(symbol_k, 1), dtype=torch.float64, device=device)
    symbol_prob_sum = torch.zeros(max(symbol_k, 1), dtype=torch.float64, device=device)
    sum_sample_entropy = torch.zeros((), dtype=torch.float64, device=device)
    total_symbol_samples = 0

    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    start_time = time.time()

    for step, batch in enumerate(data_loader, start=1):
        b = _to_device(batch, device)
        out = model(b["touch"], symbol_temperature=symbol_temperature)

        loss_m = F.cross_entropy(
            out["logits_material"],
            b["material"],
            weight=weight_material,
            label_smoothing=label_smoothing,
        )
        loss_h = F.cross_entropy(
            out["logits_hardness"],
            b["hardness"],
            weight=weight_hardness,
            label_smoothing=label_smoothing,
        )
        loss_r = F.cross_entropy(
            out["logits_roughness"],
            b["roughness"],
            weight=weight_roughness,
            label_smoothing=label_smoothing,
        )
        loss_task = (
            float(loss_weight_material) * loss_m
            + float(loss_weight_hardness) * loss_h
            + float(loss_weight_roughness) * loss_r
        )

        symbol_probs = out["symbol_probs"]
        sample_entropy = _entropy_from_probs(symbol_probs)
        loss_conf = sample_entropy.mean()
        loss_bal = -_entropy_from_probs(symbol_probs.mean(dim=0))

        relation_stats = _get_relation_stats(
            model=model,
            threshold=relation_sparsity_threshold,
        )
        loss_sparse = relation_stats["relation_l1"]
        loss_iter = (
            _kl_div_from_probs(out["probs_material"], out["prev_probs_material"])
            + _kl_div_from_probs(out["probs_hardness"], out["prev_probs_hardness"])
            + _kl_div_from_probs(out["probs_roughness"], out["prev_probs_roughness"])
        )

        loss = (
            loss_task
            + float(lambda_conf) * loss_conf
            + float(lambda_bal) * loss_bal
            + float(lambda_sparse) * loss_sparse
            + float(lambda_iter) * loss_iter
        )

        bs = int(b["material"].shape[0])
        total_samples += bs
        total_symbol_samples += bs
        total_loss += float(loss.item()) * bs
        total_loss_task += float(loss_task.item()) * bs
        total_loss_m += float(loss_m.item()) * bs
        total_loss_h += float(loss_h.item()) * bs
        total_loss_r += float(loss_r.item()) * bs
        total_loss_conf += float(loss_conf.item()) * bs
        total_loss_bal += float(loss_bal.item()) * bs
        total_loss_sparse += float(loss_sparse.item()) * bs
        total_loss_iter += float(loss_iter.item()) * bs

        reason_batch_stats = _compute_reason_batch_stats(out)
        total_reason_contrib += float(reason_batch_stats["reason_contrib_ratio"].item()) * bs
        total_message_delta_norm += float(reason_batch_stats["message_delta_norm"].item()) * bs
        total_relation_abs_mean += float(relation_stats["relation_abs_mean"].item()) * bs
        total_relation_sparsity += float(relation_stats["relation_sparsity"].item()) * bs

        pm = out["logits_material"].argmax(dim=1).detach()
        ph = out["logits_hardness"].argmax(dim=1).detach()
        pr = out["logits_roughness"].argmax(dim=1).detach()
        tm = b["material"].detach()
        th = b["hardness"].detach()
        tr = b["roughness"].detach()

        _update_confusion(conf_m, pm, tm)
        _update_confusion(conf_h, ph, th)
        _update_confusion(conf_r, pr, tr)

        symbol_probs_detached = symbol_probs.detach()
        _update_symbol_hist(symbol_hist, out["symbol_id"].detach())
        symbol_prob_sum += symbol_probs_detached.sum(dim=0).to(torch.float64)
        sum_sample_entropy += sample_entropy.detach().sum().to(torch.float64)

        if log_freq > 0 and (step % log_freq == 0 or (total_batches is not None and step == total_batches)):
            elapsed = max(time.time() - start_time, 1e-6)
            it_per_sec = step / elapsed
            eta = -1.0
            if total_batches is not None:
                eta = (total_batches - step) / max(it_per_sec, 1e-6)
            if _is_main_process():
                print(
                    f"[Val][Epoch {epoch:03d}] step {step}/{total_batches if total_batches is not None else '?'} "
                    f"loss={total_loss / max(total_samples, 1):.4f} "
                    f"tau={symbol_temperature:.4f} "
                    f"it/s={it_per_sec:.2f} "
                    f"eta={_format_eta(eta)}",
                    flush=True,
                )

    stats_tensor = torch.tensor(
        [
            total_loss,
            total_loss_task,
            total_loss_m,
            total_loss_h,
            total_loss_r,
            total_loss_conf,
            total_loss_bal,
            total_loss_sparse,
            total_loss_iter,
            total_reason_contrib,
            total_message_delta_norm,
            total_relation_abs_mean,
            total_relation_sparsity,
            float(total_samples),
            float(total_symbol_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    _reduce_if_distributed(stats_tensor)
    _reduce_if_distributed(conf_m)
    _reduce_if_distributed(conf_h)
    _reduce_if_distributed(conf_r)
    _reduce_if_distributed(symbol_hist)
    _reduce_if_distributed(symbol_prob_sum)
    _reduce_if_distributed(sum_sample_entropy)

    (
        total_loss_g,
        total_loss_task_g,
        total_loss_m_g,
        total_loss_h_g,
        total_loss_r_g,
        total_loss_conf_g,
        total_loss_bal_g,
        total_loss_sparse_g,
        total_loss_iter_g,
        total_reason_contrib_g,
        total_message_delta_norm_g,
        total_relation_abs_mean_g,
        total_relation_sparsity_g,
        total_samples_g,
        total_symbol_samples_g,
    ) = stats_tensor.tolist()
    denom = max(float(total_samples_g), 1.0)

    acc_m, f1_m = _confusion_to_acc_macro_f1(conf_m)
    acc_h, f1_h = _confusion_to_acc_macro_f1(conf_h)
    acc_r, f1_r = _confusion_to_acc_macro_f1(conf_r)
    mean_accuracy = (acc_m + acc_h + acc_r) / 3.0
    mean_macro_f1 = (f1_m + f1_h + f1_r) / 3.0
    symbol_health = _compute_symbol_health(
        symbol_hist=symbol_hist,
        symbol_prob_sum=symbol_prob_sum,
        sum_sample_entropy=float(sum_sample_entropy.item()),
        total_symbol_samples=float(total_symbol_samples_g),
    )
    return {
        "epoch": float(epoch),
        "symbol_temperature": float(symbol_temperature),
        "loss": total_loss_g / denom,
        "loss_task": total_loss_task_g / denom,
        "loss_material": total_loss_m_g / denom,
        "loss_hardness": total_loss_h_g / denom,
        "loss_roughness": total_loss_r_g / denom,
        "loss_symbol_conf": total_loss_conf_g / denom,
        "loss_symbol_bal": total_loss_bal_g / denom,
        "loss_reason_sparse": total_loss_sparse_g / denom,
        "loss_reason_iter": total_loss_iter_g / denom,
        "relation_abs_mean": total_relation_abs_mean_g / denom,
        "relation_sparsity": total_relation_sparsity_g / denom,
        "reason_contrib_ratio": total_reason_contrib_g / denom,
        "message_delta_norm": total_message_delta_norm_g / denom,
        "acc_material": acc_m,
        "acc_hardness": acc_h,
        "acc_roughness": acc_r,
        "mean_accuracy": mean_accuracy,
        "macro_f1_material": f1_m,
        "macro_f1_hardness": f1_h,
        "macro_f1_roughness": f1_r,
        "mean_macro_f1": mean_macro_f1,
        **symbol_health,
    }


def export_reason_trace(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    label_maps: Dict[str, Dict],
    output_path: str | Path,
    symbol_temperature: float = 0.3,
    max_samples: int = 0,
) -> int:
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx_to_material = label_maps.get("idx_to_material", {})
    idx_to_hardness = label_maps.get("idx_to_hardness", {})
    idx_to_roughness = label_maps.get("idx_to_roughness", {})

    def _lookup(mapping: Dict, idx: int) -> str:
        if idx in mapping:
            return str(mapping[idx])
        return str(mapping.get(str(idx), idx))

    def _pick_source(name_a: str, vec_a: torch.Tensor, name_b: str, vec_b: torch.Tensor) -> str:
        norm_a = float(vec_a.norm().item())
        norm_b = float(vec_b.norm().item())
        return name_a if norm_a >= norm_b else name_b

    def _extract_meta(batch: Dict[str, object], key: str, idx: int, default):
        value = batch.get(key, default)
        if isinstance(value, torch.Tensor):
            item = value[idx]
            return item.item() if item.dim() == 0 else item.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return value[idx]
        return value

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch in data_loader:
            touch = batch["touch"].to(device, non_blocking=True)
            touch = _prepare_stage3_video_input(touch)
            out = model(touch, symbol_temperature=symbol_temperature)
            bs = int(touch.shape[0])

            for i in range(bs):
                if max_samples > 0 and written >= max_samples:
                    return written

                material_target = int(_extract_meta(batch, "material", i, -1))
                hardness_target = int(_extract_meta(batch, "hardness", i, -1))
                roughness_target = int(_extract_meta(batch, "roughness", i, -1))

                unary_material = out["logits_material_unary"][i]
                unary_hardness = out["logits_hardness_unary"][i]
                unary_roughness = out["logits_roughness_unary"][i]
                final_material = out["logits_material"][i]
                final_hardness = out["logits_hardness"][i]
                final_roughness = out["logits_roughness"][i]

                msg_mh = out["reason_message_material_from_hardness"][i]
                msg_mr = out["reason_message_material_from_roughness"][i]
                msg_hm = out["reason_message_hardness_from_material"][i]
                msg_hr = out["reason_message_hardness_from_roughness"][i]
                msg_rm = out["reason_message_roughness_from_material"][i]
                msg_rh = out["reason_message_roughness_from_hardness"][i]

                row = {
                    "sample_id": str(_extract_meta(batch, "sample_id", i, f"sample_{written}")),
                    "sequence_id": str(_extract_meta(batch, "sequence_id", i, "")),
                    "center_frame": int(_extract_meta(batch, "center_frame", i, -1)),
                    "symbol_id": int(out["symbol_id"][i].detach().cpu().item()),
                    "material": {
                        "target_idx": material_target,
                        "target": _lookup(idx_to_material, material_target),
                        "unary_pred_idx": int(unary_material.argmax(dim=-1).item()),
                        "unary_pred": _lookup(idx_to_material, int(unary_material.argmax(dim=-1).item())),
                        "final_pred_idx": int(final_material.argmax(dim=-1).item()),
                        "final_pred": _lookup(idx_to_material, int(final_material.argmax(dim=-1).item())),
                        "top_relation_message_source": _pick_source("hardness", msg_mh, "roughness", msg_mr),
                        "message_delta_norm": float((final_material - unary_material).norm().item()),
                    },
                    "hardness": {
                        "target_idx": hardness_target,
                        "target": _lookup(idx_to_hardness, hardness_target),
                        "unary_pred_idx": int(unary_hardness.argmax(dim=-1).item()),
                        "unary_pred": _lookup(idx_to_hardness, int(unary_hardness.argmax(dim=-1).item())),
                        "final_pred_idx": int(final_hardness.argmax(dim=-1).item()),
                        "final_pred": _lookup(idx_to_hardness, int(final_hardness.argmax(dim=-1).item())),
                        "top_relation_message_source": _pick_source("material", msg_hm, "roughness", msg_hr),
                        "message_delta_norm": float((final_hardness - unary_hardness).norm().item()),
                    },
                    "roughness": {
                        "target_idx": roughness_target,
                        "target": _lookup(idx_to_roughness, roughness_target),
                        "unary_pred_idx": int(unary_roughness.argmax(dim=-1).item()),
                        "unary_pred": _lookup(idx_to_roughness, int(unary_roughness.argmax(dim=-1).item())),
                        "final_pred_idx": int(final_roughness.argmax(dim=-1).item()),
                        "final_pred": _lookup(idx_to_roughness, int(final_roughness.argmax(dim=-1).item())),
                        "top_relation_message_source": _pick_source("material", msg_rm, "hardness", msg_rh),
                        "message_delta_norm": float((final_roughness - unary_roughness).norm().item()),
                    },
                    "reason_contrib_ratio": float(
                        _compute_reason_batch_stats(
                            {
                                "logits_material": final_material.unsqueeze(0),
                                "logits_hardness": final_hardness.unsqueeze(0),
                                "logits_roughness": final_roughness.unsqueeze(0),
                                "logits_material_unary": unary_material.unsqueeze(0),
                                "logits_hardness_unary": unary_hardness.unsqueeze(0),
                                "logits_roughness_unary": unary_roughness.unsqueeze(0),
                            }
                        )["reason_contrib_ratio"].item()
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    return written
