#!/usr/bin/env python3
"""
Training/evaluation engine for hierarchical Stage3.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from Stage3_method_w.stage3_task_spec import (
    CLASSIFICATION_TASKS,
    LAYER_TASKS,
    REGRESSION_TASKS,
    SYMBOL_LEVELS,
    TASK_ORDER,
    TASK_TO_LAYER,
)


def _prepare_stage3_video_input(x: torch.Tensor) -> torch.Tensor:
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
def _update_confusion(
    conf: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    n_cls = int(conf.shape[0])
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()
    valid = (targets >= 0) & (targets < n_cls) & (preds >= 0) & (preds < n_cls)
    if mask is not None:
        valid = valid & (mask.view(-1) > 0)
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


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    valid = mask.view(-1) > 0
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return logits.sum() * 0.0, 0
    return (
        F.cross_entropy(
            logits[valid],
            targets[valid],
            weight=weight,
            label_smoothing=label_smoothing,
        ),
        valid_count,
    )


def _masked_smooth_l1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, int]:
    valid = mask.view(-1) > 0
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return preds.sum() * 0.0, 0
    pred_values = preds.view(-1)[valid]
    target_values = targets.view(-1).to(pred_values.dtype)[valid]
    return F.smooth_l1_loss(pred_values, target_values, beta=float(beta)), valid_count


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    touch = batch["touch"].to(device, non_blocking=True)
    targets = {}
    for task in TASK_ORDER:
        target = batch["targets"][task].to(device, non_blocking=True)
        targets[task] = target.float() if task in REGRESSION_TASKS else target.long()
    target_mask = {
        task: batch["target_mask"][task].to(device, non_blocking=True).float()
        for task in TASK_ORDER
    }
    return {
        "touch": _prepare_stage3_video_input(touch),
        "targets": targets,
        "target_mask": target_mask,
    }


def _get_relation_stats(model: torch.nn.Module, threshold: float = 1e-3) -> Dict[str, torch.Tensor]:
    core = _unwrap_model(model)
    return {
        "relation_l1": core.relation_l1(),
        "relation_abs_mean": core.relation_abs_mean(),
        "relation_sparsity": core.relation_sparsity(threshold=threshold),
    }


def _zero_scalar_from_model(model: torch.nn.Module) -> torch.Tensor:
    param = next(_unwrap_model(model).parameters())
    return param.sum() * 0.0


def _aggregate_layer_losses(
    model: torch.nn.Module,
    task_losses: Dict[str, torch.Tensor],
    task_valid_counts: Dict[str, int],
    enabled_layers: Iterable[str],
    layer_loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
    zero = _zero_scalar_from_model(model)
    active_layers: List[str] = []
    layer_losses: Dict[str, torch.Tensor] = {}
    weighted_sum = zero
    total_weight = 0.0

    enabled_layers = list(enabled_layers)
    for layer_name in enabled_layers:
        layer_tasks = LAYER_TASKS[layer_name]
        valid_task_losses = [
            task_losses[task]
            for task in layer_tasks
            if int(task_valid_counts.get(task, 0)) > 0
        ]
        if len(valid_task_losses) == 0:
            continue
        active_layers.append(layer_name)
        layer_loss = torch.stack(valid_task_losses, dim=0).mean()
        layer_losses[layer_name] = layer_loss
        layer_weight = float(layer_loss_weights.get(layer_name, 1.0))
        weighted_sum = weighted_sum + layer_weight * layer_loss
        total_weight += layer_weight

    if total_weight <= 0.0:
        return zero, layer_losses, active_layers
    return weighted_sum / float(total_weight), layer_losses, active_layers


def _compute_iter_loss(
    out: Dict[str, object],
    task_valid_counts: Dict[str, int],
    enabled_layers: Iterable[str],
    model: torch.nn.Module,
) -> torch.Tensor:
    zero = _zero_scalar_from_model(model)
    terms = []
    enabled_layers = set(enabled_layers)
    for task in TASK_ORDER:
        if TASK_TO_LAYER[task] not in enabled_layers:
            continue
        if int(task_valid_counts.get(task, 0)) <= 0:
            continue
        if task not in out["prev_probs"]:
            continue
        terms.append(_kl_div_from_probs(out["probs"][task], out["prev_probs"][task]))
    if len(terms) == 0:
        return zero
    return torch.stack(terms, dim=0).mean()


def _init_symbol_buffers(out: Dict[str, object], device: torch.device):
    symbol_hist = {}
    symbol_prob_sum = {}
    for symbol_name in SYMBOL_LEVELS:
        probs = out["symbols"][symbol_name]["symbol_probs"]
        symbol_hist[symbol_name] = torch.zeros((int(probs.shape[-1]),), dtype=torch.float64, device=device)
        symbol_prob_sum[symbol_name] = torch.zeros((int(probs.shape[-1]),), dtype=torch.float64, device=device)
    return symbol_hist, symbol_prob_sum


def _run_epoch(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    task_class_weights: Dict[str, torch.Tensor | None],
    label_smoothing: float,
    layer_loss_weights: Dict[str, float],
    enabled_layers: Iterable[str],
    symbol_temperature: float,
    lambda_conf: float,
    lambda_bal: float,
    lambda_sparse: float,
    lambda_iter: float,
    relation_sparsity_threshold: float,
    log_freq: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    enabled_layers = list(enabled_layers)
    enabled_tasks = [
        task
        for layer_name in enabled_layers
        for task in LAYER_TASKS[layer_name]
    ]
    active_symbol_names = ["obs"] + enabled_layers

    total_loss = 0.0
    total_loss_task = 0.0
    total_loss_conf = 0.0
    total_loss_bal = 0.0
    total_loss_sparse = 0.0
    total_loss_iter = 0.0
    total_relation_abs_mean = 0.0
    total_relation_sparsity = 0.0
    total_skip_gate = 0.0
    total_samples = 0

    per_task_loss_sum = {task: 0.0 for task in TASK_ORDER}
    per_task_count = {task: 0 for task in TASK_ORDER}
    regression_abs_error_sum = {task: 0.0 for task in REGRESSION_TASKS}
    regression_sq_error_sum = {task: 0.0 for task in REGRESSION_TASKS}
    regression_count = {task: 0 for task in REGRESSION_TASKS}
    confusion = {task: None for task in TASK_ORDER}

    symbol_hist = None
    symbol_prob_sum = None
    symbol_entropy_sum = {name: 0.0 for name in SYMBOL_LEVELS}
    symbol_sample_count = {name: 0.0 for name in SYMBOL_LEVELS}

    start = time.time()
    num_steps = max(len(data_loader), 1)

    for step, batch in enumerate(data_loader, start=1):
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        b = _to_device(batch, device=device)

        autocast_enabled = bool(use_amp and device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            out = model(b["touch"], symbol_temperature=symbol_temperature)

            if symbol_hist is None:
                symbol_hist, symbol_prob_sum = _init_symbol_buffers(out=out, device=device)
                for task in CLASSIFICATION_TASKS:
                    num_classes = int(out["logits"][task].shape[-1])
                    confusion[task] = _new_confusion(num_classes=num_classes, device=device)

            task_losses: Dict[str, torch.Tensor] = {}
            task_valid_counts: Dict[str, int] = {}
            for task in TASK_ORDER:
                if task not in out["logits"]:
                    continue
                if task not in enabled_tasks:
                    task_losses[task] = _zero_scalar_from_model(model)
                    task_valid_counts[task] = 0
                    continue
                if task in REGRESSION_TASKS:
                    loss_task, valid_count = _masked_smooth_l1(
                        preds=out["logits"][task],
                        targets=b["targets"][task],
                        mask=b["target_mask"][task],
                    )
                else:
                    loss_task, valid_count = _masked_cross_entropy(
                        logits=out["logits"][task],
                        targets=b["targets"][task],
                        mask=b["target_mask"][task],
                        weight=task_class_weights.get(task),
                        label_smoothing=label_smoothing,
                    )
                task_losses[task] = loss_task
                task_valid_counts[task] = valid_count

            loss_task, layer_losses, active_layers = _aggregate_layer_losses(
                model=model,
                task_losses=task_losses,
                task_valid_counts=task_valid_counts,
                enabled_layers=enabled_layers,
                layer_loss_weights=layer_loss_weights,
            )

            symbol_conf_terms = []
            symbol_bal_terms = []
            for symbol_name in active_symbol_names:
                probs = out["symbols"][symbol_name]["symbol_probs"]
                symbol_conf_terms.append(_entropy_from_probs(probs).mean())
                symbol_bal_terms.append(-_entropy_from_probs(probs.mean(dim=0)))
            loss_conf = torch.stack(symbol_conf_terms, dim=0).mean()
            loss_bal = torch.stack(symbol_bal_terms, dim=0).mean()

            relation_stats = _get_relation_stats(
                model=model,
                threshold=relation_sparsity_threshold,
            )
            loss_sparse = relation_stats["relation_l1"]
            loss_iter = _compute_iter_loss(
                out=out,
                task_valid_counts=task_valid_counts,
                enabled_layers=enabled_layers,
                model=model,
            )

            loss = (
                loss_task
                + float(lambda_conf) * loss_conf
                + float(lambda_bal) * loss_bal
                + float(lambda_sparse) * loss_sparse
                + float(lambda_iter) * loss_iter
            )

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = int(b["touch"].shape[0])
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_loss_task += float(loss_task.item()) * bs
        total_loss_conf += float(loss_conf.item()) * bs
        total_loss_bal += float(loss_bal.item()) * bs
        total_loss_sparse += float(loss_sparse.item()) * bs
        total_loss_iter += float(loss_iter.item()) * bs
        total_relation_abs_mean += float(relation_stats["relation_abs_mean"].item()) * bs
        total_relation_sparsity += float(relation_stats["relation_sparsity"].item()) * bs
        total_skip_gate += float(out["aux"]["contact_skip_gate_mean"].mean().item()) * bs

        for task in TASK_ORDER:
            if task in task_losses and int(task_valid_counts.get(task, 0)) > 0:
                valid_count = int(task_valid_counts[task])
                per_task_loss_sum[task] += float(task_losses[task].item()) * valid_count
                per_task_count[task] += valid_count
                if task in REGRESSION_TASKS:
                    valid = b["target_mask"][task].view(-1) > 0
                    preds = out["logits"][task].view(-1).detach()[valid]
                    targets = b["targets"][task].view(-1).detach()[valid].to(preds.dtype)
                    errors = preds - targets
                    regression_abs_error_sum[task] += float(errors.abs().sum().item())
                    regression_sq_error_sum[task] += float((errors * errors).sum().item())
                    regression_count[task] += int(valid.sum().item())
                else:
                    preds = out["logits"][task].argmax(dim=1).detach()
                    _update_confusion(
                        confusion[task],
                        preds=preds,
                        targets=b["targets"][task].detach(),
                        mask=b["target_mask"][task].detach(),
                    )

        for symbol_name in SYMBOL_LEVELS:
            probs = out["symbols"][symbol_name]["symbol_probs"].detach()
            ids = out["symbols"][symbol_name]["symbol_id"].detach()
            sample_entropy = _entropy_from_probs(probs).sum()
            _update_symbol_hist(symbol_hist[symbol_name], ids)
            symbol_prob_sum[symbol_name] += probs.sum(dim=0).to(torch.float64)
            symbol_entropy_sum[symbol_name] += float(sample_entropy.item())
            symbol_sample_count[symbol_name] += float(probs.shape[0])

        if log_freq > 0 and _is_main_process() and (step % log_freq == 0 or step == num_steps):
            elapsed = max(time.time() - start, 1e-6)
            it_s = float(step / elapsed)
            eta = float((num_steps - step) / max(it_s, 1e-6))
            phase = "Train" if is_train else "Val"
            print(
                f"[{phase}][Epoch {epoch:03d}] step {step}/{num_steps} "
                f"loss={loss.item():.4f} task={loss_task.item():.4f} "
                f"sparse={loss_sparse.item():.4f} iter={loss_iter.item():.4f} "
                f"tau={symbol_temperature:.4f} it/s={it_s:.2f} eta={_format_eta(eta)}"
            )

    totals = torch.tensor(
        [
            total_loss,
            total_loss_task,
            total_loss_conf,
            total_loss_bal,
            total_loss_sparse,
            total_loss_iter,
            total_relation_abs_mean,
            total_relation_sparsity,
            total_skip_gate,
            float(total_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    totals = _reduce_if_distributed(totals)
    (
        total_loss_g,
        total_loss_task_g,
        total_loss_conf_g,
        total_loss_bal_g,
        total_loss_sparse_g,
        total_loss_iter_g,
        total_relation_abs_mean_g,
        total_relation_sparsity_g,
        total_skip_gate_g,
        total_samples_g,
    ) = totals.tolist()
    denom = max(float(total_samples_g), 1.0)

    stats: Dict[str, float] = {
        "loss": total_loss_g / denom,
        "loss_task": total_loss_task_g / denom,
        "loss_symbol_conf": total_loss_conf_g / denom,
        "loss_symbol_bal": total_loss_bal_g / denom,
        "loss_sparse": total_loss_sparse_g / denom,
        "loss_iter": total_loss_iter_g / denom,
        "relation_abs_mean": total_relation_abs_mean_g / denom,
        "relation_sparsity": total_relation_sparsity_g / denom,
        "contact_skip_gate_mean": total_skip_gate_g / denom,
        "enabled_layer_count": float(len(enabled_layers)),
    }

    layer_acc_values = []
    layer_f1_values = []
    per_task_loss_tensor = torch.tensor(
        [per_task_loss_sum[task] for task in TASK_ORDER],
        dtype=torch.float64,
        device=device,
    )
    per_task_count_tensor = torch.tensor(
        [per_task_count[task] for task in TASK_ORDER],
        dtype=torch.float64,
        device=device,
    )
    per_task_loss_tensor = _reduce_if_distributed(per_task_loss_tensor)
    per_task_count_tensor = _reduce_if_distributed(per_task_count_tensor)
    for idx, task in enumerate(TASK_ORDER):
        per_task_loss_sum[task] = float(per_task_loss_tensor[idx].item())
        per_task_count[task] = int(round(float(per_task_count_tensor[idx].item())))

    regression_tasks = sorted(REGRESSION_TASKS)
    regression_abs_tensor = torch.tensor(
        [regression_abs_error_sum[task] for task in regression_tasks],
        dtype=torch.float64,
        device=device,
    )
    regression_sq_tensor = torch.tensor(
        [regression_sq_error_sum[task] for task in regression_tasks],
        dtype=torch.float64,
        device=device,
    )
    regression_count_tensor = torch.tensor(
        [regression_count[task] for task in regression_tasks],
        dtype=torch.float64,
        device=device,
    )
    regression_abs_tensor = _reduce_if_distributed(regression_abs_tensor)
    regression_sq_tensor = _reduce_if_distributed(regression_sq_tensor)
    regression_count_tensor = _reduce_if_distributed(regression_count_tensor)
    for idx, task in enumerate(regression_tasks):
        regression_abs_error_sum[task] = float(regression_abs_tensor[idx].item())
        regression_sq_error_sum[task] = float(regression_sq_tensor[idx].item())
        regression_count[task] = int(round(float(regression_count_tensor[idx].item())))

    for task in TASK_ORDER:
        if task in REGRESSION_TASKS:
            count = max(float(regression_count.get(task, 0)), 1.0)
            stats[f"loss_{task}"] = per_task_loss_sum[task] / max(float(per_task_count[task]), 1.0)
            stats[f"mae_{task}"] = regression_abs_error_sum[task] / count
            stats[f"rmse_{task}"] = math.sqrt(regression_sq_error_sum[task] / count)
            stats[f"acc_{task}"] = 0.0
            stats[f"macro_f1_{task}"] = 0.0
            continue
        if confusion[task] is None:
            stats[f"loss_{task}"] = 0.0
            stats[f"acc_{task}"] = 0.0
            stats[f"macro_f1_{task}"] = 0.0
            continue
        confusion[task] = _reduce_if_distributed(confusion[task])
        acc, macro_f1 = _confusion_to_acc_macro_f1(confusion[task])
        stats[f"loss_{task}"] = per_task_loss_sum[task] / max(float(per_task_count[task]), 1.0)
        stats[f"acc_{task}"] = acc
        stats[f"macro_f1_{task}"] = macro_f1

    for layer_name, layer_tasks in LAYER_TASKS.items():
        task_metrics_acc = [
            stats[f"acc_{task}"]
            for task in layer_tasks
            if task in CLASSIFICATION_TASKS and int(per_task_count.get(task, 0)) > 0
        ]
        task_metrics_f1 = [
            stats[f"macro_f1_{task}"]
            for task in layer_tasks
            if task in CLASSIFICATION_TASKS and int(per_task_count.get(task, 0)) > 0
        ]
        task_metrics_mae = [
            stats[f"mae_{task}"]
            for task in layer_tasks
            if task in REGRESSION_TASKS and int(per_task_count.get(task, 0)) > 0
        ]
        if len(task_metrics_acc) == 0:
            stats[f"layer_acc_{layer_name}"] = 0.0
            stats[f"layer_macro_f1_{layer_name}"] = 0.0
        else:
            layer_acc = float(sum(task_metrics_acc) / len(task_metrics_acc))
            layer_f1 = float(sum(task_metrics_f1) / len(task_metrics_f1))
            stats[f"layer_acc_{layer_name}"] = layer_acc
            stats[f"layer_macro_f1_{layer_name}"] = layer_f1
            layer_acc_values.append(layer_acc)
            layer_f1_values.append(layer_f1)
        stats[f"layer_mae_{layer_name}"] = (
            float(sum(task_metrics_mae) / len(task_metrics_mae))
            if len(task_metrics_mae) > 0
            else 0.0
        )

    stats["mean_active_layer_accuracy"] = (
        float(sum(layer_acc_values) / len(layer_acc_values))
        if len(layer_acc_values) > 0
        else 0.0
    )
    stats["mean_active_layer_macro_f1"] = (
        float(sum(layer_f1_values) / len(layer_f1_values))
        if len(layer_f1_values) > 0
        else 0.0
    )
    all_layer_acc_values = [stats[f"layer_acc_{layer_name}"] for layer_name in LAYER_TASKS]
    all_layer_f1_values = [stats[f"layer_macro_f1_{layer_name}"] for layer_name in LAYER_TASKS]
    stats["mean_layer_accuracy"] = float(sum(all_layer_acc_values) / len(all_layer_acc_values))
    stats["mean_layer_macro_f1"] = float(sum(all_layer_f1_values) / len(all_layer_f1_values))

    for symbol_name in SYMBOL_LEVELS:
        symbol_hist[symbol_name] = _reduce_if_distributed(symbol_hist[symbol_name])
        symbol_prob_sum[symbol_name] = _reduce_if_distributed(symbol_prob_sum[symbol_name])
        symbol_scalar = torch.tensor(
            [symbol_entropy_sum[symbol_name], symbol_sample_count[symbol_name]],
            dtype=torch.float64,
            device=device,
        )
        symbol_scalar = _reduce_if_distributed(symbol_scalar)
        symbol_entropy_sum[symbol_name] = float(symbol_scalar[0].item())
        symbol_sample_count[symbol_name] = float(symbol_scalar[1].item())
        health = _compute_symbol_health(
            symbol_hist=symbol_hist[symbol_name],
            symbol_prob_sum=symbol_prob_sum[symbol_name],
            sum_sample_entropy=symbol_entropy_sum[symbol_name],
            total_symbol_samples=symbol_sample_count[symbol_name],
        )
        for key, value in health.items():
            stats[f"{symbol_name}_{key}"] = float(value)

    stats["mean_symbol_usage_ratio"] = float(
        sum(stats[f"{name}_symbol_usage_ratio"] for name in SYMBOL_LEVELS) / len(SYMBOL_LEVELS)
    )
    stats["mean_symbol_collapse_ratio"] = float(
        sum(stats[f"{name}_top1_symbol_collapse_ratio"] for name in SYMBOL_LEVELS) / len(SYMBOL_LEVELS)
    )
    return stats


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    task_class_weights: Dict[str, torch.Tensor | None] | None = None,
    label_smoothing: float = 0.0,
    layer_loss_weights: Dict[str, float] | None = None,
    enabled_layers: Iterable[str] | None = None,
    symbol_temperature: float = 1.0,
    lambda_conf: float = 1e-2,
    lambda_bal: float = 2e-2,
    lambda_sparse: float = 1e-4,
    lambda_iter: float = 0.0,
    relation_sparsity_threshold: float = 1e-3,
    log_freq: int = 50,
) -> Dict[str, float]:
    return _run_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        epoch=epoch,
        optimizer=optimizer,
        use_amp=use_amp,
        task_class_weights=task_class_weights or {},
        label_smoothing=label_smoothing,
        layer_loss_weights=layer_loss_weights or {layer: 1.0 for layer in LAYER_TASKS},
        enabled_layers=enabled_layers or list(LAYER_TASKS.keys()),
        symbol_temperature=symbol_temperature,
        lambda_conf=lambda_conf,
        lambda_bal=lambda_bal,
        lambda_sparse=lambda_sparse,
        lambda_iter=lambda_iter,
        relation_sparsity_threshold=relation_sparsity_threshold,
        log_freq=log_freq,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    task_class_weights: Dict[str, torch.Tensor | None] | None = None,
    label_smoothing: float = 0.0,
    layer_loss_weights: Dict[str, float] | None = None,
    enabled_layers: Iterable[str] | None = None,
    symbol_temperature: float = 1.0,
    lambda_conf: float = 1e-2,
    lambda_bal: float = 2e-2,
    lambda_sparse: float = 1e-4,
    lambda_iter: float = 0.0,
    relation_sparsity_threshold: float = 1e-3,
    log_freq: int = 20,
) -> Dict[str, float]:
    return _run_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        epoch=epoch,
        optimizer=None,
        use_amp=False,
        task_class_weights=task_class_weights or {},
        label_smoothing=label_smoothing,
        layer_loss_weights=layer_loss_weights or {layer: 1.0 for layer in LAYER_TASKS},
        enabled_layers=enabled_layers or list(LAYER_TASKS.keys()),
        symbol_temperature=symbol_temperature,
        lambda_conf=lambda_conf,
        lambda_bal=lambda_bal,
        lambda_sparse=lambda_sparse,
        lambda_iter=lambda_iter,
        relation_sparsity_threshold=relation_sparsity_threshold,
        log_freq=log_freq,
    )


__all__ = [
    "_prepare_stage3_video_input",
    "train_one_epoch",
    "evaluate",
]
