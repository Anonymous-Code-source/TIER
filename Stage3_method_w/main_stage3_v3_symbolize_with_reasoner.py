#!/usr/bin/env python3
"""
Stage3_method_w V3 training closure (Symbolize + Reasoner) for Causal_data.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from Stage3_method_w.stage3_dataset_causal_v1 import build_causal_dataloader

from Stage3_method_w.config_stage3_v3_symbolize_with_reasoner import parse_args
from Stage3_method_w.engine_stage3_v3_symbolize_with_reasoner import (
    evaluate,
    export_reason_trace,
    train_one_epoch,
)
from Stage3_method_w.model_stage3_symbolize_with_reasoner import (
    Stage3SymbolizeWithReasonerModel,
    load_stage2_backbone,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def make_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_serializable(payload), ensure_ascii=False) + "\n")


def append_csv_row(path: Path, row: Dict[str, float]) -> None:
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _compute_class_weight_from_samples(
    samples,
    key: str,
    num_classes: int,
    power: float,
    device: torch.device,
):
    counts = np.zeros((num_classes,), dtype=np.float64)
    for rec in samples:
        idx = int(rec[key])
        if 0 <= idx < num_classes:
            counts[idx] += 1.0

    weights = np.zeros_like(counts)
    nonzero = counts > 0
    if np.any(nonzero):
        weights[nonzero] = 1.0 / np.power(counts[nonzero], float(power))
        weights[nonzero] *= float(np.count_nonzero(nonzero)) / float(weights[nonzero].sum())
    else:
        weights[:] = 1.0

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return counts.tolist(), weights.tolist(), weight_tensor


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _init_distributed(args) -> Dict[str, object]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return {
            "distributed": False,
            "device": device,
            "rank": 0,
            "world_size": 1,
            "local_rank": -1,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requested but CUDA is unavailable.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    return {
        "distributed": True,
        "device": device,
        "rank": _get_rank(),
        "world_size": _get_world_size(),
        "local_rank": local_rank,
    }


def _compute_symbol_temperature(epoch: int, args) -> float:
    if args.tau_anneal_epochs <= 1:
        return float(args.tau_end)
    progress = min(max(float(epoch) / float(args.tau_anneal_epochs - 1), 0.0), 1.0)
    tau = float(args.tau_start + progress * (args.tau_end - args.tau_start))
    low = min(float(args.tau_start), float(args.tau_end))
    high = max(float(args.tau_start), float(args.tau_end))
    return float(min(max(tau, low), high))


def _load_init_checkpoint(model: torch.nn.Module, init_checkpoint: str) -> Dict[str, int]:
    if init_checkpoint == "":
        return {"missing_keys": 0, "unexpected_keys": 0, "mismatched_keys": 0, "loaded_keys": 0}

    checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported init checkpoint format: {init_checkpoint}")

    model_state = model.state_dict()
    compatible_state = {}
    mismatched_keys = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if tuple(value.shape) != tuple(target.shape):
            mismatched_keys.append(key)
            continue
        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    return {
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "mismatched_keys": len(mismatched_keys),
        "loaded_keys": len(compatible_state),
    }


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _save_relation_matrix(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    metric_name: str,
    metric_value: float,
    sparsity_threshold: float,
) -> None:
    core = _unwrap_model(model)
    payload = core.export_relation_state()
    payload.update(
        {
            "epoch": int(epoch),
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
            "relation_abs_mean": float(core.relation_abs_mean().detach().cpu().item()),
            "relation_sparsity": float(
                core.relation_sparsity(threshold=sparsity_threshold).detach().cpu().item()
            ),
            "relation_sparsity_threshold": float(sparsity_threshold),
        }
    )
    torch.save(payload, path)


def _is_staged_training(args) -> bool:
    return int(args.phase_a_epochs) + int(args.phase_b_epochs) + int(args.phase_c_epochs) > 0


def _phase_name_for_epoch(epoch: int, args) -> str:
    if not _is_staged_training(args):
        return "joint"
    if epoch < int(args.phase_a_epochs):
        return "phase_a"
    if epoch < int(args.phase_a_epochs) + int(args.phase_b_epochs):
        return "phase_b"
    return "phase_c"


def _phase_total_epochs(phase_name: str, args) -> int:
    if phase_name == "phase_a":
        return max(int(args.phase_a_epochs), 1)
    if phase_name == "phase_b":
        return max(int(args.phase_b_epochs), 1)
    if phase_name == "phase_c":
        return max(int(args.phase_c_epochs), 1)
    return max(int(args.epochs), 1)


def _phase_trainability(phase_name: str, args) -> Dict[str, bool]:
    if phase_name == "phase_a":
        return {
            "backbone_trainable": False,
            "reasoner_trainable": False,
        }
    if phase_name == "phase_b":
        return {
            "backbone_trainable": False,
            "reasoner_trainable": not bool(args.freeze_reasoner),
        }
    if phase_name == "phase_c":
        return {
            "backbone_trainable": True,
            "reasoner_trainable": not bool(args.freeze_reasoner),
        }
    return {
        "backbone_trainable": not bool(args.freeze_backbone),
        "reasoner_trainable": not bool(args.freeze_reasoner),
    }


def _apply_training_phase(model: torch.nn.Module, phase_name: str, args) -> Dict[str, bool]:
    core = _unwrap_model(model)
    phase_state = _phase_trainability(phase_name=phase_name, args=args)
    core.set_backbone_trainable(phase_state["backbone_trainable"])
    core.set_reasoner_trainable(phase_state["reasoner_trainable"])
    return phase_state


def _build_optimizer_and_scheduler(
    model: torch.nn.Module,
    args,
    phase_name: str,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, List[str]]:
    core = _unwrap_model(model)
    backbone_param_ids = {id(p) for p in core.backbone.parameters()}
    reasoner_param_ids = {id(p) for p in core.reasoner.parameters()}

    base_params = []
    reasoner_params = []
    backbone_params = []
    for p in core.parameters():
        if not p.requires_grad:
            continue
        if id(p) in reasoner_param_ids:
            reasoner_params.append(p)
        elif id(p) in backbone_param_ids:
            backbone_params.append(p)
        else:
            base_params.append(p)

    param_groups = []
    active_groups: List[str] = []
    if len(base_params) > 0:
        param_groups.append({"params": base_params, "lr": float(args.lr_head), "name": "base"})
        active_groups.append("base")
    if len(reasoner_params) > 0:
        param_groups.append({"params": reasoner_params, "lr": float(args.lr_reasoner), "name": "reasoner"})
        active_groups.append("reasoner")
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": float(args.lr_backbone), "name": "backbone"})
        active_groups.append("backbone")
    if len(param_groups) == 0:
        raise RuntimeError(f"No trainable parameters found for phase={phase_name}.")

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_phase_total_epochs(phase_name=phase_name, args=args),
            eta_min=float(args.min_lr),
        )
    return optimizer, scheduler, active_groups


def _get_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return float("nan")


def _ordered_label_names(mapping: Dict, size: int) -> List[str]:
    pairs = []
    for key, value in mapping.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        pairs.append((idx, str(value)))
    pairs.sort(key=lambda item: item[0])
    labels = [name for _, name in pairs]
    if len(labels) >= size:
        return labels[:size]
    return labels + [str(idx) for idx in range(len(labels), size)]


def _export_relation_heatmap(
    relation_matrix_path: Path,
    output_path: Path,
    label_maps: Dict[str, Dict],
) -> Dict[str, object]:
    info: Dict[str, object] = {
        "relation_heatmap_path": str(output_path),
        "relation_heatmap_exported": False,
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        info["relation_heatmap_error"] = f"matplotlib unavailable: {exc}"
        return info

    payload = torch.load(relation_matrix_path, map_location="cpu", weights_only=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    psi_mh = payload["psi_mh"].detach().cpu().numpy()
    psi_mr = payload["psi_mr"].detach().cpu().numpy()
    psi_hr = payload["psi_hr"].detach().cpu().numpy()

    material_labels = _ordered_label_names(label_maps.get("idx_to_material", {}), int(psi_mh.shape[0]))
    hardness_labels = _ordered_label_names(label_maps.get("idx_to_hardness", {}), int(psi_mh.shape[1]))
    roughness_labels = _ordered_label_names(label_maps.get("idx_to_roughness", {}), int(psi_mr.shape[1]))

    matrices = [
        ("Psi_mh", psi_mh, material_labels, hardness_labels, "hardness", "material"),
        ("Psi_mr", psi_mr, material_labels, roughness_labels, "roughness", "material"),
        ("Psi_hr", psi_hr, hardness_labels, roughness_labels, "roughness", "hardness"),
    ]

    vmax = max(float(np.abs(arr).max()) for _, arr, _, _, _, _ in matrices)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, (title, arr, row_labels, col_labels, xlabel, ylabel) in zip(axes, matrices):
        image = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.92)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    info["relation_heatmap_exported"] = True
    return info


def _load_run_summary(run_dir: str | Path) -> Dict[str, object]:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["summary_source"] = "summary_json"
        payload["run_dir"] = str(run_dir)
        return payload

    best_path = run_dir / "best.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        val_stats = checkpoint.get("val_stats", {}) or {}
        payload = {
            "run_dir": str(run_dir),
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "best_metric_name": str(checkpoint.get("best_metric_name", "mean_accuracy")),
            "best_metric": _safe_float(checkpoint.get("best_metric")),
            "best_val_mean_accuracy": _safe_float(val_stats.get("mean_accuracy")),
            "best_val_mean_macro_f1": _safe_float(val_stats.get("mean_macro_f1")),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(run_dir / "last.pth"),
            "summary_source": "best_checkpoint",
        }
        return payload

    raise FileNotFoundError(f"No summary.json or best.pth found under {run_dir}")


def _write_ablation_summary(
    output_path: Path,
    args,
    current_run_dir: Path,
    current_summary: Dict[str, object],
) -> Dict[str, object]:
    variant_labels = {
        "no_reasoner": "No-Reasoner",
        "frozen_reasoner": "Frozen-Reasoner",
        "reasoner_on": "Reasoner-On",
    }
    current_variant = "frozen_reasoner" if bool(args.freeze_reasoner) else "reasoner_on"
    run_dirs = {
        "no_reasoner": args.ablation_no_reasoner_run_dir,
        "frozen_reasoner": args.ablation_frozen_reasoner_run_dir,
        "reasoner_on": args.ablation_reasoner_on_run_dir,
    }

    summaries: Dict[str, Dict[str, object]] = {}
    for variant, run_dir in run_dirs.items():
        if run_dir == "":
            continue
        summaries[variant] = _load_run_summary(run_dir)

    summaries[current_variant] = {
        **current_summary,
        "run_dir": str(current_run_dir),
        "summary_source": "current_run",
    }

    no_reasoner_f1 = _safe_float(summaries.get("no_reasoner", {}).get("best_val_mean_macro_f1"))
    frozen_reasoner_f1 = _safe_float(summaries.get("frozen_reasoner", {}).get("best_val_mean_macro_f1"))
    reasoner_on_f1 = _safe_float(summaries.get("reasoner_on", {}).get("best_val_mean_macro_f1"))

    rows = []
    for variant in ("no_reasoner", "frozen_reasoner", "reasoner_on"):
        if variant not in summaries:
            continue
        payload = summaries[variant]
        run_f1 = _safe_float(payload.get("best_val_mean_macro_f1"))
        row = {
            "variant": variant,
            "variant_label": variant_labels[variant],
            "is_current_run": int(variant == current_variant),
            "run_dir": str(payload.get("run_dir", "")),
            "summary_source": str(payload.get("summary_source", "")),
            "best_epoch": payload.get("best_epoch", ""),
            "best_metric_name": payload.get("best_metric_name", ""),
            "best_metric": _safe_float(payload.get("best_metric")),
            "best_val_mean_accuracy": _safe_float(payload.get("best_val_mean_accuracy")),
            "best_val_mean_macro_f1": run_f1,
            "delta_vs_no_reasoner_mean_macro_f1": (
                None if (run_f1 is None or no_reasoner_f1 is None) else run_f1 - no_reasoner_f1
            ),
            "delta_vs_frozen_reasoner_mean_macro_f1": (
                None if (run_f1 is None or frozen_reasoner_f1 is None) else run_f1 - frozen_reasoner_f1
            ),
        }
        rows.append(row)

    info: Dict[str, object] = {
        "ablation_summary_path": str(output_path),
        "ablation_variants_present": [row["variant"] for row in rows],
        "delta_f1_reasoning_vs_no_reasoner": (
            None
            if (reasoner_on_f1 is None or no_reasoner_f1 is None)
            else reasoner_on_f1 - no_reasoner_f1
        ),
        "delta_f1_reasoning_vs_frozen_reasoner": (
            None
            if (reasoner_on_f1 is None or frozen_reasoner_f1 is None)
            else reasoner_on_f1 - frozen_reasoner_f1
        ),
    }
    if len(rows) == 0:
        info["ablation_summary_written"] = False
        return info

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "variant_label",
        "is_current_run",
        "run_dir",
        "summary_source",
        "best_epoch",
        "best_metric_name",
        "best_metric",
        "best_val_mean_accuracy",
        "best_val_mean_macro_f1",
        "delta_vs_no_reasoner_mean_macro_f1",
        "delta_vs_frozen_reasoner_mean_macro_f1",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    info["ablation_summary_written"] = True
    return info


def _build_model(args, backbone, num_material: int, num_hardness: int, num_roughness: int) -> Stage3SymbolizeWithReasonerModel:
    return Stage3SymbolizeWithReasonerModel(
        backbone=backbone,
        num_material=num_material,
        num_hardness=num_hardness,
        num_roughness=num_roughness,
        symbol_k=args.symbol_k,
        symbol_dim=args.symbol_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        reason_steps=args.reason_steps,
        reasoner_init_std=args.reasoner_init_std,
        freeze_reasoner=args.freeze_reasoner,
    )


def main():
    args = parse_args()
    dist_state = _init_distributed(args)
    distributed = bool(dist_state["distributed"])
    device = dist_state["device"]
    rank = int(dist_state["rank"])
    world_size = int(dist_state["world_size"])
    local_rank = int(dist_state["local_rank"])

    set_seed(args.seed + rank)

    run_dir = Path(args.output_root).resolve() / args.run_name
    if _is_main_process():
        run_dir = make_run_dir(args.output_root, args.run_name)
        config_path = run_dir / "config.json"
        config_payload = dict(vars(args))
        config_payload["distributed"] = distributed
        config_payload["world_size"] = world_size
        save_json(config_path, config_payload)

    if distributed:
        dist.barrier()

    if _is_main_process():
        print("\n" + "=" * 72)
        print("Stage3_method_w V3 Symbolize + Reasoner Training")
        print("=" * 72)
        print(f"Run dir: {run_dir}")
        print(f"Device: {device}")
        if distributed:
            print(f"DDP enabled: world_size={world_size}, local_rank={local_rank}")
        print(f"Data root: {args.data_root}")
        print(f"Index file: {args.index_file}")
        print(f"Stage2 checkpoint: {args.stage2_checkpoint}")
        print(f"Stage1 checkpoint override: {args.stage1_checkpoint or '(from Stage2 ckpt args)'}")
        print(f"Init checkpoint: {args.init_checkpoint or '(none)'}")
        print(
            "Loss/LR strategy: "
            f"weighted_ce={args.use_class_weighted_ce}, "
            f"label_smoothing={args.label_smoothing}, "
            f"scheduler={args.lr_scheduler}, min_lr={args.min_lr}, "
            f"best_metric={args.best_metric}"
        )
        print(
            "Task weighting: "
            f"loss=[{args.loss_weight_material}, {args.loss_weight_hardness}, {args.loss_weight_roughness}] "
            "class_weight_power="
            f"[{args.material_class_weight_power}, {args.hardness_class_weight_power}, {args.roughness_class_weight_power}]"
        )
        print(
            "Sampling strategy: "
            f"train_center_frame_step={args.train_center_frame_step}, "
            f"eval_center_frame_step={args.eval_center_frame_step}"
        )
        print(
            "Symbolize config: "
            f"K={args.symbol_k}, dim={args.symbol_dim}, "
            f"tau={args.tau_start}->{args.tau_end}, anneal_epochs={args.tau_anneal_epochs}"
        )
        print(
            "Reasoner config: "
            f"steps={args.reason_steps}, lr_reasoner={args.lr_reasoner}, "
            f"lambda_sparse={args.lambda_sparse}, lambda_iter={args.lambda_iter}, "
            f"freeze_reasoner={args.freeze_reasoner}, init_std={args.reasoner_init_std}"
        )
        if _is_staged_training(args):
            print(
                "Phase schedule: "
                f"A={args.phase_a_epochs} (heads/symbol only), "
                f"B={args.phase_b_epochs} (reasoner on), "
                f"C={args.phase_c_epochs} (backbone finetune, lr_backbone={args.lr_backbone})"
            )
        else:
            print(
                "Phase schedule: "
                f"joint(backbone_trainable={not args.freeze_backbone}, "
                f"reasoner_trainable={not args.freeze_reasoner}, "
                f"lr_backbone={args.lr_backbone})"
            )

    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_loader, train_dataset = build_causal_dataloader(
        root_dir=args.data_root,
        index_file=args.index_file,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        random_seed=args.split_seed,
        split_ratios=split_ratios,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        train_center_frame_step=args.train_center_frame_step,
        eval_center_frame_step=args.eval_center_frame_step,
        augment=True,
        strict_frame_loading=args.strict_frame_loading,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    val_loader, val_dataset = build_causal_dataloader(
        root_dir=args.data_root,
        index_file=args.index_file,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        random_seed=args.split_seed,
        split_ratios=split_ratios,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        train_center_frame_step=args.train_center_frame_step,
        eval_center_frame_step=args.eval_center_frame_step,
        augment=False,
        strict_frame_loading=args.strict_frame_loading,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    label_maps = train_dataset.get_label_maps()
    if _is_main_process():
        save_json(run_dir / "label_maps.json", label_maps)

    num_material = len(label_maps["material_to_idx"])
    num_hardness = len(label_maps["hardness_to_idx"])
    num_roughness = len(label_maps["roughness_to_idx"])
    weight_material = None
    weight_hardness = None
    weight_roughness = None

    if args.use_class_weighted_ce:
        counts_m, weights_m, weight_material = _compute_class_weight_from_samples(
            train_dataset.samples,
            key="material",
            num_classes=num_material,
            power=args.material_class_weight_power,
            device=device,
        )
        counts_h, weights_h, weight_hardness = _compute_class_weight_from_samples(
            train_dataset.samples,
            key="hardness",
            num_classes=num_hardness,
            power=args.hardness_class_weight_power,
            device=device,
        )
        counts_r, weights_r, weight_roughness = _compute_class_weight_from_samples(
            train_dataset.samples,
            key="roughness",
            num_classes=num_roughness,
            power=args.roughness_class_weight_power,
            device=device,
        )
    else:
        counts_m = counts_h = counts_r = None
        weights_m = weights_h = weights_r = None

    if _is_main_process():
        print(
            f"\nLabel classes: material={num_material}, hardness={num_hardness}, roughness={num_roughness}"
        )
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        if args.use_class_weighted_ce:
            print("Class counts/weights (train split):")
            print(f"  material counts={counts_m}, weights={[round(x, 4) for x in weights_m]}")
            print(f"  hardness counts={counts_h}, weights={[round(x, 4) for x in weights_h]}")
            print(f"  roughness counts={counts_r}, weights={[round(x, 4) for x in weights_r]}")

    backbone, backbone_meta = load_stage2_backbone(
        stage2_checkpoint_path=args.stage2_checkpoint,
        stage1_checkpoint_override=args.stage1_checkpoint,
        device=str(device),
    )
    if _is_main_process():
        print("\nStage2 backbone loaded:")
        print(f"  stage1_checkpoint: {backbone_meta.stage1_checkpoint}")
        print(f"  projection_dim: {backbone_meta.projection_dim}")
        print(f"  missing_keys: {backbone_meta.missing_keys}, unexpected_keys: {backbone_meta.unexpected_keys}")

    model = _build_model(
        args=args,
        backbone=backbone,
        num_material=num_material,
        num_hardness=num_hardness,
        num_roughness=num_roughness,
    ).to(device)
    init_result = _load_init_checkpoint(model, args.init_checkpoint)

    if _is_main_process():
        print("\nStage3 head architecture:")
        print("  base_feature_fusion: concat(encoder_h, z_contact)")
        print("  symbolize: Symbolize(concat(encoder_h, z_contact))")
        print("  reasoner: Psi_mh / Psi_mr / Psi_hr message passing on unary logits")
        print(
            f"  dims: encoder={model.encoder_dim}, contact={model.contact_dim}, "
            f"fused_base={model.fused_dim}, symbol={model.symbol_dim}, "
            f"fused_with_symbol={model.fused_with_symbol_dim}, shared_hidden={model.hidden_dim}"
        )
        print(
            "  relation init/load: "
            f"loaded_keys={init_result['loaded_keys']}, "
            f"missing_keys={init_result['missing_keys']}, "
            f"unexpected_keys={init_result['unexpected_keys']}, "
            f"mismatched_keys={init_result['mismatched_keys']}"
        )

    model.set_backbone_trainable(True)
    model.set_reasoner_trainable(True)

    ddp_find_unused_parameters = bool(
        _is_staged_training(args)
        or args.freeze_backbone
        or args.freeze_reasoner
    )
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
        )

    active_phase_name = _phase_name_for_epoch(epoch=0, args=args)
    active_phase_state = _apply_training_phase(model=model, phase_name=active_phase_name, args=args)
    optimizer, scheduler, active_param_groups = _build_optimizer_and_scheduler(
        model=model,
        args=args,
        phase_name=active_phase_name,
    )

    if _is_main_process():
        print(
            "Initial training phase: "
            f"{active_phase_name} "
            f"(backbone_trainable={active_phase_state['backbone_trainable']}, "
            f"reasoner_trainable={active_phase_state['reasoner_trainable']}, "
            f"param_groups={active_param_groups}, "
            f"ddp_find_unused_parameters={ddp_find_unused_parameters})"
        )

    use_wandb = _is_main_process() and WANDB_AVAILABLE and (not args.disable_wandb)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags if args.wandb_tags else ["stage3_w", "v3", "symbolize", "reasoner", "causal_data"],
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        print(f"Wandb initialized: project={args.wandb_project}, run={args.wandb_name}")
    elif _is_main_process():
        print("Wandb disabled or unavailable.")

    train_log_path = run_dir / "train_log.jsonl"
    val_metrics_csv = run_dir / "val_metrics.csv"
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    relation_matrix_path = run_dir / "relation_matrix.pt"
    relation_heatmap_path = run_dir / "relation_heatmap.png"
    reason_trace_path = run_dir / "reason_trace.jsonl"
    ablation_summary_path = run_dir / "ablation_summary.csv"

    best_metric_name = str(args.best_metric)
    best_metric = -1.0
    best_epoch = -1
    best_phase_name = ""
    best_val_mean_accuracy = -1.0
    best_val_mean_macro_f1 = -1.0
    start = time.time()
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    val_sampler = val_loader.sampler if isinstance(val_loader.sampler, DistributedSampler) else None

    for epoch in range(args.epochs):
        epoch_phase_name = _phase_name_for_epoch(epoch=epoch, args=args)
        if epoch_phase_name != active_phase_name:
            active_phase_name = epoch_phase_name
            active_phase_state = _apply_training_phase(
                model=model,
                phase_name=active_phase_name,
                args=args,
            )
            optimizer, scheduler, active_param_groups = _build_optimizer_and_scheduler(
                model=model,
                args=args,
                phase_name=active_phase_name,
            )
            if _is_main_process():
                print(
                    "Phase transition: "
                    f"epoch={epoch} -> {active_phase_name} "
                    f"(backbone_trainable={active_phase_state['backbone_trainable']}, "
                    f"reasoner_trainable={active_phase_state['reasoner_trainable']}, "
                    f"param_groups={active_param_groups})"
                )

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        symbol_temperature = _compute_symbol_temperature(epoch=epoch, args=args)

        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=args.amp,
            num_material=num_material,
            num_hardness=num_hardness,
            num_roughness=num_roughness,
            weight_material=weight_material,
            weight_hardness=weight_hardness,
            weight_roughness=weight_roughness,
            label_smoothing=args.label_smoothing,
            loss_weight_material=args.loss_weight_material,
            loss_weight_hardness=args.loss_weight_hardness,
            loss_weight_roughness=args.loss_weight_roughness,
            symbol_k=args.symbol_k,
            symbol_temperature=symbol_temperature,
            lambda_conf=args.lambda_conf,
            lambda_bal=args.lambda_bal,
            lambda_sparse=args.lambda_sparse,
            lambda_iter=args.lambda_iter,
            relation_sparsity_threshold=args.relation_sparsity_threshold,
            log_freq=args.log_freq if _is_main_process() else 0,
        )

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stats = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                num_material=num_material,
                num_hardness=num_hardness,
                num_roughness=num_roughness,
                weight_material=weight_material,
                weight_hardness=weight_hardness,
                weight_roughness=weight_roughness,
                label_smoothing=args.label_smoothing,
                loss_weight_material=args.loss_weight_material,
                loss_weight_hardness=args.loss_weight_hardness,
                loss_weight_roughness=args.loss_weight_roughness,
                symbol_k=args.symbol_k,
                symbol_temperature=symbol_temperature,
                lambda_conf=args.lambda_conf,
                lambda_bal=args.lambda_bal,
                lambda_sparse=args.lambda_sparse,
                lambda_iter=args.lambda_iter,
                relation_sparsity_threshold=args.relation_sparsity_threshold,
                log_freq=args.val_log_freq if _is_main_process() else 0,
            )
        else:
            val_stats = {}

        current_mean_accuracy = float(val_stats.get("mean_accuracy", -1.0))
        current_mean_macro_f1 = float(val_stats.get("mean_macro_f1", -1.0))
        best_val_mean_accuracy = max(best_val_mean_accuracy, current_mean_accuracy)
        best_val_mean_macro_f1 = max(best_val_mean_macro_f1, current_mean_macro_f1)
        current_lr_head = _get_group_lr(optimizer=optimizer, group_name="base")
        current_lr_reasoner = _get_group_lr(optimizer=optimizer, group_name="reasoner")
        current_lr_backbone = _get_group_lr(optimizer=optimizer, group_name="backbone")

        if _is_main_process():
            row = {
                "epoch": float(epoch),
                "phase": active_phase_name,
                "tau": float(symbol_temperature),
                "train_loss": float(train_stats["loss"]),
                "train_mean_accuracy": float(train_stats["mean_accuracy"]),
                "train_mean_macro_f1": float(train_stats["mean_macro_f1"]),
                "train_symbol_usage_ratio": float(train_stats["symbol_usage_ratio"]),
                "train_top1_symbol_collapse_ratio": float(train_stats["top1_symbol_collapse_ratio"]),
                "train_reason_contrib_ratio": float(train_stats["reason_contrib_ratio"]),
                "train_relation_sparsity": float(train_stats["relation_sparsity"]),
                "val_loss": float(val_stats.get("loss", float("nan"))),
                "val_mean_accuracy": float(val_stats.get("mean_accuracy", float("nan"))),
                "val_mean_macro_f1": float(val_stats.get("mean_macro_f1", float("nan"))),
                "val_acc_material": float(val_stats.get("acc_material", float("nan"))),
                "val_acc_hardness": float(val_stats.get("acc_hardness", float("nan"))),
                "val_acc_roughness": float(val_stats.get("acc_roughness", float("nan"))),
                "val_symbol_usage_ratio": float(val_stats.get("symbol_usage_ratio", float("nan"))),
                "val_top1_symbol_collapse_ratio": float(
                    val_stats.get("top1_symbol_collapse_ratio", float("nan"))
                ),
                "val_symbol_perplexity_ratio": float(val_stats.get("symbol_perplexity_ratio", float("nan"))),
                "val_mean_sample_entropy_norm": float(
                    val_stats.get("mean_sample_entropy_norm", float("nan"))
                ),
                "val_reason_contrib_ratio": float(val_stats.get("reason_contrib_ratio", float("nan"))),
                "val_message_delta_norm": float(val_stats.get("message_delta_norm", float("nan"))),
                "val_relation_abs_mean": float(val_stats.get("relation_abs_mean", float("nan"))),
                "val_relation_sparsity": float(val_stats.get("relation_sparsity", float("nan"))),
                "val_loss_reason_sparse": float(val_stats.get("loss_reason_sparse", float("nan"))),
                "val_loss_reason_iter": float(val_stats.get("loss_reason_iter", float("nan"))),
                "lr_head": current_lr_head,
                "lr_reasoner": current_lr_reasoner,
                "lr_backbone": current_lr_backbone,
            }
            append_csv_row(val_metrics_csv, row)
            append_jsonl(
                train_log_path,
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "epoch": epoch,
                    "phase": active_phase_name,
                    "phase_state": active_phase_state,
                    "symbol_temperature": symbol_temperature,
                    "train": train_stats,
                    "val": val_stats,
                },
            )

        if use_wandb:
            wandb_log = {"epoch": epoch}
            wandb_log.update({f"train/{k}": v for k, v in train_stats.items() if isinstance(v, (int, float))})
            wandb_log.update({f"val/{k}": v for k, v in val_stats.items() if isinstance(v, (int, float))})
            wandb_log["train/lr_head"] = current_lr_head
            wandb_log["train/lr_reasoner"] = current_lr_reasoner
            wandb_log["train/lr_backbone"] = current_lr_backbone
            wandb.log(wandb_log)

        if _is_main_process() and (epoch % args.save_freq == 0 or epoch == args.epochs - 1):
            torch.save(
                {
                    "model": _unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_meta": _to_serializable(backbone_meta.__dict__),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "distributed": distributed,
                    "world_size": world_size,
                    "phase_name": active_phase_name,
                    "phase_state": active_phase_state,
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "label_maps": label_maps,
                },
                last_path,
            )

        if best_metric_name == "mean_accuracy":
            current_metric = current_mean_accuracy
        else:
            current_metric = current_mean_macro_f1
        if _is_main_process() and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_phase_name = active_phase_name
            torch.save(
                {
                    "model": _unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_meta": _to_serializable(backbone_meta.__dict__),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "distributed": distributed,
                    "world_size": world_size,
                    "phase_name": active_phase_name,
                    "phase_state": active_phase_state,
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "best_metric_name": best_metric_name,
                    "best_metric": best_metric,
                    "label_maps": label_maps,
                },
                best_path,
            )
            _save_relation_matrix(
                path=relation_matrix_path,
                model=model,
                epoch=epoch,
                metric_name=best_metric_name,
                metric_value=best_metric,
                sparsity_threshold=args.relation_sparsity_threshold,
            )

        if _is_main_process():
            print(
                f"[Epoch {epoch:03d}] "
                f"phase={active_phase_name} "
                f"tau={symbol_temperature:.4f} "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_acc={train_stats['mean_accuracy']:.4f} "
                f"val_acc={val_stats.get('mean_accuracy', float('nan')):.4f} "
                f"train_f1={train_stats['mean_macro_f1']:.4f} "
                f"val_f1={val_stats.get('mean_macro_f1', float('nan')):.4f} "
                f"val_reason={val_stats.get('reason_contrib_ratio', float('nan')):.4f} "
                f"best_{best_metric_name}={best_metric:.4f}@{best_epoch}"
            )
            print(
                "           "
                f"val_loss_heads[m/h/r]="
                f"{val_stats.get('loss_material', float('nan')):.4f}/"
                f"{val_stats.get('loss_hardness', float('nan')):.4f}/"
                f"{val_stats.get('loss_roughness', float('nan')):.4f} "
                f"val_reason[sparse/iter]="
                f"{val_stats.get('loss_reason_sparse', float('nan')):.4f}/"
                f"{val_stats.get('loss_reason_iter', float('nan')):.4f}",
            )
            print(
                "           "
                f"val_symbol[use/collapse/ppl]="
                f"{val_stats.get('symbol_usage_ratio', float('nan')):.3f}/"
                f"{val_stats.get('top1_symbol_collapse_ratio', float('nan')):.3f}/"
                f"{val_stats.get('symbol_perplexity_ratio', float('nan')):.3f} "
                f"val_relation[sparsity/abs]="
                f"{val_stats.get('relation_sparsity', float('nan')):.3f}/"
                f"{val_stats.get('relation_abs_mean', float('nan')):.6f}",
            )

        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - start
    reason_trace_written = 0
    relation_heatmap_info: Dict[str, object] = {
        "relation_heatmap_path": str(relation_heatmap_path),
        "relation_heatmap_exported": False,
    }

    if _is_main_process() and (not args.disable_reason_trace_export) and best_path.exists():
        export_backbone, _ = load_stage2_backbone(
            stage2_checkpoint_path=args.stage2_checkpoint,
            stage1_checkpoint_override=args.stage1_checkpoint,
            device=str(device),
        )
        export_model = _build_model(
            args=args,
            backbone=export_backbone,
            num_material=num_material,
            num_hardness=num_hardness,
            num_roughness=num_roughness,
        ).to(device)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        export_model.load_state_dict(best_checkpoint["model"], strict=True)
        _save_relation_matrix(
            path=relation_matrix_path,
            model=export_model,
            epoch=best_epoch,
            metric_name=best_metric_name,
            metric_value=best_metric,
            sparsity_threshold=args.relation_sparsity_threshold,
        )
        export_val_loader, _ = build_causal_dataloader(
            root_dir=args.data_root,
            index_file=args.index_file,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            random_seed=args.split_seed,
            split_ratios=split_ratios,
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            train_center_frame_step=args.train_center_frame_step,
            eval_center_frame_step=args.eval_center_frame_step,
            augment=False,
            strict_frame_loading=args.strict_frame_loading,
            distributed=False,
            rank=0,
            world_size=1,
        )
        reason_trace_written = export_reason_trace(
            model=export_model,
            data_loader=export_val_loader,
            device=device,
            label_maps=label_maps,
            output_path=reason_trace_path,
            symbol_temperature=_compute_symbol_temperature(epoch=best_epoch, args=args),
            max_samples=args.reason_trace_max_samples,
        )

    if _is_main_process():
        if best_path.exists() and (not args.disable_relation_heatmap_export) and relation_matrix_path.exists():
            relation_heatmap_info = _export_relation_heatmap(
                relation_matrix_path=relation_matrix_path,
                output_path=relation_heatmap_path,
                label_maps=label_maps,
            )
        elif args.disable_relation_heatmap_export:
            relation_heatmap_info["relation_heatmap_error"] = "disabled_by_flag"

        summary = {
            "run_dir": str(run_dir),
            "best_epoch": best_epoch,
            "best_phase_name": best_phase_name,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "best_val_mean_accuracy": best_val_mean_accuracy,
            "best_val_mean_macro_f1": best_val_mean_macro_f1,
            "elapsed_sec": elapsed,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "relation_matrix_path": str(relation_matrix_path),
            "current_variant": "frozen_reasoner" if bool(args.freeze_reasoner) else "reasoner_on",
            "reason_trace_path": str(reason_trace_path),
            "reason_trace_written": int(reason_trace_written),
            "distributed": distributed,
            "world_size": world_size,
        }
        summary.update(relation_heatmap_info)
        ablation_info = _write_ablation_summary(
            output_path=ablation_summary_path,
            args=args,
            current_run_dir=run_dir,
            current_summary=summary,
        )
        summary.update(ablation_info)
        save_json(run_dir / "summary.json", summary)
        print("\nTraining done.")
        print(f"  output_dir: {run_dir}")
        print(f"  best_epoch: {best_epoch}")
        print(f"  best_phase_name: {best_phase_name}")
        print(f"  best_metric_name: {best_metric_name}")
        print(f"  best_metric: {best_metric:.6f}")
        print(f"  best_val_mean_accuracy: {best_val_mean_accuracy:.6f}")
        print(f"  best_val_mean_macro_f1: {best_val_mean_macro_f1:.6f}")
        print(f"  relation_matrix: {relation_matrix_path}")
        print(
            f"  relation_heatmap: {relation_heatmap_info['relation_heatmap_path']} "
            f"(exported={relation_heatmap_info['relation_heatmap_exported']})"
        )
        print(f"  reason_trace: {reason_trace_path} ({reason_trace_written} rows)")
        print(
            f"  ablation_summary: {ablation_info['ablation_summary_path']} "
            f"(written={ablation_info.get('ablation_summary_written', False)})"
        )
        if ablation_info.get("delta_f1_reasoning_vs_no_reasoner") is not None:
            print(
                "  delta_f1_reasoning_vs_no_reasoner: "
                f"{ablation_info['delta_f1_reasoning_vs_no_reasoner']:.6f}"
            )
        if ablation_info.get("delta_f1_reasoning_vs_frozen_reasoner") is not None:
            print(
                "  delta_f1_reasoning_vs_frozen_reasoner: "
                f"{ablation_info['delta_f1_reasoning_vs_frozen_reasoner']:.6f}"
            )
        print(f"  elapsed_sec: {elapsed:.2f}")
        print(f"  best_checkpoint: {best_path}")
        print(f"  last_checkpoint: {last_path}")
        print(f"  distributed: {distributed} (world_size={world_size})")

        if use_wandb:
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["best_phase_name"] = best_phase_name
            wandb.summary["best_metric_name"] = best_metric_name
            wandb.summary["best_metric"] = best_metric
            wandb.summary["best_val_mean_accuracy"] = best_val_mean_accuracy
            wandb.summary["best_val_mean_macro_f1"] = best_val_mean_macro_f1
            wandb.summary["reason_trace_written"] = reason_trace_written
            wandb.summary["relation_heatmap_exported"] = relation_heatmap_info["relation_heatmap_exported"]
            if ablation_info.get("delta_f1_reasoning_vs_no_reasoner") is not None:
                wandb.summary["delta_f1_reasoning_vs_no_reasoner"] = ablation_info[
                    "delta_f1_reasoning_vs_no_reasoner"
                ]
            if ablation_info.get("delta_f1_reasoning_vs_frozen_reasoner") is not None:
                wandb.summary["delta_f1_reasoning_vs_frozen_reasoner"] = ablation_info[
                    "delta_f1_reasoning_vs_frozen_reasoner"
                ]
            wandb.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
