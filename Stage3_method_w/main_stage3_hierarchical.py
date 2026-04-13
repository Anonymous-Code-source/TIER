#!/usr/bin/env python3
"""
Hierarchical Stage3 training entrypoint.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from Stage3_method_w.config_stage3_hierarchical import parse_args
from Stage3_method_w.engine_stage3_hierarchical import evaluate, train_one_epoch
from Stage3_method_w.model_stage3_hierarchical import (
    Stage3HierarchicalModel,
    load_stage2_backbone,
)
from Stage3_method_w.stage3_dataset_causal_hierarchical import (
    build_hierarchical_causal_dataloader,
)
from Stage3_method_w.stage3_task_spec import (
    CLASSIFICATION_TASKS,
    LAYER_TASKS,
    REGRESSION_TASKS,
    TASK_ORDER,
    label_count_for_task,
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
    return (
        int(args.phase_p_epochs)
        + int(args.phase_pi_epochs)
        + int(args.phase_pic_epochs)
        + int(args.phase_joint_epochs)
    ) > 0


def _phase_name_for_epoch(epoch: int, args) -> str:
    if not _is_staged_training(args):
        return "joint"
    if epoch < int(args.phase_p_epochs):
        return "phase_p"
    if epoch < int(args.phase_p_epochs) + int(args.phase_pi_epochs):
        return "phase_pi"
    if epoch < int(args.phase_p_epochs) + int(args.phase_pi_epochs) + int(args.phase_pic_epochs):
        return "phase_pic"
    return "phase_joint"


def _phase_total_epochs(phase_name: str, args) -> int:
    mapping = {
        "phase_p": max(int(args.phase_p_epochs), 1),
        "phase_pi": max(int(args.phase_pi_epochs), 1),
        "phase_pic": max(int(args.phase_pic_epochs), 1),
        "phase_joint": max(int(args.phase_joint_epochs), 1),
    }
    return mapping.get(phase_name, max(int(args.epochs), 1))


def _phase_state(phase_name: str, args) -> Dict[str, object]:
    if phase_name == "phase_p":
        return {
            "backbone_trainable": False,
            "physical_trainable": True,
            "interaction_trainable": False,
            "contact_trainable": False,
            "enabled_layers": ["physical"],
        }
    if phase_name == "phase_pi":
        return {
            "backbone_trainable": False,
            "physical_trainable": True,
            "interaction_trainable": True,
            "contact_trainable": False,
            "enabled_layers": ["physical", "interaction"],
        }
    if phase_name == "phase_pic":
        return {
            "backbone_trainable": False,
            "physical_trainable": True,
            "interaction_trainable": True,
            "contact_trainable": True,
            "enabled_layers": ["physical", "interaction", "contact"],
        }
    return {
        "backbone_trainable": not bool(args.freeze_backbone),
        "physical_trainable": True,
        "interaction_trainable": True,
        "contact_trainable": True,
        "enabled_layers": ["physical", "interaction", "contact"],
    }


def _apply_training_phase(model: torch.nn.Module, phase_name: str, args) -> Dict[str, object]:
    core = _unwrap_model(model)
    phase = _phase_state(phase_name=phase_name, args=args)
    core.set_backbone_trainable(bool(phase["backbone_trainable"]))
    core.set_physical_trainable(bool(phase["physical_trainable"]))
    core.set_interaction_trainable(bool(phase["interaction_trainable"]))
    core.set_contact_trainable(bool(phase["contact_trainable"]))
    physical_reasoner_trainable = bool(phase["physical_trainable"]) and (not bool(args.freeze_reasoners))
    contact_reasoner_trainable = bool(phase["contact_trainable"]) and (not bool(args.freeze_reasoners))
    core.set_physical_reasoner_trainable(physical_reasoner_trainable)
    core.set_contact_reasoner_trainable(contact_reasoner_trainable)
    return phase


def _build_optimizer_and_scheduler(
    model: torch.nn.Module,
    args,
    phase_name: str,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, List[str]]:
    core = _unwrap_model(model)
    backbone_param_ids = {id(p) for p in core.backbone.parameters()}
    reasoner_param_ids = {
        id(p) for p in list(core.physical.reasoner.parameters()) + list(core.contact.reasoner.parameters())
    }

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


def _build_model(args, backbone, label_counts: Dict[str, int]) -> Stage3HierarchicalModel:
    return Stage3HierarchicalModel(
        backbone=backbone,
        num_material=label_counts["material"],
        num_hardness=label_counts["hardness"],
        num_roughness=label_counts["roughness"],
        num_deformation=label_counts["deformation"],
        num_slippage=label_counts["slippage"],
        num_rotation=label_counts["rotation"],
        num_force=label_counts["force"],
        obs_symbol_k=args.obs_symbol_k,
        obs_symbol_dim=args.obs_symbol_dim,
        physical_symbol_k=args.physical_symbol_k,
        physical_symbol_dim=args.physical_symbol_dim,
        interaction_symbol_k=args.interaction_symbol_k,
        interaction_symbol_dim=args.interaction_symbol_dim,
        contact_symbol_k=args.contact_symbol_k,
        contact_symbol_dim=args.contact_symbol_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        reason_steps_physical=args.reason_steps_physical,
        reason_steps_contact=args.reason_steps_contact,
        reasoner_init_std=args.reasoner_init_std,
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
        print("Stage3_method_w Hierarchical Stage3 Training")
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
            "Hierarchy symbols: "
            f"obs(K={args.obs_symbol_k},dim={args.obs_symbol_dim}) "
            f"physical(K={args.physical_symbol_k},dim={args.physical_symbol_dim}) "
            f"interaction(K={args.interaction_symbol_k},dim={args.interaction_symbol_dim}) "
            f"contact(K={args.contact_symbol_k},dim={args.contact_symbol_dim})"
        )
        print(
            "Phase schedule: "
            f"P={args.phase_p_epochs}, PI={args.phase_pi_epochs}, "
            f"PIC={args.phase_pic_epochs}, Joint={args.phase_joint_epochs}"
        )

    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_loader, train_dataset = build_hierarchical_causal_dataloader(
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
    val_loader, val_dataset = build_hierarchical_causal_dataloader(
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

    label_counts = {task: label_count_for_task(label_maps, task) for task in TASK_ORDER}
    if _is_main_process():
        print("\nLabel targets:")
        for task in TASK_ORDER:
            kind = "regression" if task in REGRESSION_TASKS else "classes"
            print(f"  {task}: {kind}={label_counts[task]}")
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    task_class_weights: Dict[str, torch.Tensor | None] = {}
    if args.use_class_weighted_ce:
        if _is_main_process():
            print("Class counts/weights (train split):")
        for task in CLASSIFICATION_TASKS:
            counts, weights, weight_tensor = _compute_class_weight_from_samples(
                train_dataset.samples,
                key=task,
                num_classes=label_counts[task],
                power=args.class_weight_power,
                device=device,
            )
            task_class_weights[task] = weight_tensor
            if _is_main_process():
                print(f"  {task} counts={counts}, weights={[round(x, 4) for x in weights]}")
        for task in REGRESSION_TASKS:
            task_class_weights[task] = None
    else:
        task_class_weights = {task: None for task in TASK_ORDER}

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

    model = _build_model(args=args, backbone=backbone, label_counts=label_counts).to(device)
    init_result = _load_init_checkpoint(model, args.init_checkpoint)
    if _is_main_process():
        print("\nHierarchical Stage3 model:")
        print("  path: x -> s_x -> s^p -> s^i -> s^c")
        print(
            f"  dims: encoder={model.encoder_dim}, contact={model.contact_dim}, "
            f"fused={model.fused_dim}, hidden={model.hidden_dim}"
        )
        print(
            "  init/load: "
            f"loaded_keys={init_result['loaded_keys']}, "
            f"missing_keys={init_result['missing_keys']}, "
            f"unexpected_keys={init_result['unexpected_keys']}, "
            f"mismatched_keys={init_result['mismatched_keys']}"
        )

    model.set_backbone_trainable(True)
    model.set_physical_trainable(True)
    model.set_interaction_trainable(True)
    model.set_contact_trainable(True)
    model.set_physical_reasoner_trainable(not bool(args.freeze_reasoners))
    model.set_contact_reasoner_trainable(not bool(args.freeze_reasoners))

    ddp_find_unused_parameters = bool(
        _is_staged_training(args)
        or args.freeze_backbone
        or args.freeze_reasoners
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
            f"(enabled_layers={active_phase_state['enabled_layers']}, "
            f"backbone_trainable={active_phase_state['backbone_trainable']}, "
            f"param_groups={active_param_groups}, "
            f"ddp_find_unused_parameters={ddp_find_unused_parameters})"
        )

    use_wandb = _is_main_process() and WANDB_AVAILABLE and (not args.disable_wandb)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags if args.wandb_tags else ["stage3_w", "hierarchical", "causal_data"],
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

    best_metric_name = str(args.best_metric)
    best_metric = -1.0
    best_epoch = -1
    best_phase_name = ""
    start = time.time()
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    val_sampler = val_loader.sampler if isinstance(val_loader.sampler, DistributedSampler) else None

    layer_loss_weights = {
        "physical": float(args.layer_loss_weight_physical),
        "interaction": float(args.layer_loss_weight_interaction),
        "contact": float(args.layer_loss_weight_contact),
    }

    for epoch in range(args.epochs):
        epoch_phase_name = _phase_name_for_epoch(epoch=epoch, args=args)
        if epoch_phase_name != active_phase_name:
            active_phase_name = epoch_phase_name
            active_phase_state = _apply_training_phase(model=model, phase_name=active_phase_name, args=args)
            optimizer, scheduler, active_param_groups = _build_optimizer_and_scheduler(
                model=model,
                args=args,
                phase_name=active_phase_name,
            )
            if _is_main_process():
                print(
                    "Phase transition: "
                    f"epoch={epoch} -> {active_phase_name} "
                    f"(enabled_layers={active_phase_state['enabled_layers']}, "
                    f"backbone_trainable={active_phase_state['backbone_trainable']}, "
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
            task_class_weights=task_class_weights,
            label_smoothing=args.label_smoothing,
            layer_loss_weights=layer_loss_weights,
            enabled_layers=active_phase_state["enabled_layers"],
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
                task_class_weights=task_class_weights,
                label_smoothing=args.label_smoothing,
                layer_loss_weights=layer_loss_weights,
                enabled_layers=active_phase_state["enabled_layers"],
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

        metric_key = "mean_layer_macro_f1" if best_metric_name == "mean_layer_macro_f1" else "mean_layer_accuracy"
        current_metric = float(val_stats.get(metric_key, -1.0))
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            best_phase_name = active_phase_name

        current_lr_head = _get_group_lr(optimizer=optimizer, group_name="base")
        current_lr_reasoner = _get_group_lr(optimizer=optimizer, group_name="reasoner")
        current_lr_backbone = _get_group_lr(optimizer=optimizer, group_name="backbone")

        row = {
            "epoch": epoch,
            "phase_name": active_phase_name,
            "symbol_temperature": symbol_temperature,
            "train_loss": float(train_stats.get("loss", 0.0)),
            "train_mean_active_layer_accuracy": float(train_stats.get("mean_active_layer_accuracy", 0.0)),
            "train_mean_active_layer_macro_f1": float(train_stats.get("mean_active_layer_macro_f1", 0.0)),
            "train_mean_layer_accuracy": float(train_stats.get("mean_layer_accuracy", 0.0)),
            "train_mean_layer_macro_f1": float(train_stats.get("mean_layer_macro_f1", 0.0)),
            "val_loss": float(val_stats.get("loss", 0.0)),
            "val_mean_active_layer_accuracy": float(val_stats.get("mean_active_layer_accuracy", 0.0)),
            "val_mean_active_layer_macro_f1": float(val_stats.get("mean_active_layer_macro_f1", 0.0)),
            "val_mean_layer_accuracy": float(val_stats.get("mean_layer_accuracy", 0.0)),
            "val_mean_layer_macro_f1": float(val_stats.get("mean_layer_macro_f1", 0.0)),
            "val_layer_macro_f1_physical": float(val_stats.get("layer_macro_f1_physical", 0.0)),
            "val_layer_macro_f1_interaction": float(val_stats.get("layer_macro_f1_interaction", 0.0)),
            "val_layer_macro_f1_contact": float(val_stats.get("layer_macro_f1_contact", 0.0)),
            "lr_head": current_lr_head,
            "lr_reasoner": current_lr_reasoner,
            "lr_backbone": current_lr_backbone,
            "is_best": int(is_best),
        }
        if _is_main_process():
            append_csv_row(val_metrics_csv, row)
            append_jsonl(
                train_log_path,
                {
                    "epoch": epoch,
                    "phase_name": active_phase_name,
                    "symbol_temperature": symbol_temperature,
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                },
            )

            if use_wandb:
                log_payload = {"epoch": epoch}
                log_payload.update({f"train/{k}": v for k, v in train_stats.items()})
                log_payload.update({f"val/{k}": v for k, v in val_stats.items()})
                log_payload["train/lr_head"] = current_lr_head
                log_payload["train/lr_reasoner"] = current_lr_reasoner
                log_payload["train/lr_backbone"] = current_lr_backbone
                wandb.log(log_payload)

            print(
                f"[Epoch {epoch:03d}] phase={active_phase_name} tau={symbol_temperature:.4f} "
                f"train_loss={train_stats.get('loss', 0.0):.4f} "
                f"train_f1(all/active)="
                f"{train_stats.get('mean_layer_macro_f1', 0.0):.4f}/"
                f"{train_stats.get('mean_active_layer_macro_f1', 0.0):.4f} "
                f"val_f1(all/active)="
                f"{val_stats.get('mean_layer_macro_f1', 0.0):.4f}/"
                f"{val_stats.get('mean_active_layer_macro_f1', 0.0):.4f} "
                f"val_layers[p/i/c]="
                f"{val_stats.get('layer_macro_f1_physical', 0.0):.4f}/"
                f"{val_stats.get('layer_macro_f1_interaction', 0.0):.4f}/"
                f"{val_stats.get('layer_macro_f1_contact', 0.0):.4f} "
                f"best_{metric_key}={best_metric:.4f}@{best_epoch}"
            )

        if _is_main_process():
            checkpoint = {
                "epoch": epoch,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "phase_name": active_phase_name,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "best_metric_name": best_metric_name,
                "best_metric": best_metric,
            }
            torch.save(checkpoint, last_path)
            if is_best:
                torch.save(checkpoint, best_path)
                _save_relation_matrix(
                    path=relation_matrix_path,
                    model=model,
                    epoch=epoch,
                    metric_name=best_metric_name,
                    metric_value=best_metric,
                    sparsity_threshold=args.relation_sparsity_threshold,
                )

        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - start
    summary = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_phase_name": str(best_phase_name),
        "best_metric_name": str(best_metric_name),
        "best_metric": float(best_metric),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "relation_matrix": str(relation_matrix_path),
        "elapsed_sec": float(elapsed),
        "distributed": bool(distributed),
        "world_size": int(world_size),
    }
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        summary["best_val_stats"] = best_checkpoint.get("val_stats", {})
    if _is_main_process():
        save_json(run_dir / "summary.json", summary)
        print("\nTraining done.")
        print(f"  output_dir: {run_dir}")
        print(f"  best_epoch: {best_epoch}")
        print(f"  best_phase_name: {best_phase_name}")
        print(f"  best_metric_name: {best_metric_name}")
        print(f"  best_metric: {best_metric:.6f}")
        print(f"  relation_matrix: {relation_matrix_path}")
        print(f"  elapsed_sec: {elapsed:.2f}")
        print(f"  best_checkpoint: {best_path}")
        print(f"  last_checkpoint: {last_path}")
        print(f"  distributed: {distributed} (world_size={world_size})")

    if use_wandb:
        wandb.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
