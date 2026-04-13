#!/usr/bin/env python3
"""
Configuration for Stage3_method_w V3 Symbolize + Reasoner training on Causal_data.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage3_method_w V3 Symbolize + Reasoner training on Causal_data"
    )

    # Data
    parser.add_argument("--data_root", type=str, default="./Causal_data")
    parser.add_argument("--index_file", type=str, default="")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--train_center_frame_step",
        type=int,
        default=1,
        help="Center-frame subsampling step used only for train split.",
    )
    parser.add_argument(
        "--eval_center_frame_step",
        type=int,
        default=1,
        help="Center-frame subsampling step used for val/test/all splits.",
    )
    parser.add_argument("--strict_frame_loading", action="store_true", default=True)
    parser.add_argument(
        "--no_strict_frame_loading",
        action="store_false",
        dest="strict_frame_loading",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")

    # Model checkpoint
    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        default="./log/stage2b_phase2/best.pth",
        help="Stage2 checkpoint path containing trained model weights",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="",
        help="Optional Stage1 checkpoint. If empty, try reading from Stage2 checkpoint args.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Optional Stage3 checkpoint used to warm start V3 (for example V2 symbolize no-reasoner best.pth).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--head_hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Symbolize
    parser.add_argument("--symbol_k", type=int, default=64)
    parser.add_argument("--symbol_dim", type=int, default=64)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.3)
    parser.add_argument(
        "--tau_anneal_epochs",
        type=int,
        default=0,
        help="Number of epochs for linear tau annealing; 0 means use all epochs.",
    )
    parser.add_argument("--lambda_conf", type=float, default=1e-2)
    parser.add_argument("--lambda_bal", type=float, default=2e-2)
    parser.add_argument("--disable_symbol_regularization", action="store_true", default=False)

    # Reasoner
    parser.add_argument("--reason_steps", type=int, default=2)
    parser.add_argument("--lr_reasoner", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lambda_sparse", type=float, default=1e-4)
    parser.add_argument("--lambda_iter", type=float, default=0.0)
    parser.add_argument("--reasoner_init_std", type=float, default=1e-3)
    parser.add_argument("--freeze_reasoner", action="store_true", default=False)
    parser.add_argument("--relation_sparsity_threshold", type=float, default=1e-3)

    # Optional staged training
    parser.add_argument("--phase_a_epochs", type=int, default=0)
    parser.add_argument("--phase_b_epochs", type=int, default=0)
    parser.add_argument("--phase_c_epochs", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="Learning-rate scheduler for optimizer.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing used in cross entropy (0~1).",
    )
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.0,
        help="Power for inverse-frequency class weighting.",
    )
    parser.add_argument(
        "--material_class_weight_power",
        type=float,
        default=0.35,
        help="Override class_weight_power for material. Use negative to inherit the global value.",
    )
    parser.add_argument(
        "--hardness_class_weight_power",
        type=float,
        default=0.0,
        help="Override class_weight_power for hardness. Use negative to inherit the global value.",
    )
    parser.add_argument(
        "--roughness_class_weight_power",
        type=float,
        default=0.0,
        help="Override class_weight_power for roughness. Use negative to inherit the global value.",
    )
    parser.add_argument(
        "--use_class_weighted_ce",
        action="store_true",
        dest="use_class_weighted_ce",
        help="Enable class-weighted CE for all three tasks.",
    )
    parser.add_argument(
        "--no_class_weighted_ce",
        action="store_false",
        dest="use_class_weighted_ce",
        help="Disable class-weighted CE.",
    )
    parser.set_defaults(use_class_weighted_ce=True)
    parser.add_argument("--loss_weight_material", type=float, default=1.0)
    parser.add_argument("--loss_weight_hardness", type=float, default=1.0)
    parser.add_argument("--loss_weight_roughness", type=float, default=1.0)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="mean_macro_f1",
        choices=["mean_macro_f1", "mean_accuracy"],
        help="Metric used to select best checkpoint.",
    )
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument(
        "--log_freq",
        type=int,
        default=50,
        help="Print train progress every N batches (0 to disable batch logs).",
    )
    parser.add_argument(
        "--val_log_freq",
        type=int,
        default=20,
        help="Print val progress every N batches (0 to disable batch logs).",
    )
    parser.add_argument("--amp", action="store_true", default=False)

    # Export
    parser.add_argument("--disable_reason_trace_export", action="store_true", default=False)
    parser.add_argument(
        "--reason_trace_max_samples",
        type=int,
        default=0,
        help="Max samples to export into reason_trace.jsonl; 0 means all validation samples.",
    )
    parser.add_argument("--disable_relation_heatmap_export", action="store_true", default=False)

    # Optional ablation bookkeeping
    parser.add_argument("--ablation_no_reasoner_run_dir", type=str, default="")
    parser.add_argument("--ablation_frozen_reasoner_run_dir", type=str, default="")
    parser.add_argument("--ablation_reasoner_on_run_dir", type=str, default="")

    # Output
    parser.add_argument(
        "--output_root",
        type=str,
        default="./Stage3_method_w/log",
        help="Checkpoint directories will be saved under this root.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name. If empty, generated from timestamp.",
    )

    # Wandb
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="stage3")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_tags", nargs="+", default=None)

    args = parser.parse_args()

    if args.index_file == "":
        args.index_file = str(Path(args.data_root) / "index.csv")
    if args.run_name == "":
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"stage3_w_v3_symbolize_with_reasoner_{stamp}"
    if args.wandb_name == "":
        args.wandb_name = args.run_name
    if args.tau_anneal_epochs <= 0:
        args.tau_anneal_epochs = args.epochs
    if args.disable_symbol_regularization:
        args.lambda_conf = 0.0
        args.lambda_bal = 0.0

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}"
        )
    if args.clip_len != 4:
        raise ValueError(
            f"This minimal closure is fixed to clip_len=4, got {args.clip_len}"
        )
    if args.frame_stride != 1:
        raise ValueError(
            f"This minimal closure is fixed to frame_stride=1, got {args.frame_stride}"
        )
    if args.train_center_frame_step < 1:
        raise ValueError(
            f"train_center_frame_step must be >= 1, got {args.train_center_frame_step}"
        )
    if args.eval_center_frame_step < 1:
        raise ValueError(
            f"eval_center_frame_step must be >= 1, got {args.eval_center_frame_step}"
        )
    if args.lr_head <= 0.0:
        raise ValueError(f"lr_head must be > 0, got {args.lr_head}")
    if args.lr_reasoner <= 0.0:
        raise ValueError(f"lr_reasoner must be > 0, got {args.lr_reasoner}")
    if args.lr_backbone <= 0.0:
        raise ValueError(f"lr_backbone must be > 0, got {args.lr_backbone}")
    if args.min_lr < 0.0:
        raise ValueError(f"min_lr must be >= 0, got {args.min_lr}")
    if args.min_lr > min(args.lr_head, args.lr_reasoner, args.lr_backbone):
        raise ValueError(
            "min_lr must be <= min(lr_head, lr_reasoner, lr_backbone), got "
            f"min_lr={args.min_lr}, lr_head={args.lr_head}, "
            f"lr_reasoner={args.lr_reasoner}, lr_backbone={args.lr_backbone}"
        )
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {args.label_smoothing}")
    if args.class_weight_power < 0.0:
        raise ValueError(f"class_weight_power must be >= 0, got {args.class_weight_power}")
    for key in (
        "material_class_weight_power",
        "hardness_class_weight_power",
        "roughness_class_weight_power",
    ):
        value = float(getattr(args, key))
        if value < 0.0:
            setattr(args, key, float(args.class_weight_power))
    for key in ("loss_weight_material", "loss_weight_hardness", "loss_weight_roughness"):
        value = float(getattr(args, key))
        if value <= 0.0:
            raise ValueError(f"{key} must be > 0, got {value}")
    if args.symbol_k < 2:
        raise ValueError(f"symbol_k must be >= 2, got {args.symbol_k}")
    if args.symbol_dim < 1:
        raise ValueError(f"symbol_dim must be >= 1, got {args.symbol_dim}")
    if args.tau_start <= 0 or args.tau_end <= 0:
        raise ValueError(
            f"tau_start and tau_end must be > 0, got {args.tau_start}, {args.tau_end}"
        )
    if args.tau_anneal_epochs < 1:
        raise ValueError(f"tau_anneal_epochs must be >= 1, got {args.tau_anneal_epochs}")
    if args.lambda_conf < 0 or args.lambda_bal < 0:
        raise ValueError(
            f"lambda_conf/lambda_bal must be >= 0, got {args.lambda_conf}, {args.lambda_bal}"
        )
    if args.reason_steps < 1:
        raise ValueError(f"reason_steps must be >= 1, got {args.reason_steps}")
    if args.lambda_sparse < 0.0 or args.lambda_iter < 0.0:
        raise ValueError(
            f"lambda_sparse/lambda_iter must be >= 0, got {args.lambda_sparse}, {args.lambda_iter}"
        )
    if args.reasoner_init_std < 0.0:
        raise ValueError(f"reasoner_init_std must be >= 0, got {args.reasoner_init_std}")
    if args.relation_sparsity_threshold < 0.0:
        raise ValueError(
            "relation_sparsity_threshold must be >= 0, got "
            f"{args.relation_sparsity_threshold}"
        )
    for key in ("phase_a_epochs", "phase_b_epochs", "phase_c_epochs"):
        value = int(getattr(args, key))
        if value < 0:
            raise ValueError(f"{key} must be >= 0, got {value}")
    phase_total = int(args.phase_a_epochs) + int(args.phase_b_epochs) + int(args.phase_c_epochs)
    if phase_total not in (0, int(args.epochs)):
        raise ValueError(
            "When staged training is enabled, phase_a_epochs + phase_b_epochs + phase_c_epochs "
            f"must equal epochs. Got phases={phase_total}, epochs={args.epochs}"
        )
    if args.reason_trace_max_samples < 0:
        raise ValueError(
            f"reason_trace_max_samples must be >= 0, got {args.reason_trace_max_samples}"
        )
    if args.init_checkpoint != "" and (not Path(args.init_checkpoint).exists()):
        raise FileNotFoundError(f"init_checkpoint not found: {args.init_checkpoint}")
    for key in (
        "ablation_no_reasoner_run_dir",
        "ablation_frozen_reasoner_run_dir",
        "ablation_reasoner_on_run_dir",
    ):
        value = str(getattr(args, key))
        if value != "" and (not Path(value).exists()):
            raise FileNotFoundError(f"{key} not found: {value}")

    return args
