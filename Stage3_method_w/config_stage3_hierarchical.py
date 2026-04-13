#!/usr/bin/env python3
"""
Configuration for hierarchical Stage3 training.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage3_method_w hierarchical Stage3 training on Causal_data"
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
    parser.add_argument("--train_center_frame_step", type=int, default=1)
    parser.add_argument("--eval_center_frame_step", type=int, default=1)
    parser.add_argument("--strict_frame_loading", action="store_true", default=True)
    parser.add_argument("--no_strict_frame_loading", action="store_false", dest="strict_frame_loading")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument(
        "--force_num_bins",
        type=int,
        default=3,
        help="Deprecated compatibility flag. Force is trained as continuous regression.",
    )

    # Checkpoints
    parser.add_argument("--stage2_checkpoint", type=str, default="./log/stage2b_phase2/best.pth")
    parser.add_argument("--stage1_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--freeze_reasoners", action="store_true", default=False)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--obs_symbol_k", type=int, default=64)
    parser.add_argument("--obs_symbol_dim", type=int, default=64)
    parser.add_argument("--physical_symbol_k", type=int, default=32)
    parser.add_argument("--physical_symbol_dim", type=int, default=64)
    parser.add_argument("--interaction_symbol_k", type=int, default=16)
    parser.add_argument("--interaction_symbol_dim", type=int, default=64)
    parser.add_argument("--contact_symbol_k", type=int, default=32)
    parser.add_argument("--contact_symbol_dim", type=int, default=64)
    parser.add_argument("--reason_steps_physical", type=int, default=2)
    parser.add_argument("--reason_steps_contact", type=int, default=2)
    parser.add_argument("--reasoner_init_std", type=float, default=1e-3)

    # Symbol regularization
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.3)
    parser.add_argument("--tau_anneal_epochs", type=int, default=0)
    parser.add_argument("--lambda_conf", type=float, default=1e-2)
    parser.add_argument("--lambda_bal", type=float, default=2e-2)
    parser.add_argument("--lambda_sparse", type=float, default=1e-4)
    parser.add_argument("--lambda_iter", type=float, default=0.0)
    parser.add_argument("--relation_sparsity_threshold", type=float, default=1e-3)

    # Training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--lr_reasoner", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight_power", type=float, default=0.0)
    parser.add_argument("--use_class_weighted_ce", action="store_true", dest="use_class_weighted_ce")
    parser.add_argument("--no_class_weighted_ce", action="store_false", dest="use_class_weighted_ce")
    parser.set_defaults(use_class_weighted_ce=True)

    # Layer weighting
    parser.add_argument("--layer_loss_weight_physical", type=float, default=1.0)
    parser.add_argument("--layer_loss_weight_interaction", type=float, default=1.0)
    parser.add_argument("--layer_loss_weight_contact", type=float, default=1.0)

    # Optional staged training
    parser.add_argument("--phase_p_epochs", type=int, default=5)
    parser.add_argument("--phase_pi_epochs", type=int, default=10)
    parser.add_argument("--phase_pic_epochs", type=int, default=10)
    parser.add_argument("--phase_joint_epochs", type=int, default=15)

    # Logging / selection
    parser.add_argument("--best_metric", type=str, default="mean_layer_macro_f1", choices=["mean_layer_macro_f1", "mean_layer_accuracy"])
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--val_log_freq", type=int, default=20)
    parser.add_argument("--amp", action="store_true", default=False)

    # Output
    parser.add_argument("--output_root", type=str, default="./Stage3_method_w/log_hierarchical")
    parser.add_argument("--run_name", type=str, default="")

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
        args.run_name = f"stage3_w_hierarchical_{stamp}"
    if args.wandb_name == "":
        args.wandb_name = args.run_name
    if args.tau_anneal_epochs <= 0:
        args.tau_anneal_epochs = args.epochs

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")
    if args.clip_len != 4:
        raise ValueError(f"Hierarchical Stage3 is fixed to clip_len=4, got {args.clip_len}")
    if args.frame_stride != 1:
        raise ValueError(f"Hierarchical Stage3 is fixed to frame_stride=1, got {args.frame_stride}")
    if args.force_num_bins < 1:
        raise ValueError(f"force_num_bins must be >= 1, got {args.force_num_bins}")
    if args.init_checkpoint != "" and (not Path(args.init_checkpoint).exists()):
        raise FileNotFoundError(f"init_checkpoint not found: {args.init_checkpoint}")
    if not Path(args.stage2_checkpoint).exists():
        raise FileNotFoundError(f"stage2_checkpoint not found: {args.stage2_checkpoint}")

    phase_total = (
        int(args.phase_p_epochs)
        + int(args.phase_pi_epochs)
        + int(args.phase_pic_epochs)
        + int(args.phase_joint_epochs)
    )
    if phase_total > 0 and phase_total != int(args.epochs):
        raise ValueError(
            "When staged training is enabled, "
            "phase_p_epochs + phase_pi_epochs + phase_pic_epochs + phase_joint_epochs "
            f"must equal epochs. Got phases={phase_total}, epochs={args.epochs}"
        )
    return args


__all__ = ["parse_args"]
