#!/usr/bin/env bash
set -euo pipefail

# Stage3_method_w V3:
#   Stage2 backbone + Symbolize + 3-classification heads + pairwise Reasoner
#
# Usage:
#   bash Stage3_method_w/train_stage3_v3_symbolize_with_reasoner.sh
# Default behavior:
#   Runs the Causal_data experiment with:
#     - no Stage3 warm start by default (class spaces differ from octopi_new)
#     - best_metric=mean_macro_f1
#     - phase schedule 5/20/15
#   Override any setting via environment variables if needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-./Causal_data}"
INDEX_FILE="${INDEX_FILE:-$DATA_ROOT/index.csv}"
STAGE2_CKPT="${STAGE2_CKPT:-./log/stage2b_phase2/best.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-stage3_w_v3_symbolize_with_reasoner_causal_ddp8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./Stage3_method_w/log}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29613}"
LOG_FREQ="${LOG_FREQ:-50}"
VAL_LOG_FREQ="${VAL_LOG_FREQ:-20}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
DISABLE_SYMBOL_REG="${DISABLE_SYMBOL_REG:-0}"
FREEZE_REASONER="${FREEZE_REASONER:-0}"
EXPORT_REASON_TRACE="${EXPORT_REASON_TRACE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-40}"
LR_HEAD="${LR_HEAD:-5e-4}"
LR_REASONER="${LR_REASONER:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR="${MIN_LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.0}"
MATERIAL_CLASS_WEIGHT_POWER="${MATERIAL_CLASS_WEIGHT_POWER:-0.35}"
HARDNESS_CLASS_WEIGHT_POWER="${HARDNESS_CLASS_WEIGHT_POWER:-0.0}"
ROUGHNESS_CLASS_WEIGHT_POWER="${ROUGHNESS_CLASS_WEIGHT_POWER:-0.0}"
USE_CLASS_WEIGHTED_CE="${USE_CLASS_WEIGHTED_CE:-1}"
LOSS_WEIGHT_MATERIAL="${LOSS_WEIGHT_MATERIAL:-1.0}"
LOSS_WEIGHT_HARDNESS="${LOSS_WEIGHT_HARDNESS:-1.0}"
LOSS_WEIGHT_ROUGHNESS="${LOSS_WEIGHT_ROUGHNESS:-1.0}"
BEST_METRIC="${BEST_METRIC:-mean_macro_f1}"
TRAIN_CENTER_FRAME_STEP="${TRAIN_CENTER_FRAME_STEP:-1}"
EVAL_CENTER_FRAME_STEP="${EVAL_CENTER_FRAME_STEP:-1}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-768}"
DROPOUT="${DROPOUT:-0.1}"
SYMBOL_K="${SYMBOL_K:-64}"
SYMBOL_DIM="${SYMBOL_DIM:-64}"
TAU_START="${TAU_START:-1.0}"
TAU_END="${TAU_END:-0.3}"
TAU_ANNEAL_EPOCHS="${TAU_ANNEAL_EPOCHS:-$EPOCHS}"
LAMBDA_CONF="${LAMBDA_CONF:-1e-2}"
LAMBDA_BAL="${LAMBDA_BAL:-2e-2}"
REASON_STEPS="${REASON_STEPS:-2}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-1e-4}"
LAMBDA_ITER="${LAMBDA_ITER:-0.0}"
REASONER_INIT_STD="${REASONER_INIT_STD:-1e-3}"
RELATION_SPARSITY_THRESHOLD="${RELATION_SPARSITY_THRESHOLD:-1e-3}"
REASON_TRACE_MAX_SAMPLES="${REASON_TRACE_MAX_SAMPLES:-0}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-5}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-20}"
PHASE_C_EPOCHS="${PHASE_C_EPOCHS:-15}"
EXPORT_RELATION_HEATMAP="${EXPORT_RELATION_HEATMAP:-1}"
ABLATION_NO_REASONER_RUN_DIR="${ABLATION_NO_REASONER_RUN_DIR:-}"
ABLATION_FROZEN_REASONER_RUN_DIR="${ABLATION_FROZEN_REASONER_RUN_DIR:-}"
ABLATION_REASONER_ON_RUN_DIR="${ABLATION_REASONER_ON_RUN_DIR:-}"

mkdir -p ./Stage3_method_w/log
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE="./Stage3_method_w/log/${RUN_NAME}.err"
exec > >(tee -a "./Stage3_method_w/log/${RUN_NAME}.log") 2>&1

CMD_ARGS=(
  --data_root "$DATA_ROOT"
  --index_file "$INDEX_FILE"
  --stage2_checkpoint "$STAGE2_CKPT"
  --stage1_checkpoint "$STAGE1_CKPT"
  --device "$DEVICE"
  --run_name "$RUN_NAME"
  --output_root "$OUTPUT_ROOT"
  --wandb_project stage3
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --clip_len 4
  --frame_stride 1
  --train_center_frame_step "$TRAIN_CENTER_FRAME_STEP"
  --eval_center_frame_step "$EVAL_CENTER_FRAME_STEP"
  --head_hidden_dim "$HEAD_HIDDEN_DIM"
  --dropout "$DROPOUT"
  --lr_head "$LR_HEAD"
  --lr_reasoner "$LR_REASONER"
  --lr_backbone "$LR_BACKBONE"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --lr_scheduler "$LR_SCHEDULER"
  --label_smoothing "$LABEL_SMOOTHING"
  --class_weight_power "$CLASS_WEIGHT_POWER"
  --material_class_weight_power "$MATERIAL_CLASS_WEIGHT_POWER"
  --hardness_class_weight_power "$HARDNESS_CLASS_WEIGHT_POWER"
  --roughness_class_weight_power "$ROUGHNESS_CLASS_WEIGHT_POWER"
  --loss_weight_material "$LOSS_WEIGHT_MATERIAL"
  --loss_weight_hardness "$LOSS_WEIGHT_HARDNESS"
  --loss_weight_roughness "$LOSS_WEIGHT_ROUGHNESS"
  --best_metric "$BEST_METRIC"
  --eval_freq 1
  --save_freq 1
  --log_freq "$LOG_FREQ"
  --val_log_freq "$VAL_LOG_FREQ"
  --symbol_k "$SYMBOL_K"
  --symbol_dim "$SYMBOL_DIM"
  --tau_start "$TAU_START"
  --tau_end "$TAU_END"
  --tau_anneal_epochs "$TAU_ANNEAL_EPOCHS"
  --lambda_conf "$LAMBDA_CONF"
  --lambda_bal "$LAMBDA_BAL"
  --reason_steps "$REASON_STEPS"
  --lambda_sparse "$LAMBDA_SPARSE"
  --lambda_iter "$LAMBDA_ITER"
  --reasoner_init_std "$REASONER_INIT_STD"
  --relation_sparsity_threshold "$RELATION_SPARSITY_THRESHOLD"
  --reason_trace_max_samples "$REASON_TRACE_MAX_SAMPLES"
  --phase_a_epochs "$PHASE_A_EPOCHS"
  --phase_b_epochs "$PHASE_B_EPOCHS"
  --phase_c_epochs "$PHASE_C_EPOCHS"
)

if [[ -n "$INIT_CHECKPOINT" ]]; then
  CMD_ARGS+=(--init_checkpoint "$INIT_CHECKPOINT")
fi

if [[ "$DISABLE_WANDB" == "1" ]]; then
  CMD_ARGS+=(--disable_wandb)
fi

if [[ "$USE_CLASS_WEIGHTED_CE" == "0" ]]; then
  CMD_ARGS+=(--no_class_weighted_ce)
fi

if [[ "$DISABLE_SYMBOL_REG" == "1" ]]; then
  CMD_ARGS+=(--disable_symbol_regularization)
fi

if [[ "$FREEZE_REASONER" == "1" ]]; then
  CMD_ARGS+=(--freeze_reasoner)
fi

if [[ "$EXPORT_REASON_TRACE" == "0" ]]; then
  CMD_ARGS+=(--disable_reason_trace_export)
fi

if [[ "$EXPORT_RELATION_HEATMAP" == "0" ]]; then
  CMD_ARGS+=(--disable_relation_heatmap_export)
fi

if [[ -n "$ABLATION_NO_REASONER_RUN_DIR" ]]; then
  CMD_ARGS+=(--ablation_no_reasoner_run_dir "$ABLATION_NO_REASONER_RUN_DIR")
fi

if [[ -n "$ABLATION_FROZEN_REASONER_RUN_DIR" ]]; then
  CMD_ARGS+=(--ablation_frozen_reasoner_run_dir "$ABLATION_FROZEN_REASONER_RUN_DIR")
fi

if [[ -n "$ABLATION_REASONER_ON_RUN_DIR" ]]; then
  CMD_ARGS+=(--ablation_reasoner_on_run_dir "$ABLATION_REASONER_ON_RUN_DIR")
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    Stage3_method_w/main_stage3_v3_symbolize_with_reasoner.py \
    "${CMD_ARGS[@]}"
else
  python Stage3_method_w/main_stage3_v3_symbolize_with_reasoner.py \
    "${CMD_ARGS[@]}"
fi
