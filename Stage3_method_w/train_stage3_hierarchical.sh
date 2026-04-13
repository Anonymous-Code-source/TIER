#!/usr/bin/env bash
set -euo pipefail

# Stage3_method_w hierarchical Stage3:
#   Stage2 backbone + s_x + s^p + s^i + s^c
#
# Usage:
#   bash Stage3_method_w/train_stage3_hierarchical.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-./Causal_data}"
INDEX_FILE="${INDEX_FILE:-$DATA_ROOT/index.csv}"
STAGE2_CKPT="${STAGE2_CKPT:-./log/stage2b_phase2/best.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-stage3_w_hierarchical_causal_ddp8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./Stage3_method_w/log_hierarchical}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29623}"
LOG_FREQ="${LOG_FREQ:-50}"
VAL_LOG_FREQ="${VAL_LOG_FREQ:-20}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
LR_HEAD="${LR_HEAD:-5e-4}"
LR_REASONER="${LR_REASONER:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR="${MIN_LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.0}"
USE_CLASS_WEIGHTED_CE="${USE_CLASS_WEIGHTED_CE:-1}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
FREEZE_REASONERS="${FREEZE_REASONERS:-0}"
TRAIN_CENTER_FRAME_STEP="${TRAIN_CENTER_FRAME_STEP:-1}"
EVAL_CENTER_FRAME_STEP="${EVAL_CENTER_FRAME_STEP:-1}"
FORCE_NUM_BINS="${FORCE_NUM_BINS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-768}"
DROPOUT="${DROPOUT:-0.1}"
OBS_SYMBOL_K="${OBS_SYMBOL_K:-64}"
OBS_SYMBOL_DIM="${OBS_SYMBOL_DIM:-64}"
PHYSICAL_SYMBOL_K="${PHYSICAL_SYMBOL_K:-32}"
PHYSICAL_SYMBOL_DIM="${PHYSICAL_SYMBOL_DIM:-64}"
INTERACTION_SYMBOL_K="${INTERACTION_SYMBOL_K:-16}"
INTERACTION_SYMBOL_DIM="${INTERACTION_SYMBOL_DIM:-64}"
CONTACT_SYMBOL_K="${CONTACT_SYMBOL_K:-32}"
CONTACT_SYMBOL_DIM="${CONTACT_SYMBOL_DIM:-64}"
TAU_START="${TAU_START:-1.0}"
TAU_END="${TAU_END:-0.3}"
TAU_ANNEAL_EPOCHS="${TAU_ANNEAL_EPOCHS:-$EPOCHS}"
LAMBDA_CONF="${LAMBDA_CONF:-1e-2}"
LAMBDA_BAL="${LAMBDA_BAL:-2e-2}"
LAMBDA_SPARSE="${LAMBDA_SPARSE:-1e-4}"
LAMBDA_ITER="${LAMBDA_ITER:-0.0}"
REASON_STEPS_PHYSICAL="${REASON_STEPS_PHYSICAL:-2}"
REASON_STEPS_CONTACT="${REASON_STEPS_CONTACT:-2}"
REASONER_INIT_STD="${REASONER_INIT_STD:-1e-3}"
RELATION_SPARSITY_THRESHOLD="${RELATION_SPARSITY_THRESHOLD:-1e-3}"
LAYER_LOSS_WEIGHT_PHYSICAL="${LAYER_LOSS_WEIGHT_PHYSICAL:-1.0}"
LAYER_LOSS_WEIGHT_INTERACTION="${LAYER_LOSS_WEIGHT_INTERACTION:-1.0}"
LAYER_LOSS_WEIGHT_CONTACT="${LAYER_LOSS_WEIGHT_CONTACT:-1.0}"

if [[ -z "${PHASE_P_EPOCHS+x}" && -z "${PHASE_PI_EPOCHS+x}" && -z "${PHASE_PIC_EPOCHS+x}" && -z "${PHASE_JOINT_EPOCHS+x}" ]]; then
  if (( EPOCHS < 4 )); then
    PHASE_P_EPOCHS="${EPOCHS}"
    PHASE_PI_EPOCHS="0"
    PHASE_PIC_EPOCHS="0"
    PHASE_JOINT_EPOCHS="0"
  else
    PHASE_P_EPOCHS="$(( EPOCHS * 5 / 40 ))"
    PHASE_PI_EPOCHS="$(( EPOCHS * 10 / 40 ))"
    PHASE_PIC_EPOCHS="$(( EPOCHS * 10 / 40 ))"
    if (( PHASE_P_EPOCHS < 1 )); then PHASE_P_EPOCHS=1; fi
    if (( PHASE_PI_EPOCHS < 1 )); then PHASE_PI_EPOCHS=1; fi
    if (( PHASE_PIC_EPOCHS < 1 )); then PHASE_PIC_EPOCHS=1; fi
    PHASE_JOINT_EPOCHS="$(( EPOCHS - PHASE_P_EPOCHS - PHASE_PI_EPOCHS - PHASE_PIC_EPOCHS ))"
  fi
else
  PHASE_P_EPOCHS="${PHASE_P_EPOCHS:-5}"
  PHASE_PI_EPOCHS="${PHASE_PI_EPOCHS:-10}"
  PHASE_PIC_EPOCHS="${PHASE_PIC_EPOCHS:-10}"
  PHASE_JOINT_EPOCHS="${PHASE_JOINT_EPOCHS:-15}"
fi

if (( PHASE_JOINT_EPOCHS < 0 )); then
  echo "Invalid phase schedule: P=${PHASE_P_EPOCHS}, PI=${PHASE_PI_EPOCHS}, PIC=${PHASE_PIC_EPOCHS}, JOINT=${PHASE_JOINT_EPOCHS}, EPOCHS=${EPOCHS}" >&2
  exit 1
fi

BEST_METRIC="${BEST_METRIC:-mean_layer_macro_f1}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE="${OUTPUT_ROOT}/${RUN_NAME}.err"
exec > >(tee -a "${OUTPUT_ROOT}/${RUN_NAME}.log") 2>&1

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
  --force_num_bins "$FORCE_NUM_BINS"
  --hidden_dim "$HIDDEN_DIM"
  --dropout "$DROPOUT"
  --lr_head "$LR_HEAD"
  --lr_reasoner "$LR_REASONER"
  --lr_backbone "$LR_BACKBONE"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --lr_scheduler "$LR_SCHEDULER"
  --label_smoothing "$LABEL_SMOOTHING"
  --class_weight_power "$CLASS_WEIGHT_POWER"
  --best_metric "$BEST_METRIC"
  --eval_freq 1
  --save_freq 1
  --log_freq "$LOG_FREQ"
  --val_log_freq "$VAL_LOG_FREQ"
  --obs_symbol_k "$OBS_SYMBOL_K"
  --obs_symbol_dim "$OBS_SYMBOL_DIM"
  --physical_symbol_k "$PHYSICAL_SYMBOL_K"
  --physical_symbol_dim "$PHYSICAL_SYMBOL_DIM"
  --interaction_symbol_k "$INTERACTION_SYMBOL_K"
  --interaction_symbol_dim "$INTERACTION_SYMBOL_DIM"
  --contact_symbol_k "$CONTACT_SYMBOL_K"
  --contact_symbol_dim "$CONTACT_SYMBOL_DIM"
  --tau_start "$TAU_START"
  --tau_end "$TAU_END"
  --tau_anneal_epochs "$TAU_ANNEAL_EPOCHS"
  --lambda_conf "$LAMBDA_CONF"
  --lambda_bal "$LAMBDA_BAL"
  --lambda_sparse "$LAMBDA_SPARSE"
  --lambda_iter "$LAMBDA_ITER"
  --reason_steps_physical "$REASON_STEPS_PHYSICAL"
  --reason_steps_contact "$REASON_STEPS_CONTACT"
  --reasoner_init_std "$REASONER_INIT_STD"
  --relation_sparsity_threshold "$RELATION_SPARSITY_THRESHOLD"
  --layer_loss_weight_physical "$LAYER_LOSS_WEIGHT_PHYSICAL"
  --layer_loss_weight_interaction "$LAYER_LOSS_WEIGHT_INTERACTION"
  --layer_loss_weight_contact "$LAYER_LOSS_WEIGHT_CONTACT"
  --phase_p_epochs "$PHASE_P_EPOCHS"
  --phase_pi_epochs "$PHASE_PI_EPOCHS"
  --phase_pic_epochs "$PHASE_PIC_EPOCHS"
  --phase_joint_epochs "$PHASE_JOINT_EPOCHS"
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

if [[ "$FREEZE_BACKBONE" == "0" ]]; then
  CMD_ARGS+=(--no_freeze_backbone)
fi

if [[ "$FREEZE_REASONERS" == "1" ]]; then
  CMD_ARGS+=(--freeze_reasoners)
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    Stage3_method_w/main_stage3_hierarchical.py \
    "${CMD_ARGS[@]}"
else
  python Stage3_method_w/main_stage3_hierarchical.py \
    "${CMD_ARGS[@]}"
fi
