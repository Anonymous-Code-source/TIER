#!/bin/bash

# ============================================================================
# Stage-2b Phase-2 两阶段训练脚本（低泄漏 + 高语义）
# ============================================================================
# Phase A (semantic stabilize): 先稳住接触语义与跨传感器配对
# Phase B (domain squeeze):     再增强去域强度，压低独立域探针准确率
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p ./log
LOG_FILE="./log/stage2b_phase2_fixed.log"
: > "${LOG_FILE}"
export PYTHONUNBUFFERED=1
export TORCHELASTIC_ERROR_FILE="./log/stage2b_phase2_fixed.err"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

if ! command -v python >/dev/null 2>&1; then
    echo "❌ python not found in PATH. Please activate your environment first."
    exit 127
fi

if ! python - <<'PY'
import sys
try:
    import torch  # noqa: F401
except Exception as e:
    print(f"❌ Failed to import torch: {e}")
    sys.exit(1)
PY
then
    echo "❌ Torch runtime is unavailable. Please activate the training conda env."
    exit 127
fi

if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a _GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_GPU_ARRAY[@]}"
fi
MAX_LOCAL_PROCS="$(python - <<'PY'
import torch
n = torch.cuda.device_count()
print(max(1, int(n)))
PY
)"
if [ "${NPROC_PER_NODE}" -gt "${MAX_LOCAL_PROCS}" ]; then
    echo "⚠️  NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible local CUDA count=${MAX_LOCAL_PROCS}, clamp to ${MAX_LOCAL_PROCS}"
    NPROC_PER_NODE="${MAX_LOCAL_PROCS}"
fi

MASTER_PORT="${MASTER_PORT:-}"
if [ -z "${MASTER_PORT}" ]; then
    MASTER_PORT="$(python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi

OUTPUT_DIR="./log/stage2b_phase2"
PHASEA_DIR="${OUTPUT_DIR}/phaseA"
PHASEB_DIR="${OUTPUT_DIR}"

mkdir -p "${PHASEA_DIR}" "${PHASEB_DIR}"
RUN_START_TS="$(date +%s)"

pick_checkpoint() {
    local ckpt_dir="$1"
    local min_mtime="${2:-0}"
    local file=""
    local name=""
    for name in best_semantic.pth best_pareto.pth best.pth best_any.pth last.pth; do
        file="${ckpt_dir}/${name}"
        if [ -f "${file}" ]; then
            if [ "$(stat -c %Y "${file}")" -ge "${min_mtime}" ]; then
                echo "${file}"
                return 0
            fi
        fi
    done
    if [ "${min_mtime}" -gt 0 ]; then
        return 1
    fi
    for name in best_semantic.pth best_pareto.pth best.pth best_any.pth last.pth; do
        file="${ckpt_dir}/${name}"
        if [ -f "${file}" ]; then
            echo "${file}"
            return 0
        fi
    done
    return 1
}

run_phase() {
    local phase_name="$1"
    shift
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Start ${phase_name}"
    echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "   NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "   MASTER_PORT=${MASTER_PORT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" Stage2_main.py "$@"
    return $?
}

COMMON_ARGS=(
    --stage 2b
    --primary_metric phase2_combined
    --data_sources Fine Indoor Outdoor
    --data_root_fine ./Tacquad/
    --data_root_indoor ./Tacquad/
    --data_root_outdoor ./Tacquad/
    --batch_size 96
    --accum_iter 1
    --freeze_blocks 6
    --domain_projection_dim 128
    --temperature 0.05
    --stage2_num_frames 3
    --lr 5e-5
    --lr_encoder 3e-5
    --lr_discriminator 2e-4
    --discriminator_hidden_dim 512
    --discriminator_dropout 0.0
    --min_lr 1e-6
    --adversarial_feature_target contact
    --disc_update_every 1
    --use_adversarial_training
    --use_independent_probe
    --use_linear_probe
    --prioritize_same_object_diff_frame_negatives
    --same_object_negative_keep_ratio 0.60
    --lambda_match 0.0
    --pair_loss_type cosine
    --inter_sim_margin 0.15
    --inter_hard_topk 512
    --pair_confidence_floor 0.20
    --pair_confidence_power 2.0
    --pair_confidence_min 0.25
    --use_pair_confidence_for_domain_and_inter
    --same_sim_threshold 0.72
    --diff_sim_threshold 0.35
    --probe_train_epochs 10
    --probe_lr 5e-4
    --group_dro_eta 0.0
    --group_balance_strategy repeat
    --contact_sampler_enabled
    --contact_sampler_max_per_object_per_round 2
    --warmup_epochs 5
    --weight_decay 0.02
    --num_workers 4
    --prefetch_factor 1
    --no_pin_mem
    --train_split_ratio 0.9
    --seed 42
    --random_seed 42
    --save_freq 5
    --eval_freq 1
    --wandb_project PhysioTouch-Stage2b
)

# ----------------------------------------------------------------------------
# Phase A: 语义稳态（弱去域）
# ----------------------------------------------------------------------------
PHASEA_ARGS=(
    --epochs 100
    --resume ./log/stage2b_phase1/best.pth
    --resume_model_only
    --output_dir "${PHASEA_DIR}"
    --log_dir "${PHASEA_DIR}"
    --wandb_name stage2b_phase2_phaseA_semantic
    --early_stopping_patience 100
    --lambda_im_start 0.00
    --lambda_im_end 0.10
    --lambda_im_warmup_ratio 0.0
    --lambda_im_rampup_ratio 0.20
    --phase2_lambda_im_boost 1.10
    --phase2_lambda_im_max 0.15
    --phase2_domain_acc_patience 4
    --phase2_accel_factor 1.0
    --phase2_accel_warmup_min 0.0
    --phase2_accel_rampup_min 0.20
    --phase2_accel_patience 999
    --disc_steps_per_iter 2
    --im_loss_scale 2.0
    --lambda_pair 0.16
    --lambda_inter 0.04
    --lambda_inter_sensor 0.02
    --inter_hard_topk 256
    --pair_confidence_min 0.50
    --lambda_decouple 0.010
    --lambda_domain_cls 0.00
    --lambda_coral 0.002
    --lambda_coral_residual 0.00
    --coral_include_mean
    --no_coral_residual_include_mean
    --phase2_same_sim_threshold 0.80
    --phase2_probe_acc_threshold 0.42
    --phase2_conditional_probe_threshold 0.48
    --phase2_pair_link_p90_threshold 0.40
    --phase2_semantic_guardrail
    --phase2_semantic_margin 0.01
    --phase2_guardrail_lambda_decay 0.80
    --phase2_guardrail_im_scale_decay 0.85
    --phase2_guardrail_lambda_min 0.12
    --phase2_guardrail_im_scale_min 1.5
    --phase2_guardrail_pair_step 0.01
    --phase2_guardrail_pair_max 0.20
    --phase2_guardrail_domain_boost 1.03
)

PHASEA_START_TS="$(date +%s)"
run_phase "Phase A (semantic stabilize)" "${COMMON_ARGS[@]}" "${PHASEA_ARGS[@]}"
PHASEA_EXIT_CODE=$?
if [ ${PHASEA_EXIT_CODE} -ne 0 ]; then
    echo "❌ Phase A failed (exit=${PHASEA_EXIT_CODE})"
    exit ${PHASEA_EXIT_CODE}
fi

PHASEA_CKPT="$(pick_checkpoint "${PHASEA_DIR}" "${PHASEA_START_TS}")"
if [ -z "${PHASEA_CKPT}" ]; then
    echo "❌ Phase A finished but no checkpoint found in ${PHASEA_DIR}"
    exit 2
fi
echo "✅ Phase A checkpoint selected: ${PHASEA_CKPT}"

# ----------------------------------------------------------------------------
# Phase B: 去域压缩（强去域 + 语义护栏）
# ----------------------------------------------------------------------------
PHASEB_ARGS=(
    --epochs 100
    --resume "${PHASEA_CKPT}"
    --resume_model_only
    --output_dir "${PHASEB_DIR}"
    --log_dir "${PHASEB_DIR}"
    --wandb_name stage2b_phase2_phaseB_domain
    --early_stopping_patience 100
    --lambda_im_start 0.05
    --lambda_im_end 0.40
    --lambda_im_warmup_ratio 0.0
    --lambda_im_rampup_ratio 0.15
    --phase2_lambda_im_boost 1.15
    --phase2_lambda_im_max 0.55
    --phase2_domain_acc_patience 3
    --phase2_accel_factor 1.0
    --phase2_accel_warmup_min 0.0
    --phase2_accel_rampup_min 0.15
    --phase2_accel_patience 999
    --disc_steps_per_iter 2
    --im_loss_scale 3.0
    --im_dual_target
    --im_contact_ratio_start 0.70
    --im_contact_ratio_end 0.35
    --lambda_pair 0.22
    --lambda_pair_tail 0.10
    --pair_tail_ratio 0.20
    --pair_tail_min_samples 24
    --lambda_inter 0.14
    --lambda_inter_sensor 0.12
    --lambda_residual_compact 0.08
    --inter_hard_topk 512
    --pair_confidence_power 1.2
    --pair_confidence_min 0.35
    --lambda_decouple 0.008
    --lambda_domain_cls 0.10
    --lambda_coral 0.003
    --lambda_coral_residual 0.12
    --use_digit_focus_reweight
    --digit_focus_weight 1.2
    --digit_focus_margin 0.05
    --digit_focus_step 0.10
    --digit_focus_max 2.5
    --digit_focus_decay 0.98
    --adversarial_feature_target contact
    --phase2_adv_target_switch
    --phase2_adv_target_switch_ratio 0.55
    --phase2_adv_target_late contact_residual
    --enable_dynamic_loss_balance
    --loss_balance_ema_decay 0.98
    --loss_balance_min_factor 0.25
    --loss_balance_max_factor 6.0
    --loss_balance_warmup_steps 30
    --coral_include_mean
    --coral_residual_include_mean
    --phase2_same_sim_threshold 0.80
    --phase2_probe_acc_threshold 0.38
    --phase2_use_conditional_constraints
    --phase2_conditional_probe_threshold 0.42
    --phase2_conditional_probe_sensor_threshold 0.42
    --phase2_conditional_probe_gelsight_threshold 0.42
    --phase2_conditional_probe_digit_threshold 0.38
    --phase2_conditional_probe_duragel_threshold 0.42
    --phase2_pair_link_p90_threshold 0.35
    --phase2_use_global_mix_constraints
    --phase2_sensor_centroid_dist_threshold 0.45
    --phase2_knn_mix_rate_threshold 0.55
    --global_mix_knn_k 20
    --global_mix_knn_max_samples_per_sensor 512
    --phase2_require_best
    --phase2_semantic_guardrail
    --phase2_semantic_margin 0.015
    --phase2_guardrail_lambda_decay 0.85
    --phase2_guardrail_im_scale_decay 0.90
    --phase2_guardrail_lambda_min 0.20
    --phase2_guardrail_im_scale_min 2.0
    --phase2_guardrail_pair_step 0.02
    --phase2_guardrail_pair_max 0.32
    --phase2_guardrail_pair_tail_step 0.01
    --phase2_guardrail_pair_tail_max 0.20
    --phase2_guardrail_inter_step 0.02
    --phase2_guardrail_inter_max 0.30
    --phase2_guardrail_residual_compact_step 0.01
    --phase2_guardrail_residual_compact_max 0.18
    --phase2_guardrail_conditional_restore_decay 0.98
    --phase2_guardrail_domain_boost 1.03
    --phase2_guardrail_escalate_patience 3
    --phase2_guardrail_restore_patience 3
    --phase2_guardrail_im_scale_step 0.5
    --phase2_guardrail_im_scale_cap 8.0
    --phase2_guardrail_lambda_boost 1.08
    --phase2_guardrail_lambda_cap 0.70
)

PHASEB_START_TS="$(date +%s)"
run_phase "Phase B (domain squeeze)" "${COMMON_ARGS[@]}" "${PHASEB_ARGS[@]}"
PHASEB_EXIT_CODE=$?
if [ ${PHASEB_EXIT_CODE} -ne 0 ]; then
    echo "❌ Phase B failed (exit=${PHASEB_EXIT_CODE})"
    exit ${PHASEB_EXIT_CODE}
fi

REQUIRE_CONSTRAINED_BEST="${PHASE2_REQUIRE_BEST:-0}"
FINAL_CKPT="${PHASEB_DIR}/best.pth"
if [ ! -f "${FINAL_CKPT}" ] || [ "$(stat -c %Y "${FINAL_CKPT}")" -lt "${PHASEB_START_TS}" ]; then
    if [ "${REQUIRE_CONSTRAINED_BEST}" = "1" ]; then
        echo "❌ Phase B finished but constrained best checkpoint missing in this run: ${FINAL_CKPT}"
        echo "   Set PHASE2_REQUIRE_BEST=0 to allow fallback to best_any/best_pareto/last."
        exit 3
    fi
    FINAL_CKPT="$(pick_checkpoint "${PHASEB_DIR}" "${PHASEB_START_TS}")"
    if [ -z "${FINAL_CKPT}" ]; then
        echo "❌ Phase B finished but no new checkpoint found in ${PHASEB_DIR}"
        exit 3
    fi
    echo "⚠️  Constrained best not found in this run; fallback checkpoint: ${FINAL_CKPT}"
fi
echo "✅ Final checkpoint selected: ${FINAL_CKPT}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Running cross-sensor retrieval..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
(
    cd Testing_part
    python test_cross_sensor_retrieval.py \
        --model_path "../${FINAL_CKPT#./}" \
        --stage1_checkpoint ../log/joint_alpha0.75_beta0.25/best.pth \
        --data_path ../Tacquad/ \
        --output_dir ../log/stage2b_phase2
)
RETRIEVAL_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ${RETRIEVAL_EXIT_CODE} -eq 0 ]; then
    echo "✅ Training + retrieval finished"
    echo "📄 Log: ${LOG_FILE}"
    echo "📦 Final checkpoint: ${FINAL_CKPT}"
else
    echo "⚠️  Retrieval failed (exit=${RETRIEVAL_EXIT_CODE})"
    echo "📄 Training log still available: ${LOG_FILE}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
