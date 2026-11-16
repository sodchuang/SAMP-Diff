#!/usr/bin/env bash
# One-command, one-policy, two-stage Stack_D1 curriculum:
# A learns grasp / transport through epoch 700; B continues the same policy
# with mixed alignment, controlled descent, release, and upward retreat.

set -euo pipefail

BASE_RUN_NAME="${BASE_RUN_NAME:-stack_two_stage_v1_seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
DATASET="${DATASET:-data/mimicgen/core/stack_d1.hdf5}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_MODE="${WANDB_MODE:-disabled}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-30}"
N_TEST_VIS="${N_TEST_VIS:-10}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

A_EPOCH="${A_EPOCH:-700}"
B_EPOCH="${B_EPOCH:-1800}"

A_RUN="${BASE_RUN_NAME}_A_full"
B_RUN="${BASE_RUN_NAME}_B_mixed_release"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if (( B_EPOCH <= A_EPOCH )); then
    echo "[ERROR] Require A_EPOCH < B_EPOCH." >&2
    exit 1
fi

common_env=(
    OUTPUT_ROOT="${OUTPUT_ROOT}"
    DATASET="${DATASET}"
    SEED="${SEED}"
    GPU_ID="${GPU_ID}"
    DEVICE="${DEVICE}"
    WANDB_MODE="${WANDB_MODE}"
    N_ENVS="${N_ENVS}"
    N_TEST="${N_TEST}"
    N_TEST_VIS="${N_TEST_VIS}"
    ROLLOUT_EVERY="${ROLLOUT_EVERY}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}"
)

echo "[STACK-CURRICULUM] base=${BASE_RUN_NAME} stages=${A_EPOCH}->${B_EPOCH}"

echo "[STACK-CURRICULUM] ===== A: full trajectory skeleton ====="
env "${common_env[@]}" \
    RUN_NAME="${A_RUN}" \
    NUM_EPOCHS="${A_EPOCH}" \
    RESUME_EXISTING=true \
    PHASE_LOSS_WEIGHT=0.05 \
    EVENT_OVERSAMPLE_REPEATS=3 \
    bash scripts/run_stack_delta_hg_gc_fix_from_scratch.sh

A_CKPT="${OUTPUT_ROOT}/${A_RUN}/checkpoints/latest.ckpt"
if [[ ! -f "${A_CKPT}" ]]; then
    echo "[ERROR] Stage A checkpoint missing: ${A_CKPT}" >&2
    exit 1
fi

echo "[STACK-CURRICULUM] ===== B: mixed placement and release ====="
env "${common_env[@]}" \
    RUN_NAME="${B_RUN}" \
    SOURCE_CKPT="${A_CKPT}" \
    EXPECTED_SOURCE_EPOCH="${A_EPOCH}" \
    MIN_SOURCE_EPOCH="${A_EPOCH}" \
    FINETUNE_EPOCHS="$((B_EPOCH - A_EPOCH))" \
    FINETUNE_LR=5e-6 \
    GRIPPER_USE_HISTORY=false \
    PHASE_WEIGHT=0.02 \
    PRE_RELEASE_BEFORE=12 \
    SETTLE_BEFORE=4 \
    SETTLE_AFTER=2 \
    RELEASE_RADIUS=2 \
    RELEASE_TRANSLATION_WEIGHT=1.0 \
    RELEASE_ROTATION_WEIGHT=0.6 \
    RELEASE_GRIPPER_WEIGHT=5.0 \
    POST_RELEASE_AFTER=16 \
    POST_RELEASE_GRIPPER_WEIGHT=6.0 \
    PARAMETER_ANCHOR_RATE=5e-6 \
    EVENT_OVERSAMPLE_BEFORE=12 \
    EVENT_OVERSAMPLE_AFTER=16 \
    EVENT_OVERSAMPLE_REPEATS=2 \
    bash scripts/run_stack_release_finetune.sh

echo "[STACK-CURRICULUM] complete: ${OUTPUT_ROOT}/${B_RUN}"
