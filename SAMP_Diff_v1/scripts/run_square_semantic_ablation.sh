#!/usr/bin/env bash
# Fixed-compute single-task ablation for the paper's semantic-module claim.
#
# The completed Square HG+GC seed-42 run (best 0.68) is the FULL result.
# By default this script only trains the three controls needed for a compact
# additive ablation. GC_ONLY remains available as an optional factorial check.
#   HOMO     homogeneous hand-agnostic source prior
#   SP       semantic spectral prior only
#   HG       SP + history routing (translation/rotation reused, gripper fresh)
#   GC_ONLY  optional: SP + grasp coordination loss, without history routing
#
# Example:
#   nohup bash scripts/run_square_semantic_ablation.sh \
#     > square_semantic_ablation_seed42.log 2>&1 &

set -euo pipefail

SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-HOMO SP HG}"
NUM_EPOCHS="${NUM_EPOCHS:-7000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/square_semantic_ablation_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "${OUTPUT_ROOT}"

for variant in ${VARIANTS}; do
    variant_lower="$(printf '%s' "${variant}" | tr '[:upper:]' '[:lower:]')"
    run_name="square_ablation_${variant_lower}_seed${SEED}"

    echo
    echo "============================================================"
    echo "[SQUARE_ABLATION] variant=${variant} seed=${SEED}"
    echo "[SQUARE_ABLATION] fixed: action_steps=4 inference_steps=6"
    echo "[SQUARE_ABLATION] output=${OUTPUT_ROOT}/${run_name}"
    echo "============================================================"

    SEED="${SEED}" \
    RUN_NAME="${run_name}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
    N_ENVS="${N_ENVS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HISTORY_TRAINING_MODE="legacy_shift" \
    bash scripts/run_square_group_history.sh "${variant}" \
        "training.seed=${SEED}" \
        "data_split.seed=${SEED}" \
        training.checkpoint_every=50 \
        "task.env_runner.n_test=${N_TEST}" \
        "task.env_runner.n_test_vis=${N_TEST_VIS}" \
        task.env_runner.test_start_seed=100000 \
        "logging.mode=${WANDB_MODE}"
done

echo
echo "[SQUARE_ABLATION] Complete: ${VARIANTS}"
echo "[SQUARE_ABLATION] Use square_gc_fix_seed42_v1 best=0.68 as FULL."
