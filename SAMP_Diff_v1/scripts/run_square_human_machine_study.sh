#!/usr/bin/env bash
# Square study for the thesis claim:
#   human-designed action semantics + learned conditional flow coordination.
#
# Fixed across variants: dataset, seed, compute, rollout seeds, action horizon,
# flow architecture, and inference steps. Only the human semantic structure
# and coordination modules are changed.

set -Eeuo pipefail

SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-FULL HOMO WRONG_SP SP HG}"
NUM_EPOCHS="${NUM_EPOCHS:-9000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
TEST_START_SEED="${TEST_START_SEED:-100000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/square_human_machine_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-disabled}"
RESUME_EXISTING="${RESUME_EXISTING:-auto}"
MIN_RESUME_EPOCH="${MIN_RESUME_EPOCH:-0}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

dataset="data/robomimic/datasets/square/ph/low_dim_abs.hdf5"
if [[ ! -f "${dataset}" ]]; then
    echo "[ERROR] Missing Square PH dataset: ${dataset}" >&2
    exit 2
fi

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}"

# Validate every source before spending hours on the first variant. The child
# launcher performs the stronger checkpoint-epoch check immediately before
# each resume.
if [[ "${RESUME_EXISTING}" == "true" ]]; then
    echo "[SQUARE_HMI] preflight: verifying all five 6000-epoch runs"
    for variant in ${VARIANTS}; do
        variant_lower="$(printf '%s' "${variant}" | tr '[:upper:]' '[:lower:]')"
        run_name="square_hmi_${variant_lower}_seed${SEED}"
        latest_ckpt="${OUTPUT_ROOT}/${run_name}/checkpoints/latest.ckpt"
        if [[ ! -f "${latest_ckpt}" ]]; then
            echo "[ERROR] Missing resume checkpoint: ${latest_ckpt}" >&2
            echo "Check OUTPUT_ROOT; no Square variant has been started." >&2
            exit 1
        fi
        echo "[OK] ${variant}: ${latest_ckpt}"
    done
fi

for variant in ${VARIANTS}; do
    variant_lower="$(printf '%s' "${variant}" | tr '[:upper:]' '[:lower:]')"
    run_name="square_hmi_${variant_lower}_seed${SEED}"

    echo
    echo "============================================================"
    echo "[SQUARE_HMI] variant=${variant} seed=${SEED}"
    echo "[SQUARE_HMI] output=${OUTPUT_ROOT}/${run_name}"
    echo "[SQUARE_HMI] fixed action_steps=4 inference_steps=6"
    echo "============================================================"

    SEED="${SEED}" \
    RUN_NAME="${run_name}" \
    NUM_EPOCHS="${NUM_EPOCHS}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
    N_ENVS="${N_ENVS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HISTORY_TRAINING_MODE="legacy_shift" \
    RESUME_EXISTING="${RESUME_EXISTING}" \
    MIN_RESUME_EPOCH="${MIN_RESUME_EPOCH}" \
    bash scripts/run_square_group_history.sh "${variant}" \
        "task.dataset_path=${dataset}" \
        "training.seed=${SEED}" \
        "data_split.seed=${SEED}" \
        "training.checkpoint_every=${CHECKPOINT_EVERY}" \
        "task.env_runner.n_test=${N_TEST}" \
        "task.env_runner.n_test_vis=${N_TEST_VIS}" \
        "task.env_runner.test_start_seed=${TEST_START_SEED}" \
        "logging.mode=${WANDB_MODE}" \
        "logging.project=samp_square_human_machine" \
        "logging.name=${run_name}"
done

echo
echo "[SQUARE_HMI] Complete: ${VARIANTS}"
