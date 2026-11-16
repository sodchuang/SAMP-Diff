#!/usr/bin/env bash
# Reproduce the original Square PH HG+GC recipe that reached 0.68 on seed 42.
# This is a single performance run, not the multi-variant ablation launcher.

set -Eeuo pipefail

SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-square_gc_fix_reproduce_v2_seed42}"
NUM_EPOCHS="${NUM_EPOCHS:-7001}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic}"
WANDB_MODE="${WANDB_MODE:-disabled}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET="data/robomimic/datasets/square/ph/low_dim_abs.hdf5"
if [[ ! -f "${DATASET}" ]]; then
  echo "[ERROR] Missing Square PH dataset: ${DATASET}" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Output already exists: ${OUTPUT_DIR}" >&2
  echo "Use a new RUN_NAME; reproduction must start cleanly from epoch 0." >&2
  exit 1
fi

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[SQUARE-REPRO] recipe=original_gc_fix_hg_gc"
echo "[SQUARE-REPRO] seed=${SEED} epochs=0..$((NUM_EPOCHS - 1)) eval_every=${ROLLOUT_EVERY}"
echo "[SQUARE-REPRO] action_steps=4 inference_steps=6 history=legacy_shift gripper_history=false"
echo "[SQUARE-REPRO] phase_loss=grasp_only weight=0.03 release_phase=false"
echo "[SQUARE-REPRO] output=${OUTPUT_DIR}"

SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
N_ENVS="${N_ENVS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
HISTORY_TRAINING_MODE="legacy_shift" \
RESUME_EXISTING="false" \
PHASE_LOSS_WEIGHT="0.03" \
bash scripts/run_square_group_history.sh GC \
  "task.dataset_path=${DATASET}" \
  "training.seed=${SEED}" \
  "data_split.seed=${SEED}" \
  "training.checkpoint_every=${CHECKPOINT_EVERY}" \
  "task.env_runner.n_test=${N_TEST}" \
  "task.env_runner.n_test_vis=${N_TEST_VIS}" \
  "task.env_runner.test_start_seed=100000" \
  "policy.num_inference_steps=6" \
  "logging.mode=${WANDB_MODE}" \
  "logging.name=${RUN_NAME}"
