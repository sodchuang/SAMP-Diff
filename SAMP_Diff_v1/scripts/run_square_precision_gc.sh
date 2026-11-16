#!/usr/bin/env bash
# Square PH precision-insertion recipe.
# Keeps the verified HG+GC structure while reducing high-frequency pose noise.

set -Eeuo pipefail

SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-square_precision_gc_v3_seed42}"
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
  echo "Use a new RUN_NAME; precision comparison must start from epoch 0." >&2
  exit 1
fi

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[SQUARE-PRECISION] semantic=HG+GC+release"
echo "[SQUARE-PRECISION] translation_sigma=0.22/0.08 rotation_sigma=0.10/0.05"
echo "[SQUARE-PRECISION] action_steps=4 inference_steps=6 (fixed)"
echo "[SQUARE-PRECISION] gripper_sigma=0.40/0.25"
echo "[SQUARE-PRECISION] sequence=insert_and_settle -> narrow_release -> open_and_retreat"
echo "[SQUARE-PRECISION] output=${OUTPUT_DIR}"

SEED="${SEED}" \
RUN_NAME="${RUN_NAME}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
N_ENVS="${N_ENVS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
HISTORY_TRAINING_MODE="legacy_shift" \
RESUME_EXISTING="false" \
PHASE_LOSS_WEIGHT="0.015" \
bash scripts/run_square_group_history.sh GCR \
  "task.dataset_path=${DATASET}" \
  "training.seed=${SEED}" \
  "data_split.seed=${SEED}" \
  "training.checkpoint_every=${CHECKPOINT_EVERY}" \
  "task.env_runner.n_test=${N_TEST}" \
  "task.env_runner.n_test_vis=${N_TEST_VIS}" \
  "task.env_runner.test_start_seed=100000" \
  "policy.num_inference_steps=6" \
  "policy.action_group_spectral_params.translation.sigma=0.22" \
  "policy.action_group_spectral_params.translation.sigma_high=0.08" \
  "policy.action_group_spectral_params.rotation.sigma=0.10" \
  "policy.action_group_spectral_params.rotation.sigma_high=0.05" \
  "policy.action_group_spectral_params.gripper.sigma=0.40" \
  "policy.action_group_spectral_params.gripper.sigma_high=0.25" \
  "policy.action_phase_loss_params.release_enabled=true" \
  "policy.action_phase_loss_params.release_transition_radius=1" \
  "policy.action_phase_loss_params.release_translation_weight=2.5" \
  "policy.action_phase_loss_params.release_rotation_weight=2.0" \
  "policy.action_phase_loss_params.release_gripper_weight=2.0" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.enabled=true" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.anchor=release" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.occurrence=last" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.before=12" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.after=0" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.translation_weight=4.0" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.rotation_weight=3.0" \
  "++policy.action_phase_loss_params.extra_windows.square_pre_release_insert.gripper_weight=1.5" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.enabled=true" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.anchor=release" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.occurrence=last" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.before=0" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.after=12" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.translation_weight=0.25" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.rotation_weight=0.20" \
  "++policy.action_phase_loss_params.extra_windows.square_post_release.gripper_weight=3.0" \
  "logging.mode=${WANDB_MODE}" \
  "logging.name=${RUN_NAME}"
