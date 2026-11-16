#!/usr/bin/env bash
# Sealed Square PH recipe corresponding to the verified HG+GC configuration.
# This launcher intentionally does not call the mutable curriculum / ablation
# wrappers used by ToolHang, Stack, Transport, or the Square HMI study.

set -Eeuo pipefail

SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-square_ph_hg_gc_isolated_v4_seed42}"
NUM_EPOCHS="${NUM_EPOCHS:-7001}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DATASET="${DATASET:-data/robomimic/datasets/square/ph/low_dim_abs.hdf5}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export WANDB_MODE HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] Missing Square PH dataset: ${DATASET}" >&2
    exit 2
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] Output already exists: ${OUTPUT_DIR}" >&2
    echo "Use a new RUN_NAME; isolated reproduction must start at epoch 0." >&2
    exit 1
fi

echo "[SQUARE-ISOLATED] task=square_ph control=abs_eef_pose seed=${SEED}"
echo "[SQUARE-ISOLATED] recipe=HG+GC grasp-only, legacy_shift"
echo "[SQUARE-ISOLATED] toolhang_prefix=false stack_event_sampling=false"
echo "[SQUARE-ISOLATED] release_phase=false transfer=false resume=false"
echo "[SQUARE-ISOLATED] epochs=0..$((NUM_EPOCHS - 1)) output=${OUTPUT_DIR}"

python train.py --config-name=square_ph \
    "hydra.run.dir=${OUTPUT_DIR}" \
    "task.dataset_path=${DATASET}" \
    "training.seed=${SEED}" \
    "data_split.seed=${SEED}" \
    training.resume=false \
    "training.num_epochs=${NUM_EPOCHS}" \
    "training.rollout_every=${ROLLOUT_EVERY}" \
    "training.checkpoint_every=${CHECKPOINT_EVERY}" \
    n_action_steps=4 \
    policy.num_inference_steps=6 \
    policy.cold_start_prob=0.3 \
    policy.action_group_loss_weights=null \
    policy.action_group_spectral_params.translation.freq_split_high=8 \
    policy.action_group_spectral_params.translation.sigma=0.3 \
    policy.action_group_spectral_params.translation.sigma_high=0.2 \
    policy.action_group_spectral_params.rotation.freq_split_high=16 \
    policy.action_group_spectral_params.rotation.sigma=0.15 \
    policy.action_group_spectral_params.rotation.sigma_high=0.15 \
    policy.action_group_spectral_params.gripper.freq_split_high=16 \
    policy.action_group_spectral_params.gripper.sigma=0.5 \
    policy.action_group_spectral_params.gripper.sigma_high=0.5 \
    policy.action_group_history_params.enabled=true \
    policy.action_group_history_params.shift=4 \
    policy.action_group_history_params.groups.translation.use_history=true \
    policy.action_group_history_params.groups.translation.update_rate=1.0 \
    policy.action_group_history_params.groups.rotation.use_history=true \
    policy.action_group_history_params.groups.rotation.update_rate=1.0 \
    policy.action_group_history_params.groups.gripper.use_history=false \
    policy.action_group_history_params.groups.gripper.update_rate=1.0 \
    policy.action_group_timing_params.gripper.enabled=false \
    policy.action_phase_loss_params.enabled=true \
    policy.action_phase_loss_params.weight=0.03 \
    policy.action_phase_loss_params.close_threshold=0.0 \
    policy.action_phase_loss_params.close_is_greater=true \
    policy.action_phase_loss_params.transition_radius=2 \
    policy.action_phase_loss_params.translation_weight=1.5 \
    policy.action_phase_loss_params.rotation_weight=1.3 \
    policy.action_phase_loss_params.gripper_weight=1.5 \
    policy.action_phase_loss_params.release_enabled=false \
    "task.env_runner.n_envs=${N_ENVS}" \
    "task.env_runner.n_test=${N_TEST}" \
    "task.env_runner.n_test_vis=${N_TEST_VIS}" \
    task.env_runner.test_start_seed=100000 \
    dataloader.persistent_workers=true \
    val_dataloader.persistent_workers=true \
    "logging.mode=${WANDB_MODE}" \
    "logging.name=${RUN_NAME}"
