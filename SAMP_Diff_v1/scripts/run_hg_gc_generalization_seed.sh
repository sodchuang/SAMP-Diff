#!/usr/bin/env bash
# One complete generalization seed using the successful HG + GC-fix recipe.
#
# HG:
#   reuse translation/rotation history; regenerate gripper every action chunk.
# GC fix:
#   add close-transition coordination loss with the successful Square weights.
#
# Each task is trained independently. The HG/GC module structure and test seed
# bank stay fixed, while each task keeps its validated semantic-group spectral
# values. Dual-arm tasks also receive their physically correct action indices.

set -euo pipefail

SEED="${SEED:-42}"
TASKS="${TASKS:-lift_ph can_ph square_ph transport_ph}"
NUM_EPOCHS="${NUM_EPOCHS:-7000}"
TRAINING_STOP_EPOCH="$((NUM_EPOCHS + 1))"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-4}"
TEST_START_SEED="${TEST_START_SEED:-100000}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-data/robomimic/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/hg_gc_generalization}"
RESUME="${RESUME:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}"

for task in ${TASKS}; do
    task_name="${task%_ph}"
    dataset_path="${DATA_ROOT}/${task_name}/ph/low_dim_abs.hdf5"
    output_dir="${OUTPUT_ROOT}/${task}_hg_gc_fix_seed${SEED}"

    if [[ ! -f "${dataset_path}" ]]; then
        echo "[ERROR] Missing dataset: ${dataset_path}" >&2
        exit 1
    fi

    case "${task}" in
        lift_ph|can_ph)
            translation_indices="[0,1,2]"
            rotation_indices="[3,4,5,6,7,8]"
            gripper_indices="[9]"
            # HG rollout advances history by a full action chunk.  Can was
            # verified with the matching teacher-forced aligned prior.
            history_training_mode="aligned"
            rotation_freq_split_high=12
            rotation_sigma=0.25
            rotation_sigma_high=0.3
            ;;
        square_ph)
            translation_indices="[0,1,2]"
            rotation_indices="[3,4,5,6,7,8]"
            gripper_indices="[9]"
            # Preserve the successful Square HG+GC specialization.
            # The recorded 0.60 Square GC-fix run used legacy_shift; changing
            # it globally to aligned makes this late-learning task regress.
            history_training_mode="legacy_shift"
            rotation_freq_split_high=16
            rotation_sigma=0.15
            rotation_sigma_high=0.15
            ;;
        transport_ph)
            translation_indices="[0,1,2,10,11,12]"
            rotation_indices="[3,4,5,6,7,8,13,14,15,16,17,18]"
            gripper_indices="[9,19]"
            history_training_mode="aligned"
            rotation_freq_split_high=12
            rotation_sigma=0.25
            rotation_sigma_high=0.3
            ;;
        *)
            echo "[ERROR] No verified action-semantic map for task: ${task}" >&2
            exit 2
            ;;
    esac

    echo
    echo "================================================================="
    echo "[HG_GC] task=${task} seed=${SEED}"
    echo "[HG_GC] output=${output_dir}"
    echo "[HG_GC] epochs=0..${NUM_EPOCHS} (Hydra stop=${TRAINING_STOP_EPOCH})"
    echo "[HG_GC] groups=translation(8,0.3,0.2) rotation(${rotation_freq_split_high},${rotation_sigma},${rotation_sigma_high}) gripper(16,0.5,0.5)"
    echo "[HG_GC] test seeds=${TEST_START_SEED}..$((TEST_START_SEED + N_TEST - 1))"
    echo "================================================================="

    python train.py --config-name="${task}" \
        "hydra.run.dir=${output_dir}" \
        "task.dataset_path=${dataset_path}" \
        "training.device=${DEVICE}" \
        "training.seed=${SEED}" \
        "data_split.seed=${SEED}" \
        "training.resume=${RESUME}" \
        "training.num_epochs=${TRAINING_STOP_EPOCH}" \
        "training.rollout_every=${ROLLOUT_EVERY}" \
        "training.checkpoint_every=${CHECKPOINT_EVERY}" \
        "task.env_runner.n_envs=${N_ENVS}" \
        "task.env_runner.n_test=${N_TEST}" \
        "task.env_runner.n_test_vis=${N_TEST_VIS}" \
        "task.env_runner.test_start_seed=${TEST_START_SEED}" \
        "logging.mode=${WANDB_MODE}" \
        "logging.project=samp_hg_gc_generalization" \
        "logging.name=${task}_hg_gc_fix_seed${SEED}" \
        ++task.dataset.episode_prefix_close_threshold=auto \
        ++task.dataset.episode_prefix_close_is_greater=auto \
        n_action_steps=4 \
        policy.num_inference_steps=6 \
        policy.sigma=0.3 \
        policy.cold_start_prob=0.3 \
        policy.freq_split_low=0 \
        policy.freq_split_high=8 \
        policy.sigma_high=0.2 \
        ++policy.action_group_loss_weights=null \
        "++policy.action_group_spectral_params.translation.indices=${translation_indices}" \
        ++policy.action_group_spectral_params.translation.freq_split_high=8 \
        ++policy.action_group_spectral_params.translation.sigma=0.3 \
        ++policy.action_group_spectral_params.translation.sigma_high=0.2 \
        "++policy.action_group_spectral_params.rotation.indices=${rotation_indices}" \
        "++policy.action_group_spectral_params.rotation.freq_split_high=${rotation_freq_split_high}" \
        "++policy.action_group_spectral_params.rotation.sigma=${rotation_sigma}" \
        "++policy.action_group_spectral_params.rotation.sigma_high=${rotation_sigma_high}" \
        "++policy.action_group_spectral_params.gripper.indices=${gripper_indices}" \
        ++policy.action_group_spectral_params.gripper.freq_split_high=16 \
        ++policy.action_group_spectral_params.gripper.sigma=0.5 \
        ++policy.action_group_spectral_params.gripper.sigma_high=0.5 \
        ++policy.action_group_history_params.enabled=true \
        ++policy.action_group_history_params.shift=4 \
        "++policy.action_group_history_params.groups.translation.indices=${translation_indices}" \
        ++policy.action_group_history_params.groups.translation.use_history=true \
        ++policy.action_group_history_params.groups.translation.update_rate=1.0 \
        "++policy.action_group_history_params.groups.rotation.indices=${rotation_indices}" \
        ++policy.action_group_history_params.groups.rotation.use_history=true \
        ++policy.action_group_history_params.groups.rotation.update_rate=1.0 \
        "++policy.action_group_history_params.groups.gripper.indices=${gripper_indices}" \
        ++policy.action_group_history_params.groups.gripper.use_history=false \
        ++policy.action_group_history_params.groups.gripper.update_rate=1.0 \
        "++policy.history_training_mode=${history_training_mode}" \
        ++policy.action_phase_loss_params.enabled=true \
        ++policy.action_phase_loss_params.weight=0.03 \
        "++policy.action_phase_loss_params.gripper_indices=${gripper_indices}" \
        ++policy.action_phase_loss_params.close_threshold=0.0 \
        ++policy.action_phase_loss_params.close_is_greater=true \
        ++policy.action_phase_loss_params.transition_radius=2 \
        "++policy.action_phase_loss_params.translation_indices=${translation_indices}" \
        ++policy.action_phase_loss_params.translation_weight=1.5 \
        "++policy.action_phase_loss_params.rotation_indices=${rotation_indices}" \
        ++policy.action_phase_loss_params.rotation_weight=1.3 \
        ++policy.action_phase_loss_params.gripper_weight=1.5 \
        ++policy.action_phase_loss_params.release_enabled=false
done

echo
echo "[HG_GC] Complete seed ${SEED} finished: ${TASKS}"
