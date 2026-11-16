#!/usr/bin/env bash
# Clean single-H100 Can PH source-prior ablation.
# The four variants differ only in source-prior construction.

set -euo pipefail

VARIANTS="${VARIANTS:-base uniform split full}"
NUM_EPOCHS="${NUM_EPOCHS:-3000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
N_ENVS="${N_ENVS:-16}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-4}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic_can_source_prior_ablation_3000}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DATASET="${DATASET:-data/robomimic/datasets/can/ph/low_dim_abs.hdf5}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

[[ -f "${DATASET}" ]] || { echo "[ERROR] missing Can PH dataset: ${DATASET}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"

run_variant() {
    local variant="$1"
    local run_name="can_ph_${variant}_seed${SEED}"
    local output_dir="${OUTPUT_ROOT}/${run_name}"
    local latest="${output_dir}/checkpoints/latest.ckpt"
    local prior_args=()

    if [[ -f "${latest}" ]]; then
        echo "[SKIP TRAINED] ${variant}: ${latest}"
        return
    fi

    case "${variant}" in
        base)
            # Original homogeneous frequency-domain A2A warm start.
            prior_args+=(
                "++policy.action_group_spectral_params=null"
                "++policy.action_group_history_params=null"
                "++policy.action_phase_loss_params=null"
            )
            ;;
        uniform)
            # One shared history group and one homogeneous spectral schedule.
            prior_args+=(
                "++policy.action_group_spectral_params=null"
                "++policy.action_group_history_params={enabled:true,shift:4,groups:{all:{indices:[0,1,2,3,4,5,6,7,8,9],use_history:true,update_rate:1.0}}}"
                "++policy.history_training_mode=aligned"
                "++policy.action_phase_loss_params=null"
            )
            ;;
        split)
            # Semantic history routing only; spectral schedule stays homogeneous.
            prior_args+=(
                "++policy.action_group_spectral_params=null"
                "++policy.action_group_history_params={enabled:true,shift:4,groups:{translation:{indices:[0,1,2],use_history:true,update_rate:1.0},rotation:{indices:[3,4,5,6,7,8],use_history:true,update_rate:0.75},gripper:{indices:[9],use_history:false,update_rate:1.0}}}"
                "++policy.history_training_mode=aligned"
                "++policy.action_phase_loss_params=null"
            )
            ;;
        full)
            # Split history plus action-parameter-specific spectral modulation.
            prior_args+=(
                "++policy.action_group_history_params={enabled:true,shift:4,groups:{translation:{indices:[0,1,2],use_history:true,update_rate:1.0},rotation:{indices:[3,4,5,6,7,8],use_history:true,update_rate:0.75},gripper:{indices:[9],use_history:false,update_rate:1.0}}}"
                "++policy.history_training_mode=aligned"
                "++policy.action_group_spectral_params={translation:{indices:[0,1,2],freq_split_high:8,sigma:0.3,sigma_high:0.2},rotation:{indices:[3,4,5,6,7,8],freq_split_high:16,sigma:0.18,sigma_high:0.18},gripper:{indices:[9],freq_split_high:16,sigma:0.5,sigma_high:0.5}}"
                "++policy.action_phase_loss_params=null"
            )
            ;;
        *) echo "[ERROR] unknown Can variant: ${variant}" >&2; exit 2 ;;
    esac

    mkdir -p "${output_dir}"
    echo "[CAN] variant=${variant} seed=${SEED} epochs=${NUM_EPOCHS} output=${output_dir}"
    python train.py --config-name=can_ph \
        "hydra.run.dir=${output_dir}" \
        "task.dataset_path=${DATASET}" \
        training.device=cuda:0 \
        training.resume=false \
        "training.num_epochs=${NUM_EPOCHS}" \
        "training.rollout_every=${ROLLOUT_EVERY}" \
        "training.seed=${SEED}" \
        "data_split.seed=${SEED}" \
        n_action_steps=4 \
        policy.num_inference_steps=6 \
        policy.sigma=0.3 \
        policy.cold_start_prob=0.3 \
        policy.freq_split_low=0 \
        policy.freq_split_high=8 \
        policy.sigma_high=0.2 \
        "task.env_runner.n_envs=${N_ENVS}" \
        "task.env_runner.n_test=${N_TEST}" \
        "task.env_runner.n_test_vis=${N_TEST_VIS}" \
        "logging.mode=${WANDB_MODE}" \
        "logging.name=${run_name}" \
        "${prior_args[@]}" \
        2>&1 | tee "${output_dir}/train.log"
}

for variant in ${VARIANTS}; do
    run_variant "${variant}"
done

echo "[CAN] requested variants complete: ${VARIANTS}"
