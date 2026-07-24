#!/usr/bin/env bash
set -euo pipefail

variant="${1:-B1}"

case "${variant}" in
  B0)
    name="square_legacy_rot015_6000"
    translation_sigma="0.3"
    translation_sigma_high="0.2"
    gripper_sigma="0.5"
    balanced_loss="false"
    ;;
  B1)
    name="square_balanced_rot015_6000"
    translation_sigma="0.3"
    translation_sigma_high="0.2"
    gripper_sigma="0.5"
    balanced_loss="true"
    ;;
  B2)
    name="square_balanced_trans02_rot015_6000"
    translation_sigma="0.2"
    translation_sigma_high="0.1"
    gripper_sigma="0.5"
    balanced_loss="true"
    ;;
  B3)
    name="square_balanced_trans02_rot015_grip03_6000"
    translation_sigma="0.2"
    translation_sigma_high="0.1"
    gripper_sigma="0.3"
    balanced_loss="true"
    ;;
  *)
    echo "Usage: $0 {B0|B1|B2|B3}" >&2
    exit 2
    ;;
esac

if [[ "${balanced_loss}" == "true" ]]; then
  loss_args=(
    policy.action_group_loss_weights.translation.weight=1.0
    policy.action_group_loss_weights.rotation.weight=1.0
    policy.action_group_loss_weights.gripper.weight=1.0
  )
else
  loss_args=(policy.action_group_loss_weights=null)
fi

name="${RUN_NAME:-${name}}"
num_epochs="${NUM_EPOCHS:-6000}"
rollout_every="${ROLLOUT_EVERY:-50}"
n_envs="${N_ENVS:-28}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python train.py --config-name=square_ph \
  "hydra.run.dir=data/outputs/robomimic/${name}" \
  training.resume=false \
  "training.num_epochs=${num_epochs}" \
  "training.rollout_every=${rollout_every}" \
  n_action_steps=4 \
  dataloader.persistent_workers=true \
  val_dataloader.persistent_workers=true \
  "task.env_runner.n_envs=${n_envs}" \
  policy.action_group_spectral_params.translation.freq_split_high=8 \
  "policy.action_group_spectral_params.translation.sigma=${translation_sigma}" \
  "policy.action_group_spectral_params.translation.sigma_high=${translation_sigma_high}" \
  policy.action_group_spectral_params.rotation.freq_split_high=16 \
  policy.action_group_spectral_params.rotation.sigma=0.15 \
  policy.action_group_spectral_params.rotation.sigma_high=0.15 \
  policy.action_group_spectral_params.gripper.freq_split_high=16 \
  "policy.action_group_spectral_params.gripper.sigma=${gripper_sigma}" \
  "policy.action_group_spectral_params.gripper.sigma_high=${gripper_sigma}" \
  "${loss_args[@]}" \
  "logging.name=${name}"
