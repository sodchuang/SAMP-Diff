#!/usr/bin/env bash
set -euo pipefail

variant="${1:-H1}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_overrides=("$@")

case "${variant}" in
  HOMO)
    name="square_homogeneous_prior_seed42"
    history_enabled="false"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    # Applied last so Hydra first accepts the normal Square group fields and
    # then collapses them into the homogeneous source-prior control.
    extra_overrides+=("policy.action_group_spectral_params=null")
    ;;
  SP)
    name="square_semantic_prior_only_seed42"
    history_enabled="false"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    ;;
  H0)
    name="square_history_legacy_rot015_6000"
    history_enabled="false"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    ;;
  H1)
    name="square_history_aligned_rot015_6000"
    history_enabled="true"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    ;;
  H2)
    name="square_history_independent_rot015_6000"
    history_enabled="true"
    translation_rate="0.7"
    rotation_rate="1.0"
    gripper_rate="0.5"
    gripper_history="true"
    ;;
  HG)
    name="square_history_hybrid_gripper_fresh_rot015_6000"
    history_enabled="true"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    gripper_history="false"
    ;;
  GC)
    name="square_grasp_coordination_rot015_6000"
    history_enabled="true"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    gripper_history="false"
    phase_loss_enabled="true"
    ;;
  GC_ONLY)
    name="square_grasp_coordination_only_seed42"
    history_enabled="false"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    gripper_history="false"
    phase_loss_enabled="true"
    ;;
  FULL)
    name="square_hg_gc_full_seed42"
    history_enabled="true"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    gripper_history="false"
    phase_loss_enabled="true"
    ;;
  GCR)
    name="square_grasp_release_coordination_rot015_7000"
    history_enabled="true"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    gripper_history="false"
    phase_loss_enabled="true"
    release_phase_enabled="true"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.01}"
    num_epochs="${NUM_EPOCHS:-7000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    ;;
  *)
    echo "Usage: $0 {HOMO|SP|HG|GC_ONLY|FULL|H0|H1|H2|GC|GCR}" >&2
    exit 2
    ;;
esac

gripper_history="${gripper_history:-true}"
phase_loss_enabled="${phase_loss_enabled:-false}"
release_phase_enabled="${release_phase_enabled:-false}"
phase_loss_weight="${PHASE_LOSS_WEIGHT:-${phase_loss_weight:-0.03}}"

name="${RUN_NAME:-${name}}"
num_epochs="${NUM_EPOCHS:-${num_epochs:-6000}}"
rollout_every="${ROLLOUT_EVERY:-${rollout_every:-50}}"
n_envs="${N_ENVS:-28}"
output_root="${OUTPUT_ROOT:-data/outputs/robomimic}"
history_training_mode="${HISTORY_TRAINING_MODE:-legacy_shift}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python train.py --config-name=square_ph \
  "hydra.run.dir=${output_root}/${name}" \
  training.resume=false \
  "training.num_epochs=${num_epochs}" \
  "training.rollout_every=${rollout_every}" \
  n_action_steps=4 \
  dataloader.persistent_workers=true \
  val_dataloader.persistent_workers=true \
  "task.env_runner.n_envs=${n_envs}" \
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
  "policy.action_group_history_params.enabled=${history_enabled}" \
  policy.action_group_history_params.shift=4 \
  "policy.action_group_history_params.groups.translation.update_rate=${translation_rate}" \
  "policy.action_group_history_params.groups.rotation.update_rate=${rotation_rate}" \
  "policy.action_group_history_params.groups.gripper.use_history=${gripper_history}" \
  "policy.action_group_history_params.groups.gripper.update_rate=${gripper_rate}" \
  "++policy.history_training_mode=${history_training_mode}" \
  "policy.action_phase_loss_params.enabled=${phase_loss_enabled}" \
  "policy.action_phase_loss_params.weight=${phase_loss_weight}" \
  policy.action_phase_loss_params.transition_radius=2 \
  policy.action_phase_loss_params.translation_weight=1.5 \
  policy.action_phase_loss_params.rotation_weight=1.3 \
  policy.action_phase_loss_params.gripper_weight=1.5 \
  "policy.action_phase_loss_params.release_enabled=${release_phase_enabled}" \
  policy.action_phase_loss_params.release_transition_radius=3 \
  policy.action_phase_loss_params.release_translation_weight=2.0 \
  policy.action_phase_loss_params.release_rotation_weight=1.8 \
  policy.action_phase_loss_params.release_gripper_weight=1.2 \
  "logging.name=${name}" \
  "${extra_overrides[@]}"
