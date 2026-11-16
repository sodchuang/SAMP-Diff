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
  WRONG_SP)
    name="square_wrong_semantic_prior_seed42"
    history_enabled="false"
    translation_rate="1.0"
    rotation_rate="1.0"
    gripper_rate="1.0"
    # Negative control: preserve the number of dimensions and all spectral
    # hyperparameters, but deliberately assign physical action dimensions to
    # the wrong semantic groups. This isolates the value of human-specified
    # action semantics from simply having multiple parameter groups.
    extra_overrides+=(
      "policy.action_group_spectral_params.translation.indices=[3,4,5]"
      "policy.action_group_spectral_params.rotation.indices=[0,1,2,6,7,9]"
      "policy.action_group_spectral_params.gripper.indices=[8]"
    )
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
    echo "Usage: $0 {HOMO|WRONG_SP|SP|HG|GC_ONLY|FULL|H0|H1|H2|GC|GCR}" >&2
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
resume_existing="${RESUME_EXISTING:-auto}"
min_resume_epoch="${MIN_RESUME_EPOCH:-0}"

output_dir="${output_root}/${name}"
training_resume=false
if [[ "${resume_existing}" == "auto" ]]; then
  if [[ -f "${output_dir}/checkpoints/latest.ckpt" ]]; then
    resume_existing=true
  else
    resume_existing=false
  fi
fi
if [[ "${resume_existing}" == "true" ]]; then
  latest_ckpt="${output_dir}/checkpoints/latest.ckpt"
  if [[ ! -f "${latest_ckpt}" ]]; then
    echo "[ERROR] Resume requested but checkpoint is missing: ${latest_ckpt}" >&2
    echo "Check OUTPUT_ROOT; refusing to restart ${name} from epoch 0." >&2
    exit 1
  fi

  resume_epoch="$(python - "${latest_ckpt}" <<'PY'
import dill
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", pickle_module=dill)
raw = payload.get("pickles", {}).get("epoch")
if raw is None:
    raise SystemExit("[ERROR] checkpoint does not contain epoch")
print(int(dill.loads(raw)))
PY
)"

  if (( resume_epoch < min_resume_epoch )); then
    echo "[ERROR] ${name} latest checkpoint is epoch ${resume_epoch}; expected >= ${min_resume_epoch}." >&2
    echo "Refusing to mix an incomplete run into the 6000->${num_epochs} extension." >&2
    exit 1
  fi
  if (( resume_epoch >= num_epochs - 1 )); then
    echo "[SQUARE] ${name} already reached epoch ${resume_epoch}; target=${num_epochs}. Skip."
    exit 0
  fi
  training_resume=true
  echo "[SQUARE] continuing ${name}: epoch $((resume_epoch + 1)) -> $((num_epochs - 1))"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python train.py --config-name=square_ph \
  "hydra.run.dir=${output_dir}" \
  "training.resume=${training_resume}" \
  "training.num_epochs=${num_epochs}" \
  "++training.lr_scheduler_num_epochs=${num_epochs}" \
  "++training.lr_scheduler_start_epoch=0" \
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
