#!/usr/bin/env bash
set -euo pipefail

export PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.01}"
export NUM_EPOCHS="${NUM_EPOCHS:-7000}"
export ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
export N_ENVS="${N_ENVS:-28}"

run_one() {
  local name="$1"
  shift
  echo "===== Running ${name} ====="
  RUN_NAME="${name}" bash scripts/run_square_group_history.sh GCR "$@"
}

run_one square_gcr6_base_7000

run_one square_gcr6_rot012_7000 \
  policy.action_group_spectral_params.rotation.sigma=0.12 \
  policy.action_group_spectral_params.rotation.sigma_high=0.12

run_one square_gcr6_release_heavy_7000 \
  policy.action_phase_loss_params.release_translation_weight=2.5 \
  policy.action_phase_loss_params.release_rotation_weight=2.2 \
  policy.action_phase_loss_params.release_gripper_weight=1.2
