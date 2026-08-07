#!/usr/bin/env bash
set -euo pipefail

variant="${1:-PICK}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_overrides=("$@")
dataset_path="${DATASET_PATH:-data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5}"

if [[ ! -f "${dataset_path}" ]]; then
  echo "[ERROR] ToolHang dataset not found: ${dataset_path}" >&2
  echo "[INFO] Run from the SAMP-Diff project root, or set DATASET_PATH explicitly." >&2
  echo "[INFO] Candidate ToolHang datasets under ./data:" >&2
  find data -type f -path '*tool_hang*' -name 'low_dim_abs.hdf5' -print 2>/dev/null >&2 || true
  exit 1
fi

case "${variant}" in
  PICK)
    name="${RUN_NAME:-tool_hang_pick_guided_6500}"
    num_epochs="${NUM_EPOCHS:-6500}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-28}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.015}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-2.0}"
    rotation_weight="${ROTATION_WEIGHT:-1.4}"
    gripper_weight="${GRIPPER_WEIGHT:-2.0}"
    rotation_sigma="${ROTATION_SIGMA:-0.12}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.0}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-1.0}"
    ;;
  ONE)
    name="${RUN_NAME:-tool_hang_one_layer_pick_to_hang_7000}"
    num_epochs="${NUM_EPOCHS:-7000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-300}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-12}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.006}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.5}"
    rotation_weight="${ROTATION_WEIGHT:-1.0}"
    gripper_weight="${GRIPPER_WEIGHT:-2.0}"
    rotation_sigma="${ROTATION_SIGMA:-0.10}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.0}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.8}"
    gripper_timing_enabled="${GRIPPER_TIMING_ENABLED:-true}"
    min_close_steps="${MIN_CLOSE_STEPS:-4}"
    max_latch_chunks_after_close="${MAX_LATCH_CHUNKS_AFTER_CLOSE:-8}"
    ;;
  LIFT)
    name="${RUN_NAME:-tool_hang_lift_a_pure_3000}"
    num_epochs="${NUM_EPOCHS:-3000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-300}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-12}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.013}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-2.7}"
    rotation_weight="${ROTATION_WEIGHT:-0.30}"
    gripper_weight="${GRIPPER_WEIGHT:-4.6}"
    rotation_sigma="${ROTATION_SIGMA:-0.07}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-0.4}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-0.4}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.2}"
    gripper_timing_enabled="${GRIPPER_TIMING_ENABLED:-true}"
    min_close_steps="${MIN_CLOSE_STEPS:-12}"
    max_latch_chunks_after_close="${MAX_LATCH_CHUNKS_AFTER_CLOSE:-24}"
    pick_window_anchor="${PICK_WINDOW_ANCHOR:-close}"
    pick_window_occurrence="${PICK_WINDOW_OCCURRENCE:-first}"
    pick_window_before="${PICK_WINDOW_BEFORE:-8}"
    pick_window_after="${PICK_WINDOW_AFTER:-24}"
    pick_window_translation_weight="${PICK_WINDOW_TRANSLATION_WEIGHT:-3.6}"
    pick_window_rotation_weight="${PICK_WINDOW_ROTATION_WEIGHT:-0.25}"
    pick_window_gripper_weight="${PICK_WINDOW_GRIPPER_WEIGHT:-5.2}"
    episode_prefix_enabled="${EPISODE_PREFIX_ENABLED:-true}"
    episode_prefix_after="${EPISODE_PREFIX_AFTER:-80}"
    episode_prefix_min_steps="${EPISODE_PREFIX_MIN_STEPS:-110}"
    episode_prefix_max_steps="${EPISODE_PREFIX_MAX_STEPS:-200}"
    ;;
  INSERT)
    name="${RUN_NAME:-tool_hang_closed_insert_7000}"
    num_epochs="${NUM_EPOCHS:-7000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-300}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-12}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.004}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.2}"
    rotation_weight="${ROTATION_WEIGHT:-0.9}"
    gripper_weight="${GRIPPER_WEIGHT:-1.8}"
    rotation_sigma="${ROTATION_SIGMA:-0.09}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.0}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.6}"
    gripper_timing_enabled="${GRIPPER_TIMING_ENABLED:-true}"
    min_close_steps="${MIN_CLOSE_STEPS:-4}"
    max_latch_chunks_after_close="${MAX_LATCH_CHUNKS_AFTER_CLOSE:-10}"
    hang_window_anchor="${HANG_WINDOW_ANCHOR:-closed}"
    hang_window_before="${HANG_WINDOW_BEFORE:-0}"
    hang_window_after="${HANG_WINDOW_AFTER:-0}"
    hang_window_translation_weight="${HANG_WINDOW_TRANSLATION_WEIGHT:-0.8}"
    hang_window_rotation_weight="${HANG_WINDOW_ROTATION_WEIGHT:-1.6}"
    hang_window_gripper_weight="${HANG_WINDOW_GRIPPER_WEIGHT:-0.2}"
    ;;
  FLOW)
    name="${RUN_NAME:-tool_hang_flow_curriculum_7000}"
    num_epochs="${NUM_EPOCHS:-7000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-300}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-12}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.003}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.8}"
    rotation_weight="${ROTATION_WEIGHT:-1.0}"
    gripper_weight="${GRIPPER_WEIGHT:-2.2}"
    rotation_sigma="${ROTATION_SIGMA:-0.09}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-0.8}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.6}"
    gripper_timing_enabled="${GRIPPER_TIMING_ENABLED:-true}"
    min_close_steps="${MIN_CLOSE_STEPS:-4}"
    max_latch_chunks_after_close="${MAX_LATCH_CHUNKS_AFTER_CLOSE:-10}"
    hang_window_anchor="${HANG_WINDOW_ANCHOR:-close}"
    hang_window_occurrence="${HANG_WINDOW_OCCURRENCE:-first}"
    hang_window_before="${HANG_WINDOW_BEFORE:-0}"
    hang_window_after="${HANG_WINDOW_AFTER:-8}"
    hang_window_translation_weight="${HANG_WINDOW_TRANSLATION_WEIGHT:-0.25}"
    hang_window_rotation_weight="${HANG_WINDOW_ROTATION_WEIGHT:-0.7}"
    hang_window_gripper_weight="${HANG_WINDOW_GRIPPER_WEIGHT:-0.0}"
    regrasp_window_enabled="${REGRASP_WINDOW_ENABLED:-false}"
    final_release_window_enabled="${FINAL_RELEASE_WINDOW_ENABLED:-false}"
    ;;
  PICK8)
    name="${RUN_NAME:-tool_hang_pick_guided_chunk8_6500}"
    num_epochs="${NUM_EPOCHS:-6500}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-28}"
    n_action_steps="${N_ACTION_STEPS:-8}"
    history_shift="${HISTORY_SHIFT:-8}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.012}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.8}"
    rotation_weight="${ROTATION_WEIGHT:-1.3}"
    gripper_weight="${GRIPPER_WEIGHT:-1.8}"
    rotation_sigma="${ROTATION_SIGMA:-0.12}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.0}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-1.0}"
    ;;
  HG)
    name="${RUN_NAME:-tool_hang_hg_grasp_stable_6500}"
    num_epochs="${NUM_EPOCHS:-6500}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-300}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-28}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.004}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.2}"
    rotation_weight="${ROTATION_WEIGHT:-1.2}"
    gripper_weight="${GRIPPER_WEIGHT:-1.6}"
    rotation_sigma="${ROTATION_SIGMA:-0.12}"
    release_enabled="${RELEASE_ENABLED:-false}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-3}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.0}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-1.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.5}"
    ;;
  HANG)
    name="${RUN_NAME:-tool_hang_release_guided_6500}"
    num_epochs="${NUM_EPOCHS:-6500}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-28}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.008}"
    transition_radius="${TRANSITION_RADIUS:-1}"
    translation_weight="${TRANSLATION_WEIGHT:-1.0}"
    rotation_weight="${ROTATION_WEIGHT:-1.8}"
    gripper_weight="${GRIPPER_WEIGHT:-0.8}"
    rotation_sigma="${ROTATION_SIGMA:-0.10}"
    release_enabled="${RELEASE_ENABLED:-true}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-4}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-2.2}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-2.5}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-1.0}"
    ;;
  STAGE)
    name="${RUN_NAME:-tool_hang_stage_event_anchored_6000}"
    num_epochs="${NUM_EPOCHS:-6000}"
    rollout_every="${ROLLOUT_EVERY:-100}"
    rollout_start_epoch="${ROLLOUT_START_EPOCH:-200}"
    checkpoint_every="${CHECKPOINT_EVERY:-50}"
    n_envs="${N_ENVS:-12}"
    n_action_steps="${N_ACTION_STEPS:-4}"
    history_shift="${HISTORY_SHIFT:-4}"
    phase_loss_weight="${PHASE_LOSS_WEIGHT:-0.003}"
    transition_radius="${TRANSITION_RADIUS:-2}"
    translation_weight="${TRANSLATION_WEIGHT:-1.2}"
    rotation_weight="${ROTATION_WEIGHT:-1.2}"
    gripper_weight="${GRIPPER_WEIGHT:-1.6}"
    rotation_sigma="${ROTATION_SIGMA:-0.09}"
    release_enabled="${RELEASE_ENABLED:-true}"
    release_transition_radius="${RELEASE_TRANSITION_RADIUS:-4}"
    release_translation_weight="${RELEASE_TRANSLATION_WEIGHT:-1.6}"
    release_rotation_weight="${RELEASE_ROTATION_WEIGHT:-2.0}"
    release_gripper_weight="${RELEASE_GRIPPER_WEIGHT:-0.8}"
    ;;
  *)
    echo "Usage: $0 {PICK|ONE|LIFT|INSERT|FLOW|PICK8|HG|HANG|STAGE} [hydra overrides...]" >&2
    exit 2
    ;;
esac

pick_window_enabled=false
hang_window_enabled=false
if [[ "${variant}" == "STAGE" ]]; then
  pick_window_enabled=true
  hang_window_enabled=true
elif [[ "${variant}" == "ONE" || "${variant}" == "LIFT" ]]; then
  pick_window_enabled=true
elif [[ "${variant}" == "INSERT" ]]; then
  pick_window_enabled=true
  hang_window_enabled=true
elif [[ "${variant}" == "FLOW" ]]; then
  pick_window_enabled=true
  hang_window_enabled=true
fi

# Allow staged callers to enable the same guidance windows and timing guard
# without switching the underlying policy recipe between stages.
pick_window_enabled="${PICK_WINDOW_ENABLED:-${pick_window_enabled}}"
hang_window_enabled="${HANG_WINDOW_ENABLED:-${hang_window_enabled}}"
gripper_timing_enabled="${GRIPPER_TIMING_ENABLED:-${gripper_timing_enabled:-false}}"
min_close_steps="${MIN_CLOSE_STEPS:-${min_close_steps:-2}}"
max_latch_chunks_after_close="${MAX_LATCH_CHUNKS_AFTER_CLOSE:-${max_latch_chunks_after_close:-null}}"
regrasp_window_enabled="${REGRASP_WINDOW_ENABLED:-${regrasp_window_enabled:-false}}"
final_release_window_enabled="${FINAL_RELEASE_WINDOW_ENABLED:-${final_release_window_enabled:-false}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python train.py --config-name=tool_hang_ph \
  "hydra.run.dir=data/outputs/robomimic/${name}" \
  "task.dataset_path=${dataset_path}" \
  "logging.mode=${WANDB_MODE:-disabled}" \
  "optimizer.lr=${TRAIN_LR:-0.0001}" \
  "training.resume=${TRAINING_RESUME:-false}" \
  "++training.resume_from_path=${RESUME_FROM_PATH:-null}" \
  "training.num_epochs=${num_epochs}" \
  "++training.lr_scheduler_num_epochs=${LR_SCHEDULER_NUM_EPOCHS:-${num_epochs}}" \
  "++training.lr_scheduler_start_epoch=${LR_SCHEDULER_START_EPOCH:-0}" \
  "training.rollout_every=${rollout_every}" \
  "++training.rollout_start_epoch=${rollout_start_epoch:-0}" \
  "++training.rollout_before_training=${ROLLOUT_BEFORE_TRAINING:-false}" \
  "++training.freeze_encoder_until_epoch=${FREEZE_ENCODER_UNTIL_EPOCH:-null}" \
  "++training.parameter_anchor_path=${PARAMETER_ANCHOR_PATH:-null}" \
  "++training.parameter_anchor_rate=${PARAMETER_ANCHOR_RATE:-0.0}" \
  "training.checkpoint_every=${checkpoint_every}" \
  "++task.dataset.episode_prefix_enabled=${EPISODE_PREFIX_ENABLED:-${episode_prefix_enabled:-false}}" \
  "++task.dataset.episode_prefix_anchor=${EPISODE_PREFIX_ANCHOR:-first_close}" \
  "++task.dataset.episode_prefix_close_threshold=${GRIPPER_CLOSE_THRESHOLD:-auto}" \
  "++task.dataset.episode_prefix_close_is_greater=${GRIPPER_CLOSE_IS_GREATER:-auto}" \
  "++task.dataset.episode_prefix_after=${EPISODE_PREFIX_AFTER:-${episode_prefix_after:-48}}" \
  "++task.dataset.episode_prefix_min_steps=${EPISODE_PREFIX_MIN_STEPS:-${episode_prefix_min_steps:-80}}" \
  "++task.dataset.episode_prefix_max_steps=${EPISODE_PREFIX_MAX_STEPS:-${episode_prefix_max_steps:-160}}" \
  "++task.dataset.anchor_oversample_enabled=${ANCHOR_OVERSAMPLE_ENABLED:-false}" \
  "++task.dataset.anchor_oversample_before=${ANCHOR_OVERSAMPLE_BEFORE:-48}" \
  "++task.dataset.anchor_oversample_after=${ANCHOR_OVERSAMPLE_AFTER:-8}" \
  "++task.dataset.anchor_oversample_repeats=${ANCHOR_OVERSAMPLE_REPEATS:-1}" \
  dataloader.persistent_workers=true \
  val_dataloader.persistent_workers=true \
  "n_action_steps=${n_action_steps}" \
  "task.env_runner.n_envs=${n_envs}" \
  "task.env_runner.n_train=${N_TRAIN_ROLLOUTS:-10}" \
  "task.env_runner.n_train_vis=${N_TRAIN_VIDEOS:-3}" \
  "task.env_runner.n_test=${N_TEST_ROLLOUTS:-22}" \
  "task.env_runner.n_test_vis=${N_TEST_VIDEOS:-6}" \
  "++task.env_runner.action_clip_by_dataset=${ACTION_CLIP_BY_DATASET:-true}" \
  "++task.env_runner.action_clip_margin_scale=${ACTION_CLIP_MARGIN_SCALE:-0.35}" \
  "++task.env_runner.action_clip_min_margin=${ACTION_CLIP_MIN_MARGIN:-0.05}" \
  policy.num_inference_steps=6 \
  ++policy.action_group_spectral_params.translation.indices=[0,1,2] \
  ++policy.action_group_spectral_params.translation.freq_split_high=8 \
  ++policy.action_group_spectral_params.translation.sigma=0.3 \
  ++policy.action_group_spectral_params.translation.sigma_high=0.2 \
  ++policy.action_group_spectral_params.rotation.indices=[3,4,5,6,7,8] \
  ++policy.action_group_spectral_params.rotation.freq_split_high=16 \
  "++policy.action_group_spectral_params.rotation.sigma=${rotation_sigma}" \
  "++policy.action_group_spectral_params.rotation.sigma_high=${rotation_sigma}" \
  ++policy.action_group_spectral_params.gripper.indices=[9] \
  ++policy.action_group_spectral_params.gripper.freq_split_high=16 \
  ++policy.action_group_spectral_params.gripper.sigma=0.5 \
  ++policy.action_group_spectral_params.gripper.sigma_high=0.5 \
  ++policy.action_group_history_params.enabled=true \
  "++policy.action_group_history_params.shift=${history_shift}" \
  ++policy.action_group_history_params.groups.translation.indices=[0,1,2] \
  ++policy.action_group_history_params.groups.translation.use_history=true \
  ++policy.action_group_history_params.groups.translation.update_rate=1.0 \
  ++policy.action_group_history_params.groups.rotation.indices=[3,4,5,6,7,8] \
  ++policy.action_group_history_params.groups.rotation.use_history=true \
  ++policy.action_group_history_params.groups.rotation.update_rate=1.0 \
  ++policy.action_group_history_params.groups.gripper.indices=[9] \
  ++policy.action_group_history_params.groups.gripper.use_history=false \
  ++policy.action_group_history_params.groups.gripper.update_rate=1.0 \
  "++policy.history_training_mode=${HISTORY_TRAINING_MODE:-aligned}" \
  "++policy.action_group_timing_params.gripper.enabled=${gripper_timing_enabled:-false}" \
  ++policy.action_group_timing_params.gripper.indices=[9] \
  "++policy.action_group_timing_params.gripper.close_threshold=${GRIPPER_CLOSE_THRESHOLD:-auto}" \
  "++policy.action_group_timing_params.gripper.close_is_greater=${GRIPPER_CLOSE_IS_GREATER:-auto}" \
  "++policy.action_group_timing_params.gripper.close_value=${CLOSE_VALUE:-null}" \
  "++policy.action_group_timing_params.gripper.min_close_steps=${min_close_steps:-2}" \
  "++policy.action_group_timing_params.gripper.max_latch_chunks_after_close=${max_latch_chunks_after_close:-null}" \
  "++policy.action_phase_loss_params.enabled=${PHASE_LOSS_ENABLED:-true}" \
  "++policy.action_phase_loss_params.weight=${phase_loss_weight}" \
  ++policy.action_phase_loss_params.gripper_indices=[9] \
  "++policy.action_phase_loss_params.close_threshold=${GRIPPER_CLOSE_THRESHOLD:-auto}" \
  "++policy.action_phase_loss_params.close_is_greater=${GRIPPER_CLOSE_IS_GREATER:-auto}" \
  "++policy.action_phase_loss_params.transition_radius=${transition_radius}" \
  ++policy.action_phase_loss_params.translation_indices=[0,1,2] \
  "++policy.action_phase_loss_params.translation_weight=${translation_weight}" \
  ++policy.action_phase_loss_params.rotation_indices=[3,4,5,6,7,8] \
  "++policy.action_phase_loss_params.rotation_weight=${rotation_weight}" \
  "++policy.action_phase_loss_params.gripper_weight=${gripper_weight}" \
  "++policy.action_phase_loss_params.release_enabled=${release_enabled}" \
  "++policy.action_phase_loss_params.release_transition_radius=${release_transition_radius}" \
  ++policy.action_phase_loss_params.release_translation_indices=[0,1,2] \
  "++policy.action_phase_loss_params.release_translation_weight=${release_translation_weight}" \
  ++policy.action_phase_loss_params.release_rotation_indices=[3,4,5,6,7,8] \
  "++policy.action_phase_loss_params.release_rotation_weight=${release_rotation_weight}" \
  "++policy.action_phase_loss_params.release_gripper_weight=${release_gripper_weight}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.enabled=${pick_window_enabled}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.anchor=${pick_window_anchor:-close}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.occurrence=${pick_window_occurrence:-${PICK_WINDOW_OCCURRENCE:-first}}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.before=${pick_window_before:-${PICK_WINDOW_BEFORE:-3}}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.after=${pick_window_after:-${PICK_WINDOW_AFTER:-4}}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.translation_weight=${pick_window_translation_weight:-${PICK_WINDOW_TRANSLATION_WEIGHT:-1.2}}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.rotation_weight=${pick_window_rotation_weight:-${PICK_WINDOW_ROTATION_WEIGHT:-1.8}}" \
  "++policy.action_phase_loss_params.extra_windows.pick_place.gripper_weight=${pick_window_gripper_weight:-${PICK_WINDOW_GRIPPER_WEIGHT:-1.2}}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.enabled=${hang_window_enabled}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.anchor=${hang_window_anchor:-release}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.occurrence=${hang_window_occurrence:-first}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.before=${hang_window_before:-5}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.after=${hang_window_after:-2}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.translation_weight=${hang_window_translation_weight:-2.4}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.rotation_weight=${hang_window_rotation_weight:-3.0}" \
  "++policy.action_phase_loss_params.extra_windows.hang_place.gripper_weight=${hang_window_gripper_weight:-0.6}" \
  "++policy.action_phase_loss_params.extra_windows.regrasp_insert.enabled=${regrasp_window_enabled:-false}" \
  "++policy.action_phase_loss_params.extra_windows.final_release.enabled=${final_release_window_enabled:-false}" \
  "logging.name=${name}" \
  "${extra_overrides[@]}"
