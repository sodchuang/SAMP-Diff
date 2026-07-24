#!/usr/bin/env bash
set -euo pipefail

# ToolHang staged curriculum wrapper.
#
# Usage:
#   bash scripts/run_tool_hang_curriculum.sh FULL_TASK
#   bash scripts/run_tool_hang_curriculum.sh A_GRASP
#   RESUME_FROM_PATH=data/outputs/robomimic/<A_RUN>/checkpoints/latest.ckpt \
#     bash scripts/run_tool_hang_curriculum.sh B_INSERT
#   RESUME_FROM_PATH=data/outputs/robomimic/<B_RUN>/checkpoints/latest.ckpt \
#     bash scripts/run_tool_hang_curriculum.sh C_RELEASE
#
# This script provides two modes:
#   FULL_TASK: one run for the complete task, with soft full-task guidance.
#   A/B/C    : manual gated curriculum if FULL_TASK collapses.
#
# Manual gated curriculum:
#   A_GRASP  : learn stable first grasp / lift tendency.
#   B_INSERT : resume from A, add soft first-insert guidance while preserving grasp.
#   C_RELEASE: resume from B, add final release / settle guidance.
#
# Keep the Hydra deletions at the end if the remote env_runner does not include
# dataset action clipping arguments yet.

stage="${1:-A_GRASP}"
if [[ $# -gt 0 ]]; then
  shift
fi

common_overrides=(
  '~task.env_runner.action_clip_by_dataset'
  '~task.env_runner.action_clip_margin_scale'
  '~task.env_runner.action_clip_min_margin'
)

case "${stage}" in
  FULL_TASK)
    RUN_NAME="${RUN_NAME:-tool_hang_full_task_soft_curriculum_v1}" \
    NUM_EPOCHS="${NUM_EPOCHS:-7000}" \
    ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-300}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
    N_ENVS="${N_ENVS:-12}" \
    PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.0025}" \
    TRANSLATION_WEIGHT="${TRANSLATION_WEIGHT:-1.6}" \
    ROTATION_WEIGHT="${ROTATION_WEIGHT:-0.95}" \
    GRIPPER_WEIGHT="${GRIPPER_WEIGHT:-2.3}" \
    HANG_WINDOW_ANCHOR="${HANG_WINDOW_ANCHOR:-close}" \
    HANG_WINDOW_OCCURRENCE="${HANG_WINDOW_OCCURRENCE:-first}" \
    HANG_WINDOW_BEFORE="${HANG_WINDOW_BEFORE:-0}" \
    HANG_WINDOW_AFTER="${HANG_WINDOW_AFTER:-8}" \
    HANG_WINDOW_TRANSLATION_WEIGHT="${HANG_WINDOW_TRANSLATION_WEIGHT:-0.12}" \
    HANG_WINDOW_ROTATION_WEIGHT="${HANG_WINDOW_ROTATION_WEIGHT:-0.35}" \
    HANG_WINDOW_GRIPPER_WEIGHT="${HANG_WINDOW_GRIPPER_WEIGHT:-0.0}" \
    REGRASP_WINDOW_ENABLED=true \
    FINAL_RELEASE_WINDOW_ENABLED=true \
    bash scripts/run_tool_hang_rescue.sh FLOW \
      "${common_overrides[@]}" \
      "++policy.action_phase_loss_params.extra_windows.regrasp_insert.translation_weight=${REGRASP_TRANSLATION_WEIGHT:-0.08}" \
      "++policy.action_phase_loss_params.extra_windows.regrasp_insert.rotation_weight=${REGRASP_ROTATION_WEIGHT:-0.18}" \
      "++policy.action_phase_loss_params.extra_windows.regrasp_insert.gripper_weight=${REGRASP_GRIPPER_WEIGHT:-0.05}" \
      "++policy.action_phase_loss_params.extra_windows.final_release.translation_weight=${FINAL_RELEASE_TRANSLATION_WEIGHT:-0.06}" \
      "++policy.action_phase_loss_params.extra_windows.final_release.rotation_weight=${FINAL_RELEASE_ROTATION_WEIGHT:-0.12}" \
      "++policy.action_phase_loss_params.extra_windows.final_release.gripper_weight=${FINAL_RELEASE_GRIPPER_WEIGHT:-0.08}" \
      "$@"
    ;;

  A_GRASP)
    RUN_NAME="${RUN_NAME:-tool_hang_A_grasp_first_v1}" \
    NUM_EPOCHS="${NUM_EPOCHS:-7000}" \
    ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-300}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
    N_ENVS="${N_ENVS:-12}" \
    PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.006}" \
    TRANSLATION_WEIGHT="${TRANSLATION_WEIGHT:-1.6}" \
    ROTATION_WEIGHT="${ROTATION_WEIGHT:-0.9}" \
    GRIPPER_WEIGHT="${GRIPPER_WEIGHT:-2.4}" \
    RELEASE_ENABLED=false \
    bash scripts/run_tool_hang_rescue.sh ONE "${common_overrides[@]}" "$@"
    ;;

  B_INSERT)
    if [[ -z "${RESUME_FROM_PATH:-}" ]]; then
      echo "[ERROR] B_INSERT requires RESUME_FROM_PATH from a passed A_GRASP checkpoint." >&2
      exit 2
    fi
    RUN_NAME="${RUN_NAME:-tool_hang_B_insert_after_grasp_v1}" \
    NUM_EPOCHS="${NUM_EPOCHS:-7000}" \
    ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-300}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
    N_ENVS="${N_ENVS:-12}" \
    PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.003}" \
    TRANSLATION_WEIGHT="${TRANSLATION_WEIGHT:-1.8}" \
    ROTATION_WEIGHT="${ROTATION_WEIGHT:-1.0}" \
    GRIPPER_WEIGHT="${GRIPPER_WEIGHT:-2.2}" \
    HANG_WINDOW_ANCHOR="${HANG_WINDOW_ANCHOR:-close}" \
    HANG_WINDOW_OCCURRENCE="${HANG_WINDOW_OCCURRENCE:-first}" \
    HANG_WINDOW_BEFORE="${HANG_WINDOW_BEFORE:-0}" \
    HANG_WINDOW_AFTER="${HANG_WINDOW_AFTER:-8}" \
    HANG_WINDOW_TRANSLATION_WEIGHT="${HANG_WINDOW_TRANSLATION_WEIGHT:-0.18}" \
    HANG_WINDOW_ROTATION_WEIGHT="${HANG_WINDOW_ROTATION_WEIGHT:-0.55}" \
    HANG_WINDOW_GRIPPER_WEIGHT="${HANG_WINDOW_GRIPPER_WEIGHT:-0.0}" \
    REGRASP_WINDOW_ENABLED=false \
    FINAL_RELEASE_WINDOW_ENABLED=false \
    bash scripts/run_tool_hang_rescue.sh FLOW "${common_overrides[@]}" "$@"
    ;;

  C_RELEASE)
    if [[ -z "${RESUME_FROM_PATH:-}" ]]; then
      echo "[ERROR] C_RELEASE requires RESUME_FROM_PATH from a passed B_INSERT checkpoint." >&2
      exit 2
    fi
    RUN_NAME="${RUN_NAME:-tool_hang_C_release_after_insert_v1}" \
    NUM_EPOCHS="${NUM_EPOCHS:-7000}" \
    ROLLOUT_START_EPOCH="${ROLLOUT_START_EPOCH:-300}" \
    ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
    N_ENVS="${N_ENVS:-12}" \
    PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.0025}" \
    TRANSLATION_WEIGHT="${TRANSLATION_WEIGHT:-1.5}" \
    ROTATION_WEIGHT="${ROTATION_WEIGHT:-1.0}" \
    GRIPPER_WEIGHT="${GRIPPER_WEIGHT:-1.8}" \
    HANG_WINDOW_ANCHOR="${HANG_WINDOW_ANCHOR:-close}" \
    HANG_WINDOW_OCCURRENCE="${HANG_WINDOW_OCCURRENCE:-first}" \
    HANG_WINDOW_BEFORE="${HANG_WINDOW_BEFORE:-0}" \
    HANG_WINDOW_AFTER="${HANG_WINDOW_AFTER:-8}" \
    HANG_WINDOW_TRANSLATION_WEIGHT="${HANG_WINDOW_TRANSLATION_WEIGHT:-0.18}" \
    HANG_WINDOW_ROTATION_WEIGHT="${HANG_WINDOW_ROTATION_WEIGHT:-0.55}" \
    HANG_WINDOW_GRIPPER_WEIGHT="${HANG_WINDOW_GRIPPER_WEIGHT:-0.0}" \
    FINAL_RELEASE_WINDOW_ENABLED=true \
    REGRASP_WINDOW_ENABLED=false \
    bash scripts/run_tool_hang_rescue.sh FLOW "${common_overrides[@]}" "$@"
    ;;

  *)
    echo "Usage: $0 {FULL_TASK|A_GRASP|B_INSERT|C_RELEASE} [hydra overrides...]" >&2
    exit 2
    ;;
esac
