#!/usr/bin/env bash
# Train the complete MimicGen Stack_D1 delta-action task from epoch 0.
# HG: reuse translation/rotation history and regenerate gripper actions.
# GC fix: supervise both grasp-close and stack-release transitions.

set -euo pipefail

DATASET="${DATASET:-data/mimicgen/core/stack_d1.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
RUN_NAME="${RUN_NAME:-stack_d1_delta_release_balance_v3_seed42}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-4000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-8}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# Stack-specific semantic curriculum inside one end-to-end policy.  These are
# environment variables so the recipe can be tuned without editing the policy.
N_ACTION_STEPS="${N_ACTION_STEPS:-2}"
INFERENCE_STEPS="${INFERENCE_STEPS:-8}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.14}"
PRE_RELEASE_BEFORE="${PRE_RELEASE_BEFORE:-16}"
POST_RELEASE_AFTER="${POST_RELEASE_AFTER:-16}"
BASE_GRIPPER_WEIGHT="${BASE_GRIPPER_WEIGHT:-1.25}"
RELEASE_GRIPPER_WEIGHT="${RELEASE_GRIPPER_WEIGHT:-8.0}"
POST_RELEASE_GRIPPER_WEIGHT="${POST_RELEASE_GRIPPER_WEIGHT:-8.0}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] missing Stack_D1 dataset: ${DATASET}" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}/checkpoints/latest.ckpt" ]]; then
    echo "[ERROR] ${OUTPUT_DIR} already has a checkpoint." >&2
    echo "Use a new RUN_NAME; this launcher intentionally starts from epoch 0." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

python - "${DATASET}" <<'PY'
import h5py
import json
import sys

with h5py.File(sys.argv[1], "r") as f:
    demos = f["data"]
    demo = demos["demo_0"]
    raw = demos.attrs.get("env_args", f.attrs.get("env_args"))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    args = json.loads(raw)
    if args["env_name"] != "Stack_D1":
        raise SystemExit(f"[ERROR] expected Stack_D1, got {args['env_name']}")
    if demo["actions"].shape[-1] != 7:
        raise SystemExit("[ERROR] Stack delta actions must have dimension 7")
    controller = args.get("env_kwargs", {}).get("controller_configs", {})
    if controller.get("control_delta", True) is False:
        raise SystemExit("[ERROR] dataset is not delta_eef_pose")
    print(f"[OK] Stack_D1 delta dataset: demos={len(demos)} action_dim=7")
PY

echo "[STACK-FIX] from_scratch=true epochs=0..${NUM_EPOCHS}"
echo "[STACK-FIX] control=delta_eef_pose groups=xyz[0,1,2],rot[3,4,5],gripper[6]"
echo "[STACK-FIX] semantic windows=grasp_hold,pre_release_align,release,post_release_retreat"
echo "[STACK-FIX] n_action_steps=${N_ACTION_STEPS} inference_steps=${INFERENCE_STEPS}"
echo "[STACK-FIX] gripper base=${BASE_GRIPPER_WEIGHT} release=${RELEASE_GRIPPER_WEIGHT} post_release=${POST_RELEASE_GRIPPER_WEIGHT}"
echo "[STACK-FIX] output=${OUTPUT_DIR}"

python train.py --config-name=mimicgen_single \
    "hydra.run.dir=${OUTPUT_DIR}" \
    obs_dim=32 \
    task_semantics.task_name=stack_d1 \
    task.dataset_path="${DATASET}" \
    task.abs_action=false \
    task.dataset.abs_action=false \
    task.env_runner.abs_action=false \
    training.seed="${SEED}" \
    data_split.seed="${SEED}" \
    data_split.max_train_episodes=1000 \
    training.resume=false \
    ++training.resume_from_path=null \
    training.num_epochs="$((NUM_EPOCHS + 1))" \
    training.rollout_every="${ROLLOUT_EVERY}" \
    training.checkpoint_every="${CHECKPOINT_EVERY}" \
    task.env_runner.max_steps=400 \
    task.env_runner.n_envs="${N_ENVS}" \
    task.env_runner.n_test="${N_TEST}" \
    task.env_runner.n_test_vis="${N_TEST_VIS}" \
    ++task.env_runner.stack_release_hold_steps=5 \
    ++task.env_runner.stack_min_eef_distance=0.04 \
    ++task.env_runner.stack_min_gripper_open_width=0.04 \
    logging.mode="${WANDB_MODE}" \
    logging.name="${RUN_NAME}" \
    n_action_steps="${N_ACTION_STEPS}" \
    policy.num_inference_steps="${INFERENCE_STEPS}" \
    policy.sigma=0.14 \
    policy.cold_start_prob=0.20 \
    policy.freq_split_low=0 \
    policy.freq_split_high=8 \
    policy.sigma_high=0.08 \
    policy.action_group_spectral_params.translation.freq_split_high=8 \
    policy.action_group_spectral_params.translation.sigma=0.14 \
    policy.action_group_spectral_params.translation.sigma_high=0.08 \
    policy.action_group_spectral_params.rotation.freq_split_high=12 \
    policy.action_group_spectral_params.rotation.sigma=0.08 \
    policy.action_group_spectral_params.rotation.sigma_high=0.06 \
    policy.action_group_spectral_params.gripper.freq_split_high=16 \
    policy.action_group_spectral_params.gripper.sigma=0.50 \
    policy.action_group_spectral_params.gripper.sigma_high=0.50 \
    policy.action_group_loss_weights.translation.weight=1.0 \
    policy.action_group_loss_weights.rotation.weight=1.0 \
    policy.action_group_loss_weights.gripper.weight="${BASE_GRIPPER_WEIGHT}" \
    policy.action_group_history_params.enabled=true \
    policy.action_group_history_params.shift="${N_ACTION_STEPS}" \
    policy.action_group_history_params.groups.translation.use_history=true \
    policy.action_group_history_params.groups.translation.update_rate=1.0 \
    policy.action_group_history_params.groups.rotation.use_history=true \
    policy.action_group_history_params.groups.rotation.update_rate=1.0 \
    policy.action_group_history_params.groups.gripper.use_history=false \
    policy.action_group_history_params.groups.gripper.update_rate=1.0 \
    ++policy.history_training_mode=aligned \
    policy.action_group_timing_params.gripper.enabled=false \
    policy.action_phase_loss_params.enabled=true \
    policy.action_phase_loss_params.weight="${PHASE_LOSS_WEIGHT}" \
    policy.action_phase_loss_params.close_threshold=0.0 \
    policy.action_phase_loss_params.close_is_greater=true \
    policy.action_phase_loss_params.transition_radius=2 \
    policy.action_phase_loss_params.translation_weight=1.3 \
    policy.action_phase_loss_params.rotation_weight=1.1 \
    policy.action_phase_loss_params.gripper_weight=1.8 \
    policy.action_phase_loss_params.release_enabled=true \
    policy.action_phase_loss_params.release_transition_radius=5 \
    policy.action_phase_loss_params.release_translation_weight=2.5 \
    policy.action_phase_loss_params.release_rotation_weight=1.5 \
    policy.action_phase_loss_params.release_gripper_weight="${RELEASE_GRIPPER_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.anchor=close \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.occurrence=first \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.before=0 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.after=6 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.translation_weight=0.10 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.rotation_weight=0.10 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.gripper_weight=0.60 \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.anchor=release \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.occurrence=last \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.before="${PRE_RELEASE_BEFORE}" \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.after=0 \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.translation_weight=3.0 \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.rotation_weight=1.8 \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.gripper_weight=0.8 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.anchor=release \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.occurrence=last \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.before=0 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.after="${POST_RELEASE_AFTER}" \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.translation_weight=2.5 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.rotation_weight=1.2 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.gripper_weight="${POST_RELEASE_GRIPPER_WEIGHT}" \
    ++task.env_runner.action_clip_by_dataset=true \
    ++task.env_runner.action_clip_margin_scale=0.10 \
    ++task.env_runner.action_clip_min_margin=0.02 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
