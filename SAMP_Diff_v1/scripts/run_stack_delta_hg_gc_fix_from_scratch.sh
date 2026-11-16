#!/usr/bin/env bash
# Train the complete MimicGen Stack_D1 delta-action task from epoch 0.
# HG: reuse translation/rotation history and regenerate gripper actions.
# GC fix: supervise both grasp-close and stack-release transitions.

set -euo pipefail

DATASET="${DATASET:-data/mimicgen/core/stack_d1.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
RUN_NAME="${RUN_NAME:-stack_d1_delta_semantic_gate_v4_seed42}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-3200}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-30}"
N_TEST_VIS="${N_TEST_VIS:-10}"
WANDB_MODE="${WANDB_MODE:-disabled}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_EXISTING="${RESUME_EXISTING:-false}"

# Stack-specific semantic curriculum inside one end-to-end policy.  These are
# environment variables so the recipe can be tuned without editing the policy.
N_ACTION_STEPS="${N_ACTION_STEPS:-2}"
INFERENCE_STEPS="${INFERENCE_STEPS:-8}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.05}"
PHASE_LOSS_ENABLED="${PHASE_LOSS_ENABLED:-true}"
EVENT_OVERSAMPLE_ENABLED="${EVENT_OVERSAMPLE_ENABLED:-true}"
GRIPPER_USE_HISTORY="${GRIPPER_USE_HISTORY:-false}"
HISTORY_ENABLED="${HISTORY_ENABLED:-true}"
HISTORY_TRAINING_MODE="${HISTORY_TRAINING_MODE:-aligned}"
TRANSLATION_USE_HISTORY="${TRANSLATION_USE_HISTORY:-true}"
ROTATION_USE_HISTORY="${ROTATION_USE_HISTORY:-true}"
TRANSLATION_HISTORY_UPDATE_RATE="${TRANSLATION_HISTORY_UPDATE_RATE:-1.0}"
ROTATION_HISTORY_UPDATE_RATE="${ROTATION_HISTORY_UPDATE_RATE:-1.0}"
COLD_START_PROB="${COLD_START_PROB:-0.15}"
TRANSLATION_FREQ_SPLIT_HIGH="${TRANSLATION_FREQ_SPLIT_HIGH:-8}"
TRANSLATION_SIGMA="${TRANSLATION_SIGMA:-0.08}"
TRANSLATION_SIGMA_HIGH="${TRANSLATION_SIGMA_HIGH:-0.03}"
ROTATION_FREQ_SPLIT_HIGH="${ROTATION_FREQ_SPLIT_HIGH:-12}"
ROTATION_SIGMA="${ROTATION_SIGMA:-0.05}"
ROTATION_SIGMA_HIGH="${ROTATION_SIGMA_HIGH:-0.03}"
GRIPPER_FREQ_SPLIT_HIGH="${GRIPPER_FREQ_SPLIT_HIGH:-16}"
GRIPPER_SIGMA="${GRIPPER_SIGMA:-0.30}"
GRIPPER_SIGMA_HIGH="${GRIPPER_SIGMA_HIGH:-0.20}"
TRANSLATION_LOSS_WEIGHT="${TRANSLATION_LOSS_WEIGHT:-1.8}"
ROTATION_LOSS_WEIGHT="${ROTATION_LOSS_WEIGHT:-1.6}"
GRASP_HOLD_AFTER="${GRASP_HOLD_AFTER:-10}"
PRE_RELEASE_BEFORE="${PRE_RELEASE_BEFORE:-24}"
SETTLE_BEFORE="${SETTLE_BEFORE:-8}"
SETTLE_AFTER="${SETTLE_AFTER:-3}"
RELEASE_RADIUS="${RELEASE_RADIUS:-3}"
POST_RELEASE_AFTER="${POST_RELEASE_AFTER:-24}"
BASE_GRIPPER_WEIGHT="${BASE_GRIPPER_WEIGHT:-1.60}"
GRASP_HOLD_GRIPPER_WEIGHT="${GRASP_HOLD_GRIPPER_WEIGHT:-1.40}"
RELEASE_GRIPPER_WEIGHT="${RELEASE_GRIPPER_WEIGHT:-3.0}"
POST_RELEASE_GRIPPER_WEIGHT="${POST_RELEASE_GRIPPER_WEIGHT:-4.0}"
PRE_RELEASE_TRANSLATION_WEIGHT="${PRE_RELEASE_TRANSLATION_WEIGHT:-5.0}"
PRE_RELEASE_ROTATION_WEIGHT="${PRE_RELEASE_ROTATION_WEIGHT:-3.0}"
SETTLE_TRANSLATION_WEIGHT="${SETTLE_TRANSLATION_WEIGHT:-4.0}"
SETTLE_ROTATION_WEIGHT="${SETTLE_ROTATION_WEIGHT:-2.5}"
POST_RELEASE_TRANSLATION_WEIGHT="${POST_RELEASE_TRANSLATION_WEIGHT:-1.8}"
POST_RELEASE_ROTATION_WEIGHT="${POST_RELEASE_ROTATION_WEIGHT:-1.0}"
EVENT_OVERSAMPLE_BEFORE="${EVENT_OVERSAMPLE_BEFORE:-24}"
EVENT_OVERSAMPLE_AFTER="${EVENT_OVERSAMPLE_AFTER:-16}"
EVENT_OVERSAMPLE_REPEATS="${EVENT_OVERSAMPLE_REPEATS:-3}"
# robosuite versions concatenate the 23-D Stack object observation in
# different orders. "auto" validates known layouts against expert releases.
STACK_EVENT_ANCHOR="${STACK_EVENT_ANCHOR:-first_positive_reward}"
STACK_RELEASE_GATE_INDICES="${STACK_RELEASE_GATE_INDICES:-auto}"
STACK_RELEASE_GATE_MIN="${STACK_RELEASE_GATE_MIN:-[-0.035,-0.035,-0.080]}"
STACK_RELEASE_GATE_MAX="${STACK_RELEASE_GATE_MAX:-[0.035,0.035,-0.015]}"
STACK_RELEASE_MIN_ANCHOR_RATIO="${STACK_RELEASE_MIN_ANCHOR_RATIO:-0.98}"
STACK_RELEASE_GATE_RADIUS="${STACK_RELEASE_GATE_RADIUS:-3}"
STACK_RELEASE_GATE_ENABLED="${STACK_RELEASE_GATE_ENABLED:-auto}"

if [[ "${STACK_RELEASE_GATE_ENABLED}" == "auto" ]]; then
    if [[ "${STACK_EVENT_ANCHOR}" == "last_release" ]]; then
        STACK_RELEASE_GATE_ENABLED=true
    else
        # Reward anchors are already conditioned on physical task success and
        # do not consume the object-observation gate.
        STACK_RELEASE_GATE_ENABLED=false
    fi
fi
if [[ "${STACK_RELEASE_GATE_ENABLED}" != "true" \
   && "${STACK_RELEASE_GATE_ENABLED}" != "false" ]]; then
    echo "[ERROR] STACK_RELEASE_GATE_ENABLED must be auto, true, or false." >&2
    exit 1
fi

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Each remote H100 environment exposes its own GPU as physical device 0.
# Use GPU_ID to override explicitly; do not inherit a stale CUDA_VISIBLE_DEVICES=1.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - "${DEVICE}" <<'PY'
import sys
import torch

device = sys.argv[1]
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] CUDA is unavailable. This standalone H100 environment must use "
        "GPU_ID=0 DEVICE=cuda:0."
    )
if device.startswith("cuda"):
    print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
PY

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] missing Stack_D1 dataset: ${DATASET}" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}/checkpoints/latest.ckpt" ]]; then
    if [[ "${RESUME_EXISTING}" == "true" ]]; then
        TRAINING_RESUME=true
        echo "[STACK-FIX] resuming existing run: ${OUTPUT_DIR}"
    else
        echo "[ERROR] ${OUTPUT_DIR} already has a checkpoint." >&2
        echo "Use a new RUN_NAME, or set RESUME_EXISTING=true." >&2
        exit 1
    fi
else
    TRAINING_RESUME=false
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

if [[ "${STACK_EVENT_ANCHOR}" == "last_release" && "${STACK_RELEASE_GATE_INDICES}" == "auto" ]]; then
    if ! python scripts/detect_stack_release_gate.py --help 2>&1 \
        | grep -q -- '--radius'; then
        echo "[ERROR] scripts/detect_stack_release_gate.py is outdated." >&2
        echo "Update the v5 detector together with both Stack shell scripts." >&2
        exit 1
    fi
    STACK_RELEASE_GATE_INDICES="$(
        python scripts/detect_stack_release_gate.py \
            "${DATASET}" \
            --min-coverage "${STACK_RELEASE_MIN_ANCHOR_RATIO}" \
            --radius "${STACK_RELEASE_GATE_RADIUS}"
    )"
fi

echo "[STACK-FIX] from_scratch=true epochs=0..${NUM_EPOCHS}"
echo "[STACK-FIX] control=delta_eef_pose groups=xyz[0,1,2],rot[3,4,5],gripper[6]"
echo "[STACK-FIX] semantic windows=grasp_hold,pre_release_align,contact_settle,release,post_release_retreat"
echo "[STACK-FIX] n_action_steps=${N_ACTION_STEPS} inference_steps=${INFERENCE_STEPS}"
echo "[STACK-FIX] phase_weight=${PHASE_LOSS_WEIGHT} hold_after=${GRASP_HOLD_AFTER} align_before=${PRE_RELEASE_BEFORE} settle=${SETTLE_BEFORE}+${SETTLE_AFTER} release_radius=${RELEASE_RADIUS}"
echo "[STACK-FIX] event_sampling=${STACK_EVENT_ANCHOR} before=${EVENT_OVERSAMPLE_BEFORE} after=${EVENT_OVERSAMPLE_AFTER} repeats=${EVENT_OVERSAMPLE_REPEATS}"
if [[ "${STACK_EVENT_ANCHOR}" == "last_release" ]]; then
    echo "[STACK-FIX] release_gate enabled=${STACK_RELEASE_GATE_ENABLED} indices=${STACK_RELEASE_GATE_INDICES} min=${STACK_RELEASE_GATE_MIN} max=${STACK_RELEASE_GATE_MAX}"
    echo "[STACK-FIX] release_gate temporal_radius=+/-${STACK_RELEASE_GATE_RADIUS} steps"
else
    echo "[STACK-FIX] release_gate=dataset_reward (physical task success)"
fi
echo "[STACK-FIX] gripper_semantics=auto_from_next_observed_finger_width"
echo "[STACK-FIX] history enabled=${HISTORY_ENABLED} mode=${HISTORY_TRAINING_MODE} translation=${TRANSLATION_USE_HISTORY}@${TRANSLATION_HISTORY_UPDATE_RATE} rotation=${ROTATION_USE_HISTORY}@${ROTATION_HISTORY_UPDATE_RATE} gripper=${GRIPPER_USE_HISTORY}"
echo "[STACK-FIX] group_prior translation=f${TRANSLATION_FREQ_SPLIT_HIGH}/s${TRANSLATION_SIGMA}/${TRANSLATION_SIGMA_HIGH} rotation=f${ROTATION_FREQ_SPLIT_HIGH}/s${ROTATION_SIGMA}/${ROTATION_SIGMA_HIGH} gripper=f${GRIPPER_FREQ_SPLIT_HIGH}/s${GRIPPER_SIGMA}/${GRIPPER_SIGMA_HIGH}"
echo "[STACK-FIX] gripper base=${BASE_GRIPPER_WEIGHT} grasp_hold=${GRASP_HOLD_GRIPPER_WEIGHT} release=${RELEASE_GRIPPER_WEIGHT} post_release=${POST_RELEASE_GRIPPER_WEIGHT}"
echo "[STACK-FIX] output=${OUTPUT_DIR}"

python train.py --config-name=mimicgen_single \
    "hydra.run.dir=${OUTPUT_DIR}" \
    obs_dim=32 \
    task_semantics.task_name=stack_d1 \
    task.dataset_path="${DATASET}" \
    task.abs_action=false \
    task.dataset.abs_action=false \
    ++task.dataset.episode_prefix_close_threshold=auto \
    ++task.dataset.episode_prefix_close_is_greater=auto \
    ++task.dataset.event_oversample_enabled="${EVENT_OVERSAMPLE_ENABLED}" \
    ++task.dataset.event_oversample_anchor="${STACK_EVENT_ANCHOR}" \
    ++task.dataset.event_oversample_before="${EVENT_OVERSAMPLE_BEFORE}" \
    ++task.dataset.event_oversample_after="${EVENT_OVERSAMPLE_AFTER}" \
    ++task.dataset.event_oversample_repeats="${EVENT_OVERSAMPLE_REPEATS}" \
    ++task.dataset.event_oversample_min_anchor_ratio="${STACK_RELEASE_MIN_ANCHOR_RATIO}" \
    ++task.dataset.event_oversample_observation_gate_enabled="${STACK_RELEASE_GATE_ENABLED}" \
    ++task.dataset.event_oversample_observation_gate_indices="${STACK_RELEASE_GATE_INDICES}" \
    ++task.dataset.event_oversample_observation_gate_min="${STACK_RELEASE_GATE_MIN}" \
    ++task.dataset.event_oversample_observation_gate_max="${STACK_RELEASE_GATE_MAX}" \
    ++task.dataset.event_oversample_observation_gate_radius="${STACK_RELEASE_GATE_RADIUS}" \
    task.env_runner.abs_action=false \
    training.device="${DEVICE}" \
    training.seed="${SEED}" \
    data_split.seed="${SEED}" \
    data_split.max_train_episodes=1000 \
    training.resume="${TRAINING_RESUME}" \
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
    policy.sigma=0.08 \
    policy.cold_start_prob="${COLD_START_PROB}" \
    policy.freq_split_low=0 \
    policy.freq_split_high=8 \
    policy.sigma_high=0.03 \
    policy.action_group_spectral_params.translation.freq_split_high="${TRANSLATION_FREQ_SPLIT_HIGH}" \
    policy.action_group_spectral_params.translation.sigma="${TRANSLATION_SIGMA}" \
    policy.action_group_spectral_params.translation.sigma_high="${TRANSLATION_SIGMA_HIGH}" \
    policy.action_group_spectral_params.rotation.freq_split_high="${ROTATION_FREQ_SPLIT_HIGH}" \
    policy.action_group_spectral_params.rotation.sigma="${ROTATION_SIGMA}" \
    policy.action_group_spectral_params.rotation.sigma_high="${ROTATION_SIGMA_HIGH}" \
    policy.action_group_spectral_params.gripper.freq_split_high="${GRIPPER_FREQ_SPLIT_HIGH}" \
    policy.action_group_spectral_params.gripper.sigma="${GRIPPER_SIGMA}" \
    policy.action_group_spectral_params.gripper.sigma_high="${GRIPPER_SIGMA_HIGH}" \
    policy.action_group_loss_weights.translation.weight="${TRANSLATION_LOSS_WEIGHT}" \
    policy.action_group_loss_weights.rotation.weight="${ROTATION_LOSS_WEIGHT}" \
    policy.action_group_loss_weights.gripper.weight="${BASE_GRIPPER_WEIGHT}" \
    policy.action_group_history_params.enabled="${HISTORY_ENABLED}" \
    policy.action_group_history_params.shift="${N_ACTION_STEPS}" \
    policy.action_group_history_params.groups.translation.use_history="${TRANSLATION_USE_HISTORY}" \
    policy.action_group_history_params.groups.translation.update_rate="${TRANSLATION_HISTORY_UPDATE_RATE}" \
    policy.action_group_history_params.groups.rotation.use_history="${ROTATION_USE_HISTORY}" \
    policy.action_group_history_params.groups.rotation.update_rate="${ROTATION_HISTORY_UPDATE_RATE}" \
    policy.action_group_history_params.groups.gripper.use_history="${GRIPPER_USE_HISTORY}" \
    policy.action_group_history_params.groups.gripper.update_rate=1.0 \
    ++policy.history_training_mode="${HISTORY_TRAINING_MODE}" \
    policy.action_group_timing_params.gripper.enabled=false \
    policy.action_phase_loss_params.enabled="${PHASE_LOSS_ENABLED}" \
    policy.action_phase_loss_params.weight="${PHASE_LOSS_WEIGHT}" \
    policy.action_phase_loss_params.close_threshold=0.0 \
    policy.action_phase_loss_params.close_is_greater=true \
    policy.action_phase_loss_params.transition_radius=2 \
    policy.action_phase_loss_params.translation_weight=1.8 \
    policy.action_phase_loss_params.rotation_weight=1.5 \
    policy.action_phase_loss_params.gripper_weight=1.8 \
    policy.action_phase_loss_params.release_enabled=true \
    policy.action_phase_loss_params.release_transition_radius="${RELEASE_RADIUS}" \
    policy.action_phase_loss_params.release_translation_weight=3.5 \
    policy.action_phase_loss_params.release_rotation_weight=2.5 \
    policy.action_phase_loss_params.release_gripper_weight="${RELEASE_GRIPPER_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.anchor=close \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.occurrence=first \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.before=0 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.after="${GRASP_HOLD_AFTER}" \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.translation_weight=0.30 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.rotation_weight=0.30 \
    ++policy.action_phase_loss_params.extra_windows.grasp_hold.gripper_weight="${GRASP_HOLD_GRIPPER_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.anchor=release \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.occurrence=last \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.before="${PRE_RELEASE_BEFORE}" \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.after=0 \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.translation_weight="${PRE_RELEASE_TRANSLATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.rotation_weight="${PRE_RELEASE_ROTATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.pre_release_align.gripper_weight=1.2 \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.anchor=release \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.occurrence=last \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.before="${SETTLE_BEFORE}" \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.after="${SETTLE_AFTER}" \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.translation_weight="${SETTLE_TRANSLATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.rotation_weight="${SETTLE_ROTATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.contact_settle.gripper_weight=2.0 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.enabled=true \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.anchor=release \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.occurrence=last \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.before=0 \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.after="${POST_RELEASE_AFTER}" \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.translation_weight="${POST_RELEASE_TRANSLATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.rotation_weight="${POST_RELEASE_ROTATION_WEIGHT}" \
    ++policy.action_phase_loss_params.extra_windows.post_release_retreat.gripper_weight="${POST_RELEASE_GRIPPER_WEIGHT}" \
    ++task.env_runner.action_clip_by_dataset=true \
    ++task.env_runner.action_clip_margin_scale=0.10 \
    ++task.env_runner.action_clip_min_margin=0.02 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
