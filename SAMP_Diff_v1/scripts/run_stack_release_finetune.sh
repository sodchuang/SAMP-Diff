#!/usr/bin/env bash
# Continue the verified v3.7 epoch-700 Stack_D1 policy. Keep its learned grasp,
# transport and placement trajectory; fine-tune mainly the gripper transition
# and post-release retreat. Do not overweight the downward placement motion.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
DATASET="${DATASET:-data/mimicgen/core/stack_d1.hdf5}"
SOURCE_RUN="${SOURCE_RUN:-}"
SOURCE_CKPT="${SOURCE_CKPT:-}"
RUN_NAME="${RUN_NAME:-stack_d1_safe_release_v3_9_ft700_seed42}"
SEED="${SEED:-42}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-800}"
FINETUNE_LR="${FINETUNE_LR:-3e-6}"
MIN_SOURCE_EPOCH="${MIN_SOURCE_EPOCH:-700}"
EXPECTED_SOURCE_EPOCH="${EXPECTED_SOURCE_EPOCH:-700}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-30}"
N_TEST_VIS="${N_TEST_VIS:-10}"
WANDB_MODE="${WANDB_MODE:-disabled}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"

# Release-specific supervision. Pose weights stay conservative so an already
# learned grasp / transport / alignment trajectory is not overwritten.
PHASE_WEIGHT="${PHASE_WEIGHT:-0.015}"
TRANSLATION_LOSS_WEIGHT="${TRANSLATION_LOSS_WEIGHT:-1.8}"
ROTATION_LOSS_WEIGHT="${ROTATION_LOSS_WEIGHT:-1.6}"
GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-1.6}"
PRE_RELEASE_BEFORE="${PRE_RELEASE_BEFORE:-8}"
PRE_RELEASE_TRANSLATION_WEIGHT="${PRE_RELEASE_TRANSLATION_WEIGHT:-1.0}"
PRE_RELEASE_ROTATION_WEIGHT="${PRE_RELEASE_ROTATION_WEIGHT:-1.0}"
SETTLE_BEFORE="${SETTLE_BEFORE:-3}"
SETTLE_AFTER="${SETTLE_AFTER:-2}"
SETTLE_TRANSLATION_WEIGHT="${SETTLE_TRANSLATION_WEIGHT:-0.5}"
SETTLE_ROTATION_WEIGHT="${SETTLE_ROTATION_WEIGHT:-0.5}"
RELEASE_RADIUS="${RELEASE_RADIUS:-2}"
RELEASE_TRANSLATION_WEIGHT="${RELEASE_TRANSLATION_WEIGHT:-0.20}"
RELEASE_ROTATION_WEIGHT="${RELEASE_ROTATION_WEIGHT:-0.10}"
RELEASE_GRIPPER_WEIGHT="${RELEASE_GRIPPER_WEIGHT:-8.0}"
PRE_RELEASE_GRIPPER_WEIGHT="${PRE_RELEASE_GRIPPER_WEIGHT:-0.3}"
SETTLE_GRIPPER_WEIGHT="${SETTLE_GRIPPER_WEIGHT:-1.5}"
POST_RELEASE_AFTER="${POST_RELEASE_AFTER:-12}"
POST_RELEASE_GRIPPER_WEIGHT="${POST_RELEASE_GRIPPER_WEIGHT:-8.0}"
PARAMETER_ANCHOR_RATE="${PARAMETER_ANCHOR_RATE:-2e-5}"
EVENT_OVERSAMPLE_BEFORE="${EVENT_OVERSAMPLE_BEFORE:-8}"
EVENT_OVERSAMPLE_AFTER="${EVENT_OVERSAMPLE_AFTER:-12}"
EVENT_OVERSAMPLE_REPEATS="${EVENT_OVERSAMPLE_REPEATS:-3}"
STACK_N_ACTION_STEPS="${STACK_N_ACTION_STEPS:-}"
GRIPPER_USE_HISTORY="${GRIPPER_USE_HISTORY:-false}"
GRIPPER_HISTORY_UPDATE_RATE="${GRIPPER_HISTORY_UPDATE_RATE:-1.0}"
GRIPPER_SIGMA="${GRIPPER_SIGMA:-}"
GRIPPER_SIGMA_HIGH="${GRIPPER_SIGMA_HIGH:-}"
STACK_RELEASE_HOLD_STEPS="${STACK_RELEASE_HOLD_STEPS:-5}"
STACK_MIN_EEF_DISTANCE="${STACK_MIN_EEF_DISTANCE:-0.04}"
STACK_MIN_GRIPPER_OPEN_WIDTH="${STACK_MIN_GRIPPER_OPEN_WIDTH:-0.04}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] Stack dataset not found: ${DATASET}" >&2
    exit 1
fi

if [[ -z "${SOURCE_CKPT}" ]]; then
    if [[ -n "${SOURCE_RUN}" ]]; then
        SOURCE_CHECKPOINT_DIR="${OUTPUT_ROOT}/${SOURCE_RUN}/checkpoints"
        mapfile -t MATCHING_CKPTS < <(
            find "${SOURCE_CHECKPOINT_DIR}" -maxdepth 1 -type f \
                -name "epoch=${EXPECTED_SOURCE_EPOCH}_*.ckpt" | sort
        )
        if (( ${#MATCHING_CKPTS[@]} != 1 )); then
            echo "[ERROR] Expected exactly one epoch-${EXPECTED_SOURCE_EPOCH} checkpoint in ${SOURCE_CHECKPOINT_DIR}, found ${#MATCHING_CKPTS[@]}." >&2
            echo "Set SOURCE_CKPT explicitly when the source filename differs." >&2
            exit 1
        fi
        SOURCE_CKPT="${MATCHING_CKPTS[0]}"
    else
        mapfile -t MATCHING_CKPTS < <(
            find "${OUTPUT_ROOT}" -maxdepth 4 -type f \
                -path '*/stack_d1_event_release_v3_7*/checkpoints/*' \
                -name "epoch=${EXPECTED_SOURCE_EPOCH}_*.ckpt" | sort
        )
        if (( ${#MATCHING_CKPTS[@]} != 1 )); then
            echo "[ERROR] Expected exactly one v3.7 epoch-${EXPECTED_SOURCE_EPOCH} checkpoint, found ${#MATCHING_CKPTS[@]}." >&2
            printf '  %s\n' "${MATCHING_CKPTS[@]}" >&2
            echo "Set SOURCE_CKPT explicitly when more than one candidate exists." >&2
            exit 1
        fi
        SOURCE_CKPT="${MATCHING_CKPTS[0]}"
    fi
fi

if [[ -z "${SOURCE_CKPT}" || ! -f "${SOURCE_CKPT}" ]]; then
    echo "[ERROR] Could not find the trained Stack latest.ckpt." >&2
    echo "Set SOURCE_CKPT=/absolute/or/project/relative/latest.ckpt and retry." >&2
    exit 1
fi

SOURCE_DIR="$(dirname "$(dirname "${SOURCE_CKPT}")")"
SOURCE_CONFIG="${SOURCE_DIR}/.hydra/config.yaml"
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "[ERROR] Source Hydra config not found: ${SOURCE_CONFIG}" >&2
    exit 1
fi

SOURCE_EPOCH="$(python - "${SOURCE_CKPT}" <<'PY'
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

if (( SOURCE_EPOCH < MIN_SOURCE_EPOCH )); then
    echo "[ERROR] Refusing stale Stack source epoch ${SOURCE_EPOCH}; expected >= ${MIN_SOURCE_EPOCH}." >&2
    echo "Set SOURCE_CKPT to the verified epoch-${EXPECTED_SOURCE_EPOCH} checkpoint." >&2
    exit 1
fi

if (( SOURCE_EPOCH != EXPECTED_SOURCE_EPOCH )); then
    echo "[ERROR] Expected source epoch ${EXPECTED_SOURCE_EPOCH}, got ${SOURCE_EPOCH}." >&2
    echo "Refusing to fine-tune from an unintended Stack checkpoint." >&2
    exit 1
fi

# Preserve the source run's validated motion-prior settings. Only release
# supervision and learning rate are changed below.
mapfile -t SOURCE_VALUES < <(python - "${SOURCE_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
task_name = str(OmegaConf.select(cfg, "task_semantics.task_name", default=""))
dataset_path = str(OmegaConf.select(cfg, "task.dataset_path", default=""))
if task_name != "stack_d1" and "stack_d1" not in dataset_path.lower():
    raise SystemExit(
        f"[ERROR] source checkpoint is not Stack_D1: "
        f"task_name={task_name!r}, dataset={dataset_path!r}"
    )

keys_and_defaults = (
    ("policy.action_group_spectral_params.translation.sigma", 0.24),
    ("policy.action_group_spectral_params.translation.sigma_high", 0.08),
    ("policy.action_group_spectral_params.rotation.sigma", 0.12),
    ("policy.action_group_spectral_params.rotation.sigma_high", 0.06),
    ("policy.action_group_spectral_params.rotation.freq_split_high", 12),
    ("policy.action_group_history_params.shift", 2),
    ("n_action_steps", 2),
    ("policy.num_inference_steps", 8),
    ("policy.action_group_spectral_params.gripper.sigma", 0.35),
    ("policy.action_group_spectral_params.gripper.sigma_high", 0.25),
)
for key, default in keys_and_defaults:
    print(OmegaConf.select(cfg, key, default=default))
PY
)

TRANSLATION_SIGMA="${SOURCE_VALUES[0]}"
TRANSLATION_SIGMA_HIGH="${SOURCE_VALUES[1]}"
ROTATION_SIGMA="${SOURCE_VALUES[2]}"
ROTATION_SIGMA_HIGH="${SOURCE_VALUES[3]}"
ROTATION_SPLIT="${SOURCE_VALUES[4]}"
HISTORY_SHIFT="${SOURCE_VALUES[5]}"
N_ACTION_STEPS="${STACK_N_ACTION_STEPS:-${SOURCE_VALUES[6]}}"
INFERENCE_STEPS="${SOURCE_VALUES[7]}"
GRIPPER_SIGMA="${GRIPPER_SIGMA:-${SOURCE_VALUES[8]}}"
GRIPPER_SIGMA_HIGH="${GRIPPER_SIGMA_HIGH:-${SOURCE_VALUES[9]}}"

TARGET_EPOCH="$((SOURCE_EPOCH + FINETUNE_EPOCHS + 1))"
SCHEDULER_EPOCHS="$((FINETUNE_EPOCHS + 1))"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/checkpoints/latest.ckpt" ]]; then
    EXISTING_EPOCH="$(python - "${OUTPUT_DIR}/checkpoints/latest.ckpt" <<'PY'
import dill
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", pickle_module=dill)
raw = payload.get("pickles", {}).get("epoch")
if raw is None:
    raise SystemExit("[ERROR] existing checkpoint does not contain epoch")
print(int(dill.loads(raw)))
PY
)"
    if (( EXISTING_EPOCH < SOURCE_EPOCH || EXISTING_EPOCH >= TARGET_EPOCH )); then
        echo "[ERROR] incompatible existing fine-tune checkpoint: ${OUTPUT_DIR}/checkpoints/latest.ckpt" >&2
        echo "[ERROR] existing_epoch=${EXISTING_EPOCH}, expected range=${SOURCE_EPOCH}..$((TARGET_EPOCH - 1))" >&2
        echo "Use a new RUN_NAME; refusing to silently resume an unrelated run." >&2
        exit 1
    fi
    TRAINING_RESUME=true
    RESUME_FROM_PATH=null
    RESUME_OVERRIDE="++training.resume_from_path=null"
    echo "[STACK-RELEASE] resuming existing fine-tune: ${OUTPUT_DIR}"
else
    TRAINING_RESUME=false
    RESUME_FROM_PATH="${SOURCE_CKPT}"
    # Hydra needs the path quoted because the checkpoint filename contains '='.
    RESUME_OVERRIDE="++training.resume_from_path='${SOURCE_CKPT}'"
    echo "[STACK-RELEASE] transferring trained Stack checkpoint"
fi

echo "[STACK-RELEASE] source=${SOURCE_CKPT}"
echo "[STACK-RELEASE] source_epoch=${SOURCE_EPOCH} target_epoch=$((TARGET_EPOCH - 1))"
echo "[STACK-RELEASE] lr=${FINETUNE_LR} release_radius=${RELEASE_RADIUS} release_gripper_weight=${RELEASE_GRIPPER_WEIGHT}"
echo "[STACK-RELEASE] align_before=${PRE_RELEASE_BEFORE} settle=${SETTLE_BEFORE}+${SETTLE_AFTER} anchor_rate=${PARAMETER_ANCHOR_RATE}"
echo "[STACK-RELEASE] event_sampling=last_release repeats=${EVENT_OVERSAMPLE_REPEATS} phase_weight=${PHASE_WEIGHT}"
echo "[STACK-RELEASE] synchronized_history gripper=${GRIPPER_USE_HISTORY} shift=${HISTORY_SHIFT}"
echo "[STACK-RELEASE] gripper_prior sigma=${GRIPPER_SIGMA}/${GRIPPER_SIGMA_HIGH} update_rate=${GRIPPER_HISTORY_UPDATE_RATE}"
echo "[STACK-RELEASE] output=${OUTPUT_DIR}"

python train.py --config-name=mimicgen_single \
    "hydra.run.dir=${OUTPUT_DIR}" \
    "obs_dim=32" \
    "task_semantics.task_name=stack_d1" \
    "task.dataset_path=${DATASET}" \
    "task.abs_action=false" \
    "task.dataset.abs_action=false" \
    "++task.dataset.episode_prefix_close_threshold=auto" \
    "++task.dataset.episode_prefix_close_is_greater=auto" \
    "++task.dataset.event_oversample_enabled=true" \
    "++task.dataset.event_oversample_anchor=last_release" \
    "++task.dataset.event_oversample_before=${EVENT_OVERSAMPLE_BEFORE}" \
    "++task.dataset.event_oversample_after=${EVENT_OVERSAMPLE_AFTER}" \
    "++task.dataset.event_oversample_repeats=${EVENT_OVERSAMPLE_REPEATS}" \
    "++task.dataset.event_oversample_min_anchor_ratio=0.95" \
    "task.env_runner.abs_action=false" \
    "training.device=${DEVICE}" \
    "training.seed=${SEED}" \
    "data_split.seed=${SEED}" \
    "data_split.max_train_episodes=1000" \
    "training.resume=${TRAINING_RESUME}" \
    "${RESUME_OVERRIDE}" \
    "++training.rollout_before_training=true" \
    "++training.parameter_anchor_path='${SOURCE_CKPT}'" \
    "++training.parameter_anchor_rate=${PARAMETER_ANCHOR_RATE}" \
    "training.num_epochs=${TARGET_EPOCH}" \
    "++training.lr_scheduler_num_epochs=${SCHEDULER_EPOCHS}" \
    "++training.lr_scheduler_start_epoch=${SOURCE_EPOCH}" \
    "++training.rollout_start_epoch=${SOURCE_EPOCH}" \
    "training.rollout_every=${ROLLOUT_EVERY}" \
    "training.checkpoint_every=${CHECKPOINT_EVERY}" \
    "optimizer.lr=${FINETUNE_LR}" \
    "task.env_runner.max_steps=400" \
    "task.env_runner.n_envs=${N_ENVS}" \
    "task.env_runner.n_test=${N_TEST}" \
    "task.env_runner.n_test_vis=${N_TEST_VIS}" \
    "++task.env_runner.stack_release_hold_steps=${STACK_RELEASE_HOLD_STEPS}" \
    "++task.env_runner.stack_min_eef_distance=${STACK_MIN_EEF_DISTANCE}" \
    "++task.env_runner.stack_min_gripper_open_width=${STACK_MIN_GRIPPER_OPEN_WIDTH}" \
    "logging.mode=${WANDB_MODE}" \
    "logging.name=${RUN_NAME}" \
    "n_action_steps=${N_ACTION_STEPS}" \
    "policy.num_inference_steps=${INFERENCE_STEPS}" \
    "policy.action_group_spectral_params.translation.sigma=${TRANSLATION_SIGMA}" \
    "policy.action_group_spectral_params.translation.sigma_high=${TRANSLATION_SIGMA_HIGH}" \
    "policy.action_group_spectral_params.rotation.freq_split_high=${ROTATION_SPLIT}" \
    "policy.action_group_spectral_params.rotation.sigma=${ROTATION_SIGMA}" \
    "policy.action_group_spectral_params.rotation.sigma_high=${ROTATION_SIGMA_HIGH}" \
    "policy.action_group_spectral_params.gripper.sigma=${GRIPPER_SIGMA}" \
    "policy.action_group_spectral_params.gripper.sigma_high=${GRIPPER_SIGMA_HIGH}" \
    "policy.action_group_loss_weights.translation.weight=${TRANSLATION_LOSS_WEIGHT}" \
    "policy.action_group_loss_weights.rotation.weight=${ROTATION_LOSS_WEIGHT}" \
    "policy.action_group_loss_weights.gripper.weight=${GRIPPER_LOSS_WEIGHT}" \
    "policy.action_group_history_params.enabled=true" \
    "policy.action_group_history_params.shift=${HISTORY_SHIFT}" \
    "policy.action_group_history_params.groups.translation.use_history=true" \
    "policy.action_group_history_params.groups.rotation.use_history=true" \
    "policy.action_group_history_params.groups.gripper.use_history=${GRIPPER_USE_HISTORY}" \
    "policy.action_group_history_params.groups.gripper.update_rate=${GRIPPER_HISTORY_UPDATE_RATE}" \
    "++policy.history_training_mode=aligned" \
    "policy.action_group_timing_params.gripper.enabled=false" \
    "policy.action_phase_loss_params.enabled=true" \
    "policy.action_phase_loss_params.weight=${PHASE_WEIGHT}" \
    "policy.action_phase_loss_params.release_enabled=true" \
    "policy.action_phase_loss_params.release_transition_radius=${RELEASE_RADIUS}" \
    "policy.action_phase_loss_params.release_translation_weight=${RELEASE_TRANSLATION_WEIGHT}" \
    "policy.action_phase_loss_params.release_rotation_weight=${RELEASE_ROTATION_WEIGHT}" \
    "policy.action_phase_loss_params.release_gripper_weight=${RELEASE_GRIPPER_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.enabled=true" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.anchor=release" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.occurrence=last" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.before=${PRE_RELEASE_BEFORE}" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.after=0" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.translation_weight=${PRE_RELEASE_TRANSLATION_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.rotation_weight=${PRE_RELEASE_ROTATION_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.pre_release_align.gripper_weight=${PRE_RELEASE_GRIPPER_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.enabled=true" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.anchor=release" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.occurrence=last" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.before=${SETTLE_BEFORE}" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.after=${SETTLE_AFTER}" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.translation_weight=${SETTLE_TRANSLATION_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.rotation_weight=${SETTLE_ROTATION_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.contact_settle.gripper_weight=${SETTLE_GRIPPER_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.enabled=true" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.anchor=release" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.occurrence=last" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.before=1" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.after=${POST_RELEASE_AFTER}" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.translation_weight=0.2" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.rotation_weight=0.1" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.gripper_weight=${POST_RELEASE_GRIPPER_WEIGHT}" \
    "++task.env_runner.action_clip_by_dataset=true" \
    "++task.env_runner.action_clip_margin_scale=0.10" \
    "++task.env_runner.action_clip_min_margin=0.02" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
