#!/usr/bin/env bash
# Continue a trained MimicGen Stack_D1 policy and focus supervision on the
# final gripper release. This is a weight transfer, not training from scratch.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
DATASET="${DATASET:-data/mimicgen/core/stack_d1.hdf5}"
SOURCE_RUN="${SOURCE_RUN:-}"
SOURCE_CKPT="${SOURCE_CKPT:-}"
RUN_NAME="${RUN_NAME:-stack_d1_release_fix_final_v3_seed42}"
SEED="${SEED:-42}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-600}"
FINETUNE_LR="${FINETUNE_LR:-8e-6}"
MIN_SOURCE_EPOCH="${MIN_SOURCE_EPOCH:-2500}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-8}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# Release-specific supervision. Pose weights stay conservative so an already
# learned grasp / transport / alignment trajectory is not overwritten.
PHASE_WEIGHT="${PHASE_WEIGHT:-0.120}"
GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-3.0}"
RELEASE_RADIUS="${RELEASE_RADIUS:-6}"
RELEASE_TRANSLATION_WEIGHT="${RELEASE_TRANSLATION_WEIGHT:-0.35}"
RELEASE_ROTATION_WEIGHT="${RELEASE_ROTATION_WEIGHT:-0.25}"
RELEASE_GRIPPER_WEIGHT="${RELEASE_GRIPPER_WEIGHT:-10.0}"
STACK_N_ACTION_STEPS="${STACK_N_ACTION_STEPS:-1}"
STACK_RELEASE_HOLD_STEPS="${STACK_RELEASE_HOLD_STEPS:-1}"
STACK_MIN_EEF_DISTANCE="${STACK_MIN_EEF_DISTANCE:-0.04}"
STACK_MIN_GRIPPER_OPEN_WIDTH="${STACK_MIN_GRIPPER_OPEN_WIDTH:-0.04}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] Stack dataset not found: ${DATASET}" >&2
    exit 1
fi

if [[ -z "${SOURCE_CKPT}" ]]; then
    if [[ -n "${SOURCE_RUN}" ]]; then
        SOURCE_CKPT="${OUTPUT_ROOT}/${SOURCE_RUN}/checkpoints/latest.ckpt"
    else
        # Continue from the newest Stack checkpoint, including a preceding
        # release fine-tune. Exclude only this run's own output so a stopped
        # run is resumed through training.resume below instead of transferred
        # into itself.
        SOURCE_CKPT="$({
            find "${OUTPUT_ROOT}" -maxdepth 3 -type f \
                -path '*/stack_d1_*/checkpoints/latest.ckpt' \
                ! -path "${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}/checkpoints/latest.ckpt" \
                -printf '%T@ %p\n' 2>/dev/null || true
        } | sort -nr | head -n 1 | cut -d' ' -f2-)"
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
    echo "Set SOURCE_CKPT to the completed 2600 checkpoint." >&2
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
    ("policy.action_group_spectral_params.rotation.sigma", 0.12),
    ("policy.action_group_spectral_params.rotation.freq_split_high", 12),
    ("policy.action_group_history_params.shift", 2),
    ("n_action_steps", 2),
)
for key, default in keys_and_defaults:
    print(OmegaConf.select(cfg, key, default=default))
PY
)

TRANSLATION_SIGMA="${SOURCE_VALUES[0]}"
ROTATION_SIGMA="${SOURCE_VALUES[1]}"
ROTATION_SPLIT="${SOURCE_VALUES[2]}"
HISTORY_SHIFT="${SOURCE_VALUES[3]}"
N_ACTION_STEPS="${STACK_N_ACTION_STEPS}"

TARGET_EPOCH="$((SOURCE_EPOCH + FINETUNE_EPOCHS + 1))"
SCHEDULER_EPOCHS="$((FINETUNE_EPOCHS + 1))"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/checkpoints/latest.ckpt" ]]; then
    TRAINING_RESUME=true
    RESUME_FROM_PATH=null
    echo "[STACK-RELEASE] resuming existing fine-tune: ${OUTPUT_DIR}"
else
    TRAINING_RESUME=false
    RESUME_FROM_PATH="${SOURCE_CKPT}"
    echo "[STACK-RELEASE] transferring trained Stack checkpoint"
fi

echo "[STACK-RELEASE] source=${SOURCE_CKPT}"
echo "[STACK-RELEASE] source_epoch=${SOURCE_EPOCH} target_epoch=$((TARGET_EPOCH - 1))"
echo "[STACK-RELEASE] lr=${FINETUNE_LR} release_radius=${RELEASE_RADIUS} release_gripper_weight=${RELEASE_GRIPPER_WEIGHT}"
echo "[STACK-RELEASE] output=${OUTPUT_DIR}"

python train.py --config-name=mimicgen_single \
    "hydra.run.dir=${OUTPUT_DIR}" \
    "obs_dim=32" \
    "task_semantics.task_name=stack_d1" \
    "task.dataset_path=${DATASET}" \
    "training.seed=${SEED}" \
    "data_split.seed=${SEED}" \
    "data_split.max_train_episodes=1000" \
    "training.resume=${TRAINING_RESUME}" \
    "++training.resume_from_path=${RESUME_FROM_PATH}" \
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
    "policy.num_inference_steps=6" \
    "policy.action_group_spectral_params.translation.sigma=${TRANSLATION_SIGMA}" \
    "policy.action_group_spectral_params.translation.sigma_high=${TRANSLATION_SIGMA}" \
    "policy.action_group_spectral_params.rotation.freq_split_high=${ROTATION_SPLIT}" \
    "policy.action_group_spectral_params.rotation.sigma=${ROTATION_SIGMA}" \
    "policy.action_group_spectral_params.rotation.sigma_high=${ROTATION_SIGMA}" \
    "policy.action_group_loss_weights.gripper.weight=${GRIPPER_LOSS_WEIGHT}" \
    "policy.action_group_history_params.enabled=true" \
    "policy.action_group_history_params.shift=${HISTORY_SHIFT}" \
    "policy.action_group_history_params.groups.translation.use_history=true" \
    "policy.action_group_history_params.groups.rotation.use_history=true" \
    "policy.action_group_history_params.groups.gripper.use_history=false" \
    "++policy.history_training_mode=aligned" \
    "policy.action_group_timing_params.gripper.enabled=false" \
    "policy.action_phase_loss_params.enabled=true" \
    "policy.action_phase_loss_params.weight=${PHASE_WEIGHT}" \
    "policy.action_phase_loss_params.release_enabled=true" \
    "policy.action_phase_loss_params.release_transition_radius=${RELEASE_RADIUS}" \
    "policy.action_phase_loss_params.release_translation_weight=${RELEASE_TRANSLATION_WEIGHT}" \
    "policy.action_phase_loss_params.release_rotation_weight=${RELEASE_ROTATION_WEIGHT}" \
    "policy.action_phase_loss_params.release_gripper_weight=${RELEASE_GRIPPER_WEIGHT}" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.enabled=true" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.anchor=release" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.occurrence=last" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.before=2" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.after=5" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.translation_weight=0.30" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.rotation_weight=0.20" \
    "++policy.action_phase_loss_params.extra_windows.stack_release.gripper_weight=12.0" \
    "++task.env_runner.action_clip_by_dataset=true" \
    "++task.env_runner.action_clip_margin_scale=0.10" \
    "++task.env_runner.action_clip_min_margin=0.02" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
