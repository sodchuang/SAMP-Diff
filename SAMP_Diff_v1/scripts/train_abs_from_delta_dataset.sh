#!/usr/bin/env bash
# Stored absolute axis-angle 7D -> dataset rotation6D -> policy 10D.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 TASK ABS_DATASET RUN_NAME" >&2
    exit 2
fi
TASK_NAME="$1"
DATASET_PATH="$2"
RUN_NAME="$3"
case "$TASK_NAME" in
    stack_d1) EXPECTED_ENV=Stack_D1; DEFAULT_MAX_STEPS=400 ;;
    nut_assembly_d0) EXPECTED_ENV=NutAssembly_D0; DEFAULT_MAX_STEPS=700 ;;
    threading_d2) EXPECTED_ENV=Threading_D2; DEFAULT_MAX_STEPS=500 ;;
    *) echo "[ERROR] Unsupported task: $TASK_NAME" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-5000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-null}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_abs}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
WANDB_MODE="${WANDB_MODE:-disabled}"
MODE="${MODE:-train}"
if [[ "$MODE" != train && "$MODE" != config ]]; then
    echo '[ERROR] MODE must be train or config' >&2; exit 2
fi
if [[ -d "$OUTPUT_DIR/checkpoints" ]] && compgen -G "$OUTPUT_DIR/checkpoints/*.ckpt" > /dev/null; then
    echo "[ERROR] Existing checkpoints in $OUTPUT_DIR; choose a new RUN_NAME" >&2
    exit 1
fi

# Do not inherit Stack's obs_dim=32 for the other two tasks.
META=$(python - "$DATASET_PATH" "$EXPECTED_ENV" <<'PY'
import json, sys
from pathlib import Path
import h5py
import numpy as np
p = Path(sys.argv[1]).expanduser().resolve()
keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
with h5py.File(p, 'r') as f:
    meta = json.loads(f['data'].attrs['env_args'])
    if meta['env_name'] != sys.argv[2]:
        raise ValueError(f"Expected {sys.argv[2]}, found {meta['env_name']}")
    if meta['env_kwargs']['controller_configs'].get('control_delta') is not False:
        raise ValueError('Not a converted absolute dataset: control_delta must be False')
    dims = set()
    for name, demo in f['data'].items():
        a = demo['actions'][:]
        if a.ndim != 2 or a.shape[1] != 7 or len(a) == 0 or not np.isfinite(a).all():
            raise ValueError(f'{name}: invalid stored 7D actions')
        dim = 0
        for key in keys:
            obs = demo['obs'][key][:]
            if obs.ndim != 2 or len(obs) != len(a) or not np.isfinite(obs).all():
                raise ValueError(f'{name}: invalid observation {key}')
            dim += obs.shape[1]
        dims.add(dim)
    if len(dims) != 1:
        raise ValueError(f'Inconsistent observation dimensions: {dims}')
    print(next(iter(dims)))
    print(str(p))
    print(f'[DATA] {meta["env_name"]}: {len(f["data"])} demos validated', file=sys.stderr)
PY
)
OBS_DIM=$(printf '%s\n' "$META" | head -n 1)
DATASET_PATH=$(printf '%s\n' "$META" | tail -n 1)
echo "[TRAIN] task=$TASK_NAME obs_dim=$OBS_DIM policy_action_dim=10 epochs=$NUM_EPOCHS"
echo "[DATASET] $DATASET_PATH"
echo "[OUTPUT] $OUTPUT_DIR"
echo "[ROLLOUT] every=$ROLLOUT_EVERY (null means train now, eval separately)"
EXTRA=()
if [[ "$MODE" == config ]]; then EXTRA=(--cfg job); fi
mkdir -p "$OUTPUT_DIR"
python train.py --config-name=mimicgen_single \
    "obs_dim=$OBS_DIM" "action_dim=10" \
    "++task_semantics.task_name=$TASK_NAME" \
    "++task_semantics.control_mode=abs_eef_pose" \
    "++task_semantics.action_representation=xyz_rotation6d_gripper" \
    '++task_semantics.action_groups.translation=[0,1,2]' \
    '++task_semantics.action_groups.rotation=[3,4,5,6,7,8]' \
    '++task_semantics.action_groups.gripper=[9]' \
    '++task_semantics.diagnostics.rotation_groups.rotation=[3,4,5,6,7,8]' \
    '++task_semantics.diagnostics.contact_groups.gripper=[9]' \
    '++task.action_dim=10' '++task.abs_action=true' \
    "++task.dataset_path=$DATASET_PATH" \
    "++task.dataset.dataset_path=$DATASET_PATH" '++task.dataset.abs_action=true' \
    "++task.env_runner.dataset_path=$DATASET_PATH" '++task.env_runner.abs_action=true' \
    "++task.env_runner.max_steps=${MAX_STEPS:-$DEFAULT_MAX_STEPS}" \
    "++task.env_runner.n_envs=${N_ENVS:-4}" \
    '++policy.action_group_spectral_params.translation.indices=[0,1,2]' \
    '++policy.action_group_spectral_params.rotation.indices=[3,4,5,6,7,8]' \
    '++policy.action_group_spectral_params.gripper.indices=[9]' \
    '++policy.action_group_loss_weights.translation.indices=[0,1,2]' \
    '++policy.action_group_loss_weights.rotation.indices=[3,4,5,6,7,8]' \
    '++policy.action_group_loss_weights.gripper.indices=[9]' \
    '++policy.action_group_history_params.groups.translation.indices=[0,1,2]' \
    '++policy.action_group_history_params.groups.rotation.indices=[3,4,5,6,7,8]' \
    '++policy.action_group_history_params.groups.gripper.indices=[9]' \
    '++policy.action_group_timing_params.gripper.indices=[9]' \
    '++policy.action_phase_loss_params.translation_indices=[0,1,2]' \
    '++policy.action_phase_loss_params.rotation_indices=[3,4,5,6,7,8]' \
    '++policy.action_phase_loss_params.gripper_indices=[9]' \
    '++policy.action_phase_loss_params.release_translation_indices=[0,1,2]' \
    '++policy.action_phase_loss_params.release_rotation_indices=[3,4,5,6,7,8]' \
    "training.device=$DEVICE" "training.seed=$SEED" "++task.dataset.seed=$SEED" \
    "training.num_epochs=$NUM_EPOCHS" 'training.resume=false' \
    "training.rollout_every=$ROLLOUT_EVERY" "training.checkpoint_every=$CHECKPOINT_EVERY" \
    "logging.mode=$WANDB_MODE" "logging.name=$RUN_NAME" \
    "hydra.run.dir=$OUTPUT_DIR" "${EXTRA[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
