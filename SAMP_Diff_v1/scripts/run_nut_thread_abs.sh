#!/usr/bin/env bash
# Train independent absolute-action policies, sequentially on one GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TASKS="${TASKS:-nut_assembly_d0 threading_d2}"
DATA_ROOT="${DATA_ROOT:-$HOME/Freqpolicy}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export NUM_EPOCHS="${NUM_EPOCHS:-5000}"
export ROLLOUT_EVERY="${ROLLOUT_EVERY:-null}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
export SEED="${SEED:-42}"
export DEVICE="${DEVICE:-cuda:0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
read -r -a SELECTED <<< "$TASKS"
for TASK in "${SELECTED[@]}"; do
    case "$TASK" in
        nut_assembly_d0|threading_d2) ;;
        *) echo "[ERROR] TASKS supports nut_assembly_d0 threading_d2" >&2; exit 2 ;;
    esac
    if [[ ! -f "$DATA_ROOT/${TASK}_abs.hdf5" ]]; then
        echo "[ERROR] Missing $DATA_ROOT/${TASK}_abs.hdf5" >&2; exit 1
    fi
done
for TASK in "${SELECTED[@]}"; do
    bash scripts/train_abs_from_delta_dataset.sh \
        "$TASK" "$DATA_ROOT/${TASK}_abs.hdf5" "${TASK}_abs_seed${SEED}_${RUN_TAG}"
done
echo '[DONE] Requested tasks completed'
