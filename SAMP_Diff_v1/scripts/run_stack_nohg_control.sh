#!/usr/bin/env bash
# Controlled Stack_D1 comparison: preserve the full semantic / reward recipe,
# but remove history guidance from delta actions.

set -euo pipefail

RUN_NAME="${RUN_NAME:-stack_clean_full_nohg_v1_seed42}"
NUM_EPOCHS="${NUM_EPOCHS:-2200}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
HISTORY_ENABLED=true \
HISTORY_TRAINING_MODE=aligned \
TRANSLATION_USE_HISTORY=false \
ROTATION_USE_HISTORY=false \
GRIPPER_USE_HISTORY=false \
bash scripts/run_stack_clean_full_task.sh
