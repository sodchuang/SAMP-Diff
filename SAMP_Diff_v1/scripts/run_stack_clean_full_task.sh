#!/usr/bin/env bash
# Full-task Stack semantic policy: one uninterrupted run. Reuse history only
# for continuous xyz / rotation groups; regenerate the discontinuous gripper
# group so the policy must learn observation-conditioned release timing.

set -euo pipefail

RUN_NAME="${RUN_NAME:-stack_clean_full_semantic_v3_seed42}"
NUM_EPOCHS="${NUM_EPOCHS:-2200}"
HISTORY_ENABLED="${HISTORY_ENABLED:-true}"
HISTORY_TRAINING_MODE="${HISTORY_TRAINING_MODE:-aligned}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
PHASE_LOSS_ENABLED=true \
PHASE_LOSS_WEIGHT=0.05 \
EVENT_OVERSAMPLE_ENABLED=true \
EVENT_OVERSAMPLE_REPEATS=3 \
GRIPPER_USE_HISTORY=false \
HISTORY_ENABLED="${HISTORY_ENABLED}" \
HISTORY_TRAINING_MODE="${HISTORY_TRAINING_MODE}" \
bash scripts/run_stack_delta_hg_gc_fix_from_scratch.sh
