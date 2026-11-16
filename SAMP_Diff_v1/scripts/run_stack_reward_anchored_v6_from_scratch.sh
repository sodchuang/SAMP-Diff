#!/usr/bin/env bash
# Stack_D1 v6: train one complete delta-action policy from epoch 0.
#
# The sampler is anchored on the first positive expert reward instead of every
# last gripper release.  This emphasizes demonstrated successful stack / release
# / retreat transitions without treating off-target releases as good examples.
# Motion history is retained for continuous pose channels; the discontinuous
# gripper channel is regenerated from the current observation.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME:-stack_reward_anchored_v6_seed42}"
NUM_EPOCHS="${NUM_EPOCHS:-3500}"

echo "[STACK-V6] clean training from epoch 0"
echo "[STACK-V6] event anchor=first_positive_reward (successful expert states)"
echo "[STACK-V6] one end-to-end policy; no checkpoint transfer or curriculum stage"

RUN_NAME="${RUN_NAME}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
RESUME_EXISTING=false \
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}" \
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
N_ENVS="${N_ENVS:-28}" \
N_TEST="${N_TEST:-30}" \
N_TEST_VIS="${N_TEST_VIS:-10}" \
N_ACTION_STEPS=2 \
INFERENCE_STEPS=8 \
HISTORY_ENABLED=true \
HISTORY_TRAINING_MODE=aligned \
TRANSLATION_USE_HISTORY=true \
TRANSLATION_HISTORY_UPDATE_RATE=1.0 \
ROTATION_USE_HISTORY=true \
ROTATION_HISTORY_UPDATE_RATE=1.0 \
GRIPPER_USE_HISTORY=false \
COLD_START_PROB=0.20 \
TRANSLATION_FREQ_SPLIT_HIGH=8 \
TRANSLATION_SIGMA=0.18 \
TRANSLATION_SIGMA_HIGH=0.06 \
ROTATION_FREQ_SPLIT_HIGH=12 \
ROTATION_SIGMA=0.10 \
ROTATION_SIGMA_HIGH=0.05 \
GRIPPER_FREQ_SPLIT_HIGH=16 \
GRIPPER_SIGMA=0.40 \
GRIPPER_SIGMA_HIGH=0.30 \
TRANSLATION_LOSS_WEIGHT=1.80 \
ROTATION_LOSS_WEIGHT=1.60 \
BASE_GRIPPER_WEIGHT=2.40 \
PHASE_LOSS_ENABLED=true \
PHASE_LOSS_WEIGHT=0.020 \
STACK_EVENT_ANCHOR=first_positive_reward \
STACK_RELEASE_GATE_ENABLED=false \
STACK_RELEASE_MIN_ANCHOR_RATIO=0.98 \
EVENT_OVERSAMPLE_ENABLED=true \
EVENT_OVERSAMPLE_BEFORE=16 \
EVENT_OVERSAMPLE_AFTER=12 \
EVENT_OVERSAMPLE_REPEATS=2 \
GRASP_HOLD_AFTER=12 \
GRASP_HOLD_GRIPPER_WEIGHT=1.80 \
PRE_RELEASE_BEFORE=14 \
PRE_RELEASE_TRANSLATION_WEIGHT=2.20 \
PRE_RELEASE_ROTATION_WEIGHT=1.60 \
SETTLE_BEFORE=4 \
SETTLE_AFTER=3 \
SETTLE_TRANSLATION_WEIGHT=2.50 \
SETTLE_ROTATION_WEIGHT=1.50 \
RELEASE_RADIUS=3 \
RELEASE_GRIPPER_WEIGHT=4.50 \
POST_RELEASE_AFTER=16 \
POST_RELEASE_TRANSLATION_WEIGHT=0.80 \
POST_RELEASE_ROTATION_WEIGHT=0.40 \
POST_RELEASE_GRIPPER_WEIGHT=5.50 \
bash scripts/run_stack_delta_hg_gc_fix_from_scratch.sh
