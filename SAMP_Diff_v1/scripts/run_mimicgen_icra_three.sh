#!/usr/bin/env bash
# Final fixed three-task MimicGen recipe for the ICRA experiment table.

set -euo pipefail

TASKS="${TASKS:-stack_d1 nut_assembly_d0 threading_d2}"
NUM_EPOCHS="${NUM_EPOCHS:-5000}"
SEED="${SEED:-42}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TASKS="${TASKS}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
SEED="${SEED}" \
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}" \
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" \
N_ENVS="${N_ENVS:-28}" \
N_TEST="${N_TEST:-50}" \
N_TEST_VIS="${N_TEST_VIS:-6}" \
GPU_ID="${GPU_ID:-0}" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
bash scripts/run_mimicgen_single_tasks.sh train
