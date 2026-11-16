#!/usr/bin/env bash
# Continue v5.2 from its latest checkpoint. Keep the working gripper history,
# but learn a longer align-and-settle approach before the final release.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
SOURCE_RUN="${SOURCE_RUN:-stack_delta_release_history_v5_2_seed42}"
RUN_NAME="${RUN_NAME:-stack_delta_align_then_release_v5_3_seed42}"
SOURCE_CKPT="${SOURCE_CKPT:-}"
SOURCE_SELECTION="${SOURCE_SELECTION:-latest}"
N_TEST="${N_TEST:-30}"
# Save every test rollout so a non-zero score always has a matching video.
N_TEST_VIS="${N_TEST_VIS:-${N_TEST}}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -z "${SOURCE_CKPT}" ]]; then
    CHECKPOINT_DIR="${OUTPUT_ROOT}/${SOURCE_RUN}/checkpoints"
    if [[ "${SOURCE_SELECTION}" == "latest" ]]; then
        SOURCE_CKPT="${CHECKPOINT_DIR}/latest.ckpt"
    elif [[ "${SOURCE_SELECTION}" == "best_score" ]]; then
        SOURCE_CKPT="$(python - "${CHECKPOINT_DIR}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"epoch=(\d+)_test_score=([-+0-9.eE]+)_")
candidates = []
for path in root.glob("epoch=*_test_score=*.ckpt"):
    match = pattern.search(path.name)
    if match:
        candidates.append((float(match.group(2)), int(match.group(1)), path))
if candidates:
    # Highest verified rollout score wins; the later epoch breaks ties.
    print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
)"
    else
        echo "[ERROR] SOURCE_SELECTION must be latest or best_score." >&2
        exit 1
    fi
    if [[ -z "${SOURCE_CKPT}" || ! -f "${SOURCE_CKPT}" ]]; then
        echo "[ERROR] no scored checkpoint found in ${CHECKPOINT_DIR}" >&2
        exit 1
    fi
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

echo "[STACK-V5.1] source=${SOURCE_CKPT}"
echo "[STACK-V5.1] source_epoch=${SOURCE_EPOCH}"
echo "[STACK-V5.1] source_selection=${SOURCE_SELECTION}"
echo "[STACK-V5.3] change=preserve gripper history; align and settle before release"

SOURCE_CKPT="${SOURCE_CKPT}" \
RUN_NAME="${RUN_NAME}" \
EXPECTED_SOURCE_EPOCH="${SOURCE_EPOCH}" \
MIN_SOURCE_EPOCH="${SOURCE_EPOCH}" \
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-800}" \
N_TEST="${N_TEST}" \
N_TEST_VIS="${N_TEST_VIS}" \
FINETUNE_LR="${FINETUNE_LR:-8e-7}" \
PHASE_WEIGHT="${PHASE_WEIGHT:-0.015}" \
PARAMETER_ANCHOR_RATE="${PARAMETER_ANCHOR_RATE:-5e-5}" \
GRIPPER_USE_HISTORY=true \
GRIPPER_HISTORY_UPDATE_RATE=1.0 \
GRIPPER_SIGMA="${GRIPPER_SIGMA:-0.10}" \
GRIPPER_SIGMA_HIGH="${GRIPPER_SIGMA_HIGH:-0.03}" \
GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-2.0}" \
RELEASE_GRIPPER_WEIGHT="${RELEASE_GRIPPER_WEIGHT:-3.5}" \
POST_RELEASE_GRIPPER_WEIGHT="${POST_RELEASE_GRIPPER_WEIGHT:-4.0}" \
RELEASE_TRANSLATION_WEIGHT="${RELEASE_TRANSLATION_WEIGHT:-0.8}" \
RELEASE_ROTATION_WEIGHT="${RELEASE_ROTATION_WEIGHT:-0.4}" \
PRE_RELEASE_BEFORE="${PRE_RELEASE_BEFORE:-18}" \
PRE_RELEASE_TRANSLATION_WEIGHT="${PRE_RELEASE_TRANSLATION_WEIGHT:-1.8}" \
PRE_RELEASE_ROTATION_WEIGHT="${PRE_RELEASE_ROTATION_WEIGHT:-1.2}" \
SETTLE_BEFORE="${SETTLE_BEFORE:-8}" \
SETTLE_AFTER="${SETTLE_AFTER:-3}" \
SETTLE_TRANSLATION_WEIGHT="${SETTLE_TRANSLATION_WEIGHT:-1.2}" \
SETTLE_ROTATION_WEIGHT="${SETTLE_ROTATION_WEIGHT:-0.8}" \
POST_RELEASE_AFTER="${POST_RELEASE_AFTER:-10}" \
EVENT_OVERSAMPLE_BEFORE="${EVENT_OVERSAMPLE_BEFORE:-20}" \
EVENT_OVERSAMPLE_AFTER="${EVENT_OVERSAMPLE_AFTER:-8}" \
EVENT_OVERSAMPLE_REPEATS=2 \
bash scripts/run_stack_release_finetune.sh
