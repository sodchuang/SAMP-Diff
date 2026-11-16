#!/usr/bin/env bash
# Resume the promising Stack v5 run, preserve its best checkpoint, and switch
# to gripper-history finetuning only after a verified score plateau.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
SOURCE_RUN="${SOURCE_RUN:-stack_delta_aligned_release_v5_seed42}"
CONTINUE_TO_EPOCH="${CONTINUE_TO_EPOCH:-2000}"
CONTINUE_V5="${CONTINUE_V5:-false}"
PATIENCE_EVALS="${PATIENCE_EVALS:-4}"
MIN_RECENT_SCORE="${MIN_RECENT_SCORE:-0.10}"
KEEP_V5_ON_SCORE="${KEEP_V5_ON_SCORE:-false}"
RELEASE_RUN="${RELEASE_RUN:-stack_delta_release_history_v5_2_seed42}"
RELEASE_EPOCHS="${RELEASE_EPOCHS:-1000}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINT_DIR="${OUTPUT_ROOT}/${SOURCE_RUN}/checkpoints"
if [[ ! -f "${CHECKPOINT_DIR}/latest.ckpt" ]]; then
    echo "[ERROR] missing resumable v5 checkpoint: ${CHECKPOINT_DIR}/latest.ckpt" >&2
    exit 1
fi

if [[ "${CONTINUE_V5}" == "true" ]]; then
    echo "[STACK-GUARD] resume=${SOURCE_RUN} target_epoch=${CONTINUE_TO_EPOCH}"
    RUN_NAME="${SOURCE_RUN}" \
    NUM_EPOCHS="${CONTINUE_TO_EPOCH}" \
    RESUME_EXISTING=true \
    bash scripts/run_stack_delta_parameter_prior.sh
else
    echo "[STACK-GUARD] skip unchanged v5 continuation; use its best scored checkpoint"
fi

readarray -t DECISION < <(python - "${CHECKPOINT_DIR}" "${PATIENCE_EVALS}" "${MIN_RECENT_SCORE}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
patience = int(sys.argv[2])
minimum = float(sys.argv[3])
pattern = re.compile(r"epoch=(\d+)_test_score=([-+0-9.eE]+)_")
rows = []
for path in root.glob("epoch=*_test_score=*.ckpt"):
    match = pattern.search(path.name)
    if match:
        rows.append((int(match.group(1)), float(match.group(2)), path))
if not rows:
    raise SystemExit("[ERROR] no scored checkpoint available after v5 continuation")
rows.sort(key=lambda item: item[0])
best = max(rows, key=lambda item: (item[1], item[0]))
recent = rows[-patience:]
recent_max = max(score for _, score, _ in recent)
print(best[2])
print(f"{best[1]:.9g}")
print(f"{recent_max:.9g}")
print("keep" if recent_max >= minimum else "switch")
PY
)

BEST_CKPT="${DECISION[0]}"
BEST_SCORE="${DECISION[1]}"
RECENT_MAX="${DECISION[2]}"
ACTION="${DECISION[3]}"

echo "[STACK-GUARD] best_score=${BEST_SCORE} best_checkpoint=${BEST_CKPT}"
echo "[STACK-GUARD] recent_${PATIENCE_EVALS}_max=${RECENT_MAX} threshold=${MIN_RECENT_SCORE}"

if [[ "${ACTION}" == "keep" && "${KEEP_V5_ON_SCORE}" == "true" ]]; then
    echo "[STACK-GUARD] KEEP v5: recent evaluations still contain real successes."
    echo "[STACK-GUARD] No automatic architecture switch was made."
    exit 0
fi

if [[ "${ACTION}" == "keep" ]]; then
    echo "[STACK-GUARD] score is promising but not visually confirmed; release-history remains required."
else
    echo "[STACK-GUARD] score plateau confirmed."
fi
echo "[STACK-GUARD] switching from the latest aligned v5 checkpoint to release history."
SOURCE_CKPT="${CHECKPOINT_DIR}/latest.ckpt" \
SOURCE_SELECTION=latest \
RUN_NAME="${RELEASE_RUN}" \
FINETUNE_EPOCHS="${RELEASE_EPOCHS}" \
N_TEST="${N_TEST:-30}" \
N_TEST_VIS="${N_TEST_VIS:-30}" \
bash scripts/run_stack_release_history_v5_1.sh
