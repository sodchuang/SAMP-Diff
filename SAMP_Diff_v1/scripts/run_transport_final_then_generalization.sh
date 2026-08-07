#!/usr/bin/env bash
# One final Transport recipe, then automatically move to single-task dataset
# generalization when Transport still has no full success.

set -euo pipefail

SEED="${SEED:-42}"
TRANSPORT_SOURCE="${TRANSPORT_SOURCE:-mh}"
# The red-object pick is a simple single-arm primitive.  With the balanced
# recipe it should be visible around epoch 500-700; 1600 leaves additional
# room for hammer retention/handoff without spending another 3000+ epochs on
# a run whose full Transport score remains zero.
TRANSPORT_EPOCHS="${TRANSPORT_EPOCHS:-1200}"
TRANSPORT_MIN_SCORE="${TRANSPORT_MIN_SCORE:-0.02}"
TRANSPORT_RUN_SUFFIX="${TRANSPORT_RUN_SUFFIX:-precision_grasp_v7}"
TRANSPORT_OUTPUT_ROOT="${TRANSPORT_OUTPUT_ROOT:-data/outputs/transport_final_v7}"

# Override this list when some of these controls already exist.  Keeping the
# output root separate prevents old results from being overwritten.
FALLBACK_JOBS="${FALLBACK_JOBS:-lift:mh lift:mg can:mh can:mg square:mh}"
FALLBACK_EPOCHS="${FALLBACK_EPOCHS:-4000}"
FALLBACK_OUTPUT_ROOT="${FALLBACK_OUTPUT_ROOT:-data/outputs/dataset_source_generalization_stage2}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[FINAL_TRANSPORT] training ${TRANSPORT_SOURCE} for ${TRANSPORT_EPOCHS} epochs"
SEED="${SEED}" \
JOBS="transport:${TRANSPORT_SOURCE}" \
VARIANTS="FULL" \
NUM_EPOCHS="${TRANSPORT_EPOCHS}" \
ROLLOUT_EVERY="50" \
CHECKPOINT_EVERY="50" \
N_TEST="50" \
N_TEST_VIS="8" \
RESUME="false" \
TRANSPORT_RUN_SUFFIX="${TRANSPORT_RUN_SUFFIX}" \
OUTPUT_ROOT="${TRANSPORT_OUTPUT_ROOT}" \
bash scripts/run_dataset_source_generalization.sh

run_dir="${TRANSPORT_OUTPUT_ROOT}/transport_${TRANSPORT_SOURCE}_full_seed${SEED}_${TRANSPORT_RUN_SUFFIX}"
log_path="${run_dir}/logs.json.txt"

best_score="$(python - "${log_path}" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
best = 0.0
if path.is_file():
    for line in path.open(encoding="utf-8", errors="ignore"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        value = row.get("test/mean_score")
        if isinstance(value, (int, float)) and math.isfinite(value):
            best = max(best, float(value))
print(best)
PY
)"

echo "[FINAL_TRANSPORT] best test/mean_score=${best_score}"
if python - "${best_score}" "${TRANSPORT_MIN_SCORE}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
then
    echo "[FINAL_TRANSPORT] success signal found; preserving the run and stopping here."
    exit 0
fi

echo "[FINAL_TRANSPORT] no success by epoch ${TRANSPORT_EPOCHS}; moving to single-task generalization."
if [[ -z "${FALLBACK_JOBS// }" ]]; then
    echo "[FINAL_TRANSPORT] FALLBACK_JOBS is empty; finished."
    exit 0
fi

SEED="${SEED}" \
JOBS="${FALLBACK_JOBS}" \
VARIANTS="FULL" \
NUM_EPOCHS="${FALLBACK_EPOCHS}" \
ROLLOUT_EVERY="50" \
CHECKPOINT_EVERY="50" \
RESUME="false" \
OUTPUT_ROOT="${FALLBACK_OUTPUT_ROOT}" \
bash scripts/run_dataset_source_generalization.sh

echo "[GENERALIZATION] complete: ${FALLBACK_JOBS}"
