#!/usr/bin/env bash
set -Eeuo pipefail

# Final paper evaluation only. This script never calls train.py.
# Required protocol: fixed checkpoint, 100 test episodes, no train episodes.

DEVICE="${DEVICE:-cuda:0}"
N_ENVS="${N_ENVS:-20}"
N_TEST="${N_TEST:-100}"
SQUARE_TEST_VIS="${SQUARE_TEST_VIS:-50}"
OTHER_TEST_VIS="${OTHER_TEST_VIS:-0}"
STACK_TEST_VIS="${STACK_TEST_VIS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/paper1_unified_eval100_seed42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-data/outputs}"
RUN_STACK="${RUN_STACK:-false}"
OVERWRITE="${OVERWRITE:-false}"

mkdir -p "${OUTPUT_ROOT}"

resolve_checkpoint() {
    local explicit_path="$1"
    local run_regex="$2"

    if [[ -n "${explicit_path}" ]]; then
        [[ -f "${explicit_path}" ]] || return 2
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    python scripts/find_best_eval_checkpoint.py \
        --root "${CHECKPOINT_ROOT}" \
        --run-regex "${run_regex}"
}

run_eval() {
    local label="$1"
    local checkpoint="$2"
    local n_test_vis="$3"
    local output_dir="${OUTPUT_ROOT}/${label}"

    if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo -e "${label}\tMISSING\t\t\t" >> "${OUTPUT_ROOT}/summary.tsv"
        echo "[WARN] ${label}: checkpoint not found; skipped." >&2
        return 0
    fi

    if [[ -e "${output_dir}" && "${OVERWRITE}" != "true" ]]; then
        echo "[SKIP] ${label}: ${output_dir} already exists. Set OVERWRITE=true to rerun."
    else
        echo
        echo "[EVAL] ${label}"
        echo "[EVAL] checkpoint=${checkpoint}"
        python eval.py \
            -c "${checkpoint}" \
            -o "${output_dir}" \
            -d "${DEVICE}" \
            --overwrite \
            --n-envs "${N_ENVS}" \
            --n-test "${N_TEST}" \
            --n-train 0 \
            --n-test-vis "${n_test_vis}" \
            --n-train-vis 0 \
            --profile-actions
    fi

    python - "${label}" "${checkpoint}" "${output_dir}" >> "${OUTPUT_ROOT}/summary.tsv" <<'PY'
import json
import sys
from pathlib import Path

label, checkpoint, output_dir = sys.argv[1:]
log_path = Path(output_dir) / "eval_log.json"
if not log_path.is_file():
    print(f"{label}\tNO_EVAL_LOG\t\t\t{checkpoint}")
    raise SystemExit(0)

data = json.loads(log_path.read_text())
score = data.get("test/mean_score", data.get("test_mean_score", ""))
latency = data.get("eval/action_latency_ms_mean", data.get("action_latency_ms", ""))
hz = data.get("eval/action_hz", data.get("action_hz", ""))
print(f"{label}\tOK\t{score}\t{latency}\t{hz}\t{checkpoint}")
PY
}

make_square_failure_manifest() {
    local output_dir="${OUTPUT_ROOT}/square_ph"
    local manifest="${OUTPUT_ROOT}/square_failure_breakdown.tsv"

    python - "${output_dir}" "${manifest}" <<'PY'
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
videos = sorted(
    p for p in output_dir.rglob("*")
    if p.suffix.lower() in {".mp4", ".avi", ".mov"}
)[:50]

with manifest.open("w", encoding="utf-8") as f:
    f.write("index\tvideo\tfailure_type\tnote\n")
    for index, video in enumerate(videos, 1):
        f.write(f"{index}\t{video}\t\t\n")

print(f"[DONE] Square failure manifest: {manifest} ({len(videos)} videos)")
PY
}

printf 'run\tstatus\ttest_mean_score\tlatency_ms\thz\tcheckpoint\n' \
    > "${OUTPUT_ROOT}/summary.tsv"

# A. Most important stress-case repeat first.
SQUARE_PH_CKPT="$(resolve_checkpoint "${SQUARE_PH_CKPT:-}" 'hg_gc_task_semantic_v4_square/square_ph_hg_gc_fix_seed42' || true)"
run_eval "square_ph" "${SQUARE_PH_CKPT}" "${SQUARE_TEST_VIS}"
make_square_failure_manifest

# B. Unified PH/MH evaluation.
LIFT_PH_CKPT="$(resolve_checkpoint "${LIFT_PH_CKPT:-}" 'hg_gc_generalization_v2/lift_ph_hg_gc_fix_seed42' || true)"
LIFT_MH_CKPT="$(resolve_checkpoint "${LIFT_MH_CKPT:-}" 'lift_mh_(full|hg_gc_fix)_seed42' || true)"
CAN_PH_CKPT="$(resolve_checkpoint "${CAN_PH_CKPT:-}" 'hg_gc_task_semantic_v4_aligned/can_ph_hg_gc_fix_seed42' || true)"
CAN_MH_CKPT="$(resolve_checkpoint "${CAN_MH_CKPT:-}" 'can_mh_(full|hg_gc_fix)_seed42' || true)"

run_eval "lift_ph" "${LIFT_PH_CKPT}" "${OTHER_TEST_VIS}"
run_eval "lift_mh" "${LIFT_MH_CKPT}" "${OTHER_TEST_VIS}"
run_eval "can_ph" "${CAN_PH_CKPT}" "${OTHER_TEST_VIS}"
run_eval "can_mh" "${CAN_MH_CKPT}" "${OTHER_TEST_VIS}"

# C. Optional Stack verification. No training is performed.
if [[ "${RUN_STACK}" == "true" ]]; then
    STACK_SEMANTIC_CKPT="$(resolve_checkpoint "${STACK_SEMANTIC_CKPT:-}" 'stack_d1_semantic_delta_v1_seed42' || true)"
    STACK_PRECISION_CKPT="$(resolve_checkpoint "${STACK_PRECISION_CKPT:-}" 'stack_d1_semantic_delta_precision_v2_seed42' || true)"
    STACK_RELEASE_CKPT="$(resolve_checkpoint "${STACK_RELEASE_CKPT:-}" 'stack_d1_release_focus_v1_seed42' || true)"

    run_eval "stack_semantic_delta" "${STACK_SEMANTIC_CKPT}" "${STACK_TEST_VIS}"
    run_eval "stack_delta_precision" "${STACK_PRECISION_CKPT}" "${STACK_TEST_VIS}"
    run_eval "stack_release_focus" "${STACK_RELEASE_CKPT}" "${STACK_TEST_VIS}"
fi

echo
echo "[DONE] ${OUTPUT_ROOT}/summary.tsv"
column -t -s $'\t' "${OUTPUT_ROOT}/summary.tsv" 2>/dev/null \
    || cat "${OUTPUT_ROOT}/summary.tsv"
