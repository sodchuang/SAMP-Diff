#!/usr/bin/env bash
# Evaluate every checkpoint in a robomimic experiment directory and rank them.
#
# Usage:
#   cd SAMP_Diff_v1
#   EXP_DIR=data/outputs/robomimic_benchmark_6000/main_can_ph_step6_seed42 \
#   bash scripts/eval_robomimic_checkpoints.sh
#
# Useful overrides:
#   DEVICE=cuda:1 bash scripts/eval_robomimic_checkpoints.sh
#   MAX_CKPTS=10 bash scripts/eval_robomimic_checkpoints.sh
#   SKIP_DONE=false bash scripts/eval_robomimic_checkpoints.sh
#   EXPECT_TASK=can_ph bash scripts/eval_robomimic_checkpoints.sh

set -euo pipefail

EXP_DIR="${EXP_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ROOT="${EVAL_ROOT:-results/checkpoint_sweep}"
SKIP_DONE="${SKIP_DONE:-true}"
MAX_CKPTS="${MAX_CKPTS:-}"
EXPECT_TASK="${EXPECT_TASK:-}"

if [[ -z "${EXP_DIR}" ]]; then
    echo "[ERROR] EXP_DIR is required." >&2
    echo "Example:" >&2
    echo "  EXP_DIR=data/outputs/robomimic_benchmark_6000/main_can_ph_step6_seed42 bash scripts/eval_robomimic_checkpoints.sh" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ ! -d "${EXP_DIR}/checkpoints" ]]; then
    echo "[ERROR] Missing checkpoints directory: ${EXP_DIR}/checkpoints" >&2
    exit 1
fi

exp_name="$(basename "${EXP_DIR}")"
out_root="${EVAL_ROOT}/${exp_name}"
summary="${out_root}/checkpoint_scores.tsv"
ranked="${out_root}/checkpoint_scores_ranked.tsv"

mkdir -p "${out_root}"
printf "checkpoint\tepoch\ttest_mean_score\ttrain_mean_score\toutput_dir\n" > "${summary}"

mapfile -t ckpts < <(
    find "${EXP_DIR}/checkpoints" -maxdepth 1 -type f -name "*.ckpt" \
        ! -name "latest.ckpt" \
        | sort -V
)

if [[ -n "${MAX_CKPTS}" ]]; then
    ckpts=("${ckpts[@]:0:${MAX_CKPTS}}")
fi

if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "[ERROR] No .ckpt files found in ${EXP_DIR}/checkpoints" >&2
    exit 1
fi

for ckpt in "${ckpts[@]}"; do
    if [[ -n "${EXPECT_TASK}" ]]; then
        python - "${ckpt}" "${EXPECT_TASK}" <<'PY'
import pathlib
import sys
import torch
import dill

ckpt = sys.argv[1]
expect_task = sys.argv[2]
expect_name = expect_task[:-3] if expect_task.endswith("_ph") else expect_task

payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
cfg = payload["cfg"]
dataset_path = str(cfg.task.dataset_path)
actual_name = pathlib.PurePosixPath(dataset_path.replace("\\", "/")).parts[-3]

if actual_name != expect_name:
    raise SystemExit(
        f"[ERROR] checkpoint/task mismatch: expected {expect_task}, "
        f"but checkpoint dataset is {dataset_path}"
    )

print(f"[CHECK] {pathlib.Path(ckpt).name}: {dataset_path}")
PY
    fi

    ckpt_base="$(basename "${ckpt}" .ckpt)"
    safe_name="${ckpt_base//[^A-Za-z0-9._=-]/_}"
    out_dir="${out_root}/${safe_name}"
    log_json="${out_dir}/eval_log.json"

    echo "================================================================="
    echo "[EVAL] ${ckpt}"
    echo "  output_dir: ${out_dir}"
    echo "  device    : ${DEVICE}"
    echo "================================================================="

    if [[ "${SKIP_DONE}" == "true" && -f "${log_json}" ]]; then
        echo "[SKIP DONE] ${log_json}"
    else
        rm -rf "${out_dir}"
        python eval.py \
            -c "${ckpt}" \
            -o "${out_dir}" \
            -d "${DEVICE}"
    fi

    python - "${ckpt}" "${out_dir}" "${log_json}" >> "${summary}" <<'PY'
import json
import pathlib
import re
import sys

ckpt = sys.argv[1]
out_dir = sys.argv[2]
log_json = pathlib.Path(sys.argv[3])

if not log_json.is_file():
    raise SystemExit(f"missing eval log: {log_json}")

data = json.loads(log_json.read_text())
name = pathlib.Path(ckpt).name
epoch_match = re.search(r"epoch=(\d+)", name)
epoch = epoch_match.group(1) if epoch_match else "NA"
test_score = data.get("test/mean_score", "NA")
train_score = data.get("train/mean_score", "NA")
print(f"{ckpt}\t{epoch}\t{test_score}\t{train_score}\t{out_dir}")
PY
done

{
    head -n 1 "${summary}"
    tail -n +2 "${summary}" | sort -t $'\t' -k3,3gr -k2,2nr
} > "${ranked}"

echo "================================================================="
echo "Checkpoint sweep finished."
echo "Summary: ${summary}"
echo "Ranked : ${ranked}"
echo ""
echo "Top checkpoints:"
head -n 6 "${ranked}"
echo "================================================================="
