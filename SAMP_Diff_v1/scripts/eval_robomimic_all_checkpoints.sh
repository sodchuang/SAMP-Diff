#!/usr/bin/env bash
# Evaluate checkpoints for every robomimic benchmark experiment directory.
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/eval_robomimic_all_checkpoints.sh
#
# Useful overrides:
#   OUTPUT_ROOT=data/outputs/robomimic_benchmark_6000 DEVICE=cuda:0 bash scripts/eval_robomimic_all_checkpoints.sh
#   EXP_FILTER="main_*" bash scripts/eval_robomimic_all_checkpoints.sh
#   MAX_CKPTS=5 bash scripts/eval_robomimic_all_checkpoints.sh

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic_benchmark_6000}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ROOT="${EVAL_ROOT:-results/checkpoint_sweep}"
EXP_FILTER="${EXP_FILTER:-*}"
SKIP_DONE="${SKIP_DONE:-true}"
MAX_CKPTS="${MAX_CKPTS:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    echo "[ERROR] Missing OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
    exit 1
fi

mkdir -p "${EVAL_ROOT}"
best_summary="${EVAL_ROOT}/best_checkpoints.tsv"
printf "experiment\ttask\tcheckpoint\tepoch\ttest_mean_score\ttrain_mean_score\toutput_dir\n" > "${best_summary}"

infer_task() {
    local exp_name="$1"
    case "${exp_name}" in
        *lift_ph*) printf "lift_ph\n" ;;
        *can_ph*) printf "can_ph\n" ;;
        *square_ph*) printf "square_ph\n" ;;
        *tool_hang_ph*) printf "tool_hang_ph\n" ;;
        *transport_ph*) printf "transport_ph\n" ;;
        *) printf "\n" ;;
    esac
}

mapfile -t exp_dirs < <(
    find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${EXP_FILTER}" \
        | sort -V
)

if [[ "${#exp_dirs[@]}" -eq 0 ]]; then
    echo "[ERROR] No experiment directories found under ${OUTPUT_ROOT} matching ${EXP_FILTER}" >&2
    exit 1
fi

for exp_dir in "${exp_dirs[@]}"; do
    exp_name="$(basename "${exp_dir}")"

    if [[ ! -d "${exp_dir}/checkpoints" ]]; then
        echo "[SKIP] ${exp_name}: no checkpoints directory"
        continue
    fi

    task="$(infer_task "${exp_name}")"
    if [[ -z "${task}" ]]; then
        echo "[WARN] ${exp_name}: cannot infer task from experiment name; task check disabled"
    fi

    echo "================================================================="
    echo "[SWEEP] ${exp_name}"
    echo "  exp_dir : ${exp_dir}"
    echo "  task    : ${task:-UNKNOWN}"
    echo "  device  : ${DEVICE}"
    echo "================================================================="

    EXP_DIR="${exp_dir}" \
    EXPECT_TASK="${task}" \
    DEVICE="${DEVICE}" \
    EVAL_ROOT="${EVAL_ROOT}" \
    SKIP_DONE="${SKIP_DONE}" \
    MAX_CKPTS="${MAX_CKPTS}" \
    bash scripts/eval_robomimic_checkpoints.sh

    ranked="${EVAL_ROOT}/${exp_name}/checkpoint_scores_ranked.tsv"
    if [[ -f "${ranked}" ]]; then
        best_row="$(tail -n +2 "${ranked}" | head -n 1)"
        if [[ -n "${best_row}" ]]; then
            printf "%s\t%s\t%s\n" "${exp_name}" "${task:-UNKNOWN}" "${best_row}" >> "${best_summary}"
        fi
    fi
done

echo "================================================================="
echo "All checkpoint sweeps finished."
echo "Best summary: ${best_summary}"
echo ""
echo "Best checkpoints:"
column -t -s $'\t' "${best_summary}" 2>/dev/null || cat "${best_summary}"
echo "================================================================="
