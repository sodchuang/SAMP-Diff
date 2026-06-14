
#!/usr/bin/env bash
# Batch runner for eval.py on PushT ablation checkpoints.
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/run_eval.sh
#
# Useful overrides:
#   DEVICE=cuda:1 bash scripts/run_eval.sh
#   RUN_FILTER=A2 bash scripts/run_eval.sh
#   START_AFTER_EXP=v2c_split08_sh02_step4 bash scripts/run_eval.sh
#   SKIP_DONE=true bash scripts/run_eval.sh
#   N_EPISODES=100 MAX_STEPS=500 bash scripts/run_eval.sh

set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
RUN_FILTER="${RUN_FILTER:-ALL}"
START_AFTER_EXP="${START_AFTER_EXP:-}"
SKIP_DONE="${SKIP_DONE:-true}"
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-300}"
METRICS_SCRIPT="${METRICS_SCRIPT:-compute_motion_metrics.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/pusht_ablation_all}"
EVAL_ROOT="${EVAL_ROOT:-results/eval_output}"

norm_path() {
    local p="$1"
    # strip trailing slash for robust string compare
    p="${p%/}"
    printf "%s\n" "$p"
}

OUTPUT_ROOT_NORM="$(norm_path "${OUTPUT_ROOT}")"
EVAL_ROOT_NORM="$(norm_path "${EVAL_ROOT}")"

if [[ "${OUTPUT_ROOT_NORM}" == "${EVAL_ROOT_NORM}" \
   || "${EVAL_ROOT_NORM}" == "${OUTPUT_ROOT_NORM}"/* \
   || "${OUTPUT_ROOT_NORM}" == "${EVAL_ROOT_NORM}"/* ]]; then
    echo "[ERROR] OUTPUT_ROOT and EVAL_ROOT overlap."
    echo "  OUTPUT_ROOT=${OUTPUT_ROOT_NORM}"
    echo "  EVAL_ROOT=${EVAL_ROOT_NORM}"
    echo "Set different roots to avoid overwrite."
    exit 1
fi

mkdir -p "${EVAL_ROOT}"

should_run_section() {
    local section="$1"
    [[ "${RUN_FILTER}" == "ALL" || "${RUN_FILTER}" == "${section}" ]]
}

latest_checkpoint() {
    local exp_dir="$1"
    if [[ -f "${exp_dir}/checkpoints/latest.ckpt" ]]; then
        printf "%s\n" "${exp_dir}/checkpoints/latest.ckpt"
    else
        find "${exp_dir}/checkpoints" -maxdepth 1 -name "*.ckpt" -type f 2>/dev/null \
            | sort -r \
            | head -n 1
    fi
}

REACHED_START=false

run_eval() {
    local section="$1"
    local exp_name="$2"

    if [[ -n "${START_AFTER_EXP}" && "${REACHED_START}" == "false" ]]; then
        if [[ "${exp_name}" == "${START_AFTER_EXP}" ]]; then
            REACHED_START=true
            echo "[START MARK] reached ${START_AFTER_EXP}; continue from next experiment"
            return
        fi
        echo "[SKIP BEFORE START] ${exp_name}"
        return
    fi

    if ! should_run_section "${section}"; then
        return
    fi

    local exp_dir="${OUTPUT_ROOT}/${exp_name}"
    local out_dir="${EVAL_ROOT}/${exp_name}"
    local ckpt=""
    ckpt="$(latest_checkpoint "${exp_dir}" || true)"

    if [[ -z "${ckpt}" ]]; then
        echo "[WARN] ${exp_name}: checkpoint not found in ${exp_dir}/checkpoints"
        return
    fi

    if [[ "${SKIP_DONE}" == "true" && -f "${out_dir}/eval_log.json" ]]; then
        echo "[SKIP DONE] ${exp_name}: ${out_dir}/eval_log.json exists"
        return
    fi

    rm -rf "${out_dir}"

    echo "================================================================="
    echo "[${section}] ${exp_name}"
    echo "  checkpoint : ${ckpt}"
    echo "  output_dir : ${out_dir}"
    echo "  device     : ${DEVICE}"
    echo "  episodes   : ${N_EPISODES}"
    echo "  max_steps  : ${MAX_STEPS}"
    echo "================================================================="

    python eval.py \
        -c "${ckpt}" \
        -o "${out_dir}" \
        -d "${DEVICE}"

    python "${METRICS_SCRIPT}" \
        -c "${ckpt}" \
        -d "${DEVICE}" \
        --n_episodes "${N_EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        > "${out_dir}/motion_metrics.txt"

    echo "[DONE] ${exp_name} -> ${out_dir}/eval_log.json"
}

# A1
run_eval A1 freq_full_warm_step4
run_eval A1 freq_split04_random_step4
run_eval A1 freq_split08_random_step4
run_eval A1 v2c_split08_sh02_step4
run_eval A1 freq_split04_sh02_step4
run_eval A1 freq_split08_sh05_step4

# A2
run_eval A2 v2c_split08_sh02_step1
run_eval A2 v2c_split08_sh02_step2
run_eval A2 v2c_split08_sh02_step4
run_eval A2 v2c_split08_sh02_step6
run_eval A2 v2c_split08_sh02_step8

# A3-1
run_eval A3_SPLIT coef_split2_step4
run_eval A3_SPLIT coef_split4_step4
run_eval A3_SPLIT coef_split8_step4
run_eval A3_SPLIT coef_split12_step4
run_eval A3_SPLIT coef_split16_step4

# A3-2
run_eval A3_SH coef_sh_random_step4
run_eval A3_SH coef_sh0p0_step4
run_eval A3_SH coef_sh0p1_step4
run_eval A3_SH coef_sh0p2_step4
run_eval A3_SH coef_sh0p3_step4
run_eval A3_SH coef_sh0p5_step4

# A3-3
run_eval A3_SIGMA coef_sigma0p1_step4
run_eval A3_SIGMA coef_sigma0p2_step4
run_eval A3_SIGMA coef_sigma0p3_step4
run_eval A3_SIGMA coef_sigma0p5_step4

# A3-4
run_eval A3_COLD coef_cold0p0_step4
run_eval A3_COLD coef_cold0p1_step4
run_eval A3_COLD coef_cold0p3_step4
run_eval A3_COLD coef_cold0p5_step4

if [[ -n "${START_AFTER_EXP}" && "${REACHED_START}" == "false" ]]; then
    echo "[WARN] START_AFTER_EXP not found: ${START_AFTER_EXP}"
fi

echo "================================================================="
echo "Eval finished for RUN_FILTER=${RUN_FILTER}"
echo "Outputs: ${EVAL_ROOT}"
echo "================================================================="