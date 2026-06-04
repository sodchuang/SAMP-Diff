#!/usr/bin/env bash
# Run PushT low-dim SAMP-Diff ablations for the paper tables.
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/run_experiments.sh
#
# Useful overrides:
#   DEVICE=cuda:1 NUM_EPOCHS=4000 N_EPISODES=50 bash scripts/run_experiments.sh
#   RUN_FILTER=A1 bash scripts/run_experiments.sh
#   RUN_FILTER=A2 SKIP_TRAIN_IF_CKPT=true bash scripts/run_experiments.sh
#
# RUN_FILTER options:
#   ALL, A1, A2, A3_SPLIT, A3_SH, A3_SIGMA, A3_COLD
#
# Output:
#   results/pusht_ablation_all/
#     experiments.tsv       experiment settings
#     metrics_summary.tsv   one row per finished experiment
#     raw_metrics/*.txt     full compute_motion_metrics.py output
#     train_logs/*.log      full training output

set -euo pipefail

BASE_CFG="${BASE_CFG:-pusht}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPOCHS="${NUM_EPOCHS:-4000}"
N_EPISODES="${N_EPISODES:-50}"
METRICS_SCRIPT="${METRICS_SCRIPT:-compute_motion_metrics.py}"
RESULTS_DIR="${RESULTS_DIR:-results/pusht_ablation_all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/pusht_ablation_all}"
RUN_FILTER="${RUN_FILTER:-ALL}"
SKIP_TRAIN_IF_CKPT="${SKIP_TRAIN_IF_CKPT:-false}"

mkdir -p "${RESULTS_DIR}/raw_metrics" "${RESULTS_DIR}/train_logs" "${OUTPUT_ROOT}"

EXPERIMENTS_TSV="${RESULTS_DIR}/experiments.tsv"
SUMMARY_TSV="${RESULTS_DIR}/metrics_summary.tsv"

printf "section\texperiment\tsteps\tfreq_split_low\tfreq_split_high\tsigma\tsigma_high\tcold_start_prob\tpurpose\n" > "${EXPERIMENTS_TSV}"
printf "section\texperiment\tcheckpoint\tmean_score\tmean_score_std\tpath_length\tpath_length_std\tjerk_cost\tjerk_cost_std\tdiscontinuity\tdiscontinuity_std\n" > "${SUMMARY_TSV}"

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

metric_mean() {
    local label="$1"
    local file="$2"
    awk -v label="$label" '
        index($0, label) {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^[-+]?([0-9]*[.])?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    print $i
                    exit
                }
            }
        }
    ' "$file"
}

metric_std() {
    local label="$1"
    local file="$2"
    awk -v label="$label" '
        index($0, label) {
            count = 0
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^[-+]?([0-9]*[.])?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    count++
                    value[count] = $i
                }
            }
            if (count >= 2) {
                print value[2]
            }
            exit
        }
    ' "$file"
}

append_experiment_row() {
    local section="$1"
    local exp_name="$2"
    local steps="$3"
    local split_low="$4"
    local split_high="$5"
    local sigma="$6"
    local sigma_high="$7"
    local cold_start="$8"
    local purpose="$9"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${section}" "${exp_name}" "${steps}" "${split_low}" "${split_high}" \
        "${sigma}" "${sigma_high}" "${cold_start}" "${purpose}" >> "${EXPERIMENTS_TSV}"
}

run_exp() {
    local section="$1"
    local exp_name="$2"
    local steps="$3"
    local split_low="$4"
    local split_high="$5"
    local sigma="$6"
    local sigma_high="$7"
    local cold_start="$8"
    local purpose="$9"

    append_experiment_row \
        "${section}" "${exp_name}" "${steps}" "${split_low}" "${split_high}" \
        "${sigma}" "${sigma_high}" "${cold_start}" "${purpose}"

    if ! should_run_section "${section}"; then
        return
    fi

    local exp_dir="${OUTPUT_ROOT}/${exp_name}"
    local train_log="${RESULTS_DIR}/train_logs/${exp_name}.log"
    local metrics_out="${RESULTS_DIR}/raw_metrics/${exp_name}_metrics.txt"

    mkdir -p "${exp_dir}"

    echo "================================================================="
    echo "[${section}] ${exp_name}"
    echo "  purpose         : ${purpose}"
    echo "  output_dir      : ${exp_dir}"
    echo "  steps           : ${steps}"
    echo "  freq_split      : [${split_low}, ${split_high})"
    echo "  sigma           : ${sigma}"
    echo "  sigma_high      : ${sigma_high}"
    echo "  cold_start_prob : ${cold_start}"
    echo "================================================================="

    local ckpt=""
    ckpt="$(latest_checkpoint "${exp_dir}" || true)"

    if [[ "${SKIP_TRAIN_IF_CKPT}" == "true" && -n "${ckpt}" ]]; then
        echo "[SKIP TRAIN] Found existing checkpoint: ${ckpt}"
    else
        python train.py \
            --config-name="${BASE_CFG}" \
            "hydra.run.dir=${exp_dir}" \
            "training.device=${DEVICE}" \
            "training.num_epochs=${NUM_EPOCHS}" \
            "training.resume=false" \
            "logging.name=${exp_name}" \
            "logging.project=samp_diff_pusht_ablation" \
            "logging.mode=online" \
            "policy.num_inference_steps=${steps}" \
            "policy.freq_split_low=${split_low}" \
            "policy.freq_split_high=${split_high}" \
            "policy.sigma=${sigma}" \
            "policy.sigma_high=${sigma_high}" \
            "policy.cold_start_prob=${cold_start}" \
            "task.env_runner.n_test=50" \
            "task.env_runner.n_train=6" \
            "task.env_runner.max_steps=300" \
            2>&1 | tee "${train_log}"
    fi

    ckpt="$(latest_checkpoint "${exp_dir}" || true)"
    if [[ -z "${ckpt}" ]]; then
        echo "[WARN] ${exp_name}: checkpoint not found; metrics skipped"
        return
    fi

    echo "[METRICS] ${exp_name}: ${ckpt}"
    python "${METRICS_SCRIPT}" \
        -c "${ckpt}" \
        -d "${DEVICE}" \
        --n_episodes "${N_EPISODES}" \
        --max_steps 300 \
        2>&1 | tee "${metrics_out}"

    local mean_score mean_score_std path_length path_length_std jerk_cost jerk_cost_std discontinuity discontinuity_std
    mean_score="$(metric_mean "Mean Score" "${metrics_out}")"
    mean_score_std="$(metric_std "Mean Score" "${metrics_out}")"
    path_length="$(metric_mean "Path Length" "${metrics_out}")"
    path_length_std="$(metric_std "Path Length" "${metrics_out}")"
    jerk_cost="$(metric_mean "Jerk Cost" "${metrics_out}")"
    jerk_cost_std="$(metric_std "Jerk Cost" "${metrics_out}")"
    discontinuity="$(metric_mean "Discontinuity" "${metrics_out}")"
    discontinuity_std="$(metric_std "Discontinuity" "${metrics_out}")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${section}" "${exp_name}" "${ckpt}" \
        "${mean_score:-NA}" "${mean_score_std:-NA}" \
        "${path_length:-NA}" "${path_length_std:-NA}" \
        "${jerk_cost:-NA}" "${jerk_cost_std:-NA}" \
        "${discontinuity:-NA}" "${discontinuity_std:-NA}" \
        >> "${SUMMARY_TSV}"
}

# A1. Main 4-step ablation.
#run_exp A1 freq_full_warm_step4        4 0 16 0.3  0.3 0.3 "Full-frequency warm-start baseline"
#run_exp A1 freq_split04_random_step4   4 0  4 0.3 -1.0 0.3 "Low 4 frequencies warm, high frequencies random"
#run_exp A1 freq_split08_random_step4   4 0  8 0.3 -1.0 0.3 "Low 8 frequencies warm, high frequencies random"
#run_exp A1 v2c_split08_sh02_step4      4 0  8 0.3  0.2 0.3 "Main method: low 8 warm, high-frequency weak anchor"
run_exp A1 freq_split04_sh02_step4     4 0  4 0.3  0.2 0.3 "Split-range control with weak high-frequency anchor"
run_exp A1 freq_split08_sh05_step4     4 0  8 0.3  0.5 0.3 "High-frequency perturbation too strong control"

# A2. v2c inference-step ablation.
run_exp A2 v2c_split08_sh02_step1      1 0  8 0.3  0.2 0.3 "Extreme low-step inference"
run_exp A2 v2c_split08_sh02_step2      2 0  8 0.3  0.2 0.3 "Low-step inference comparable to few-step baselines"
run_exp A2 v2c_split08_sh02_step4      4 0  8 0.3  0.2 0.3 "Main 4-step setting"
run_exp A2 v2c_split08_sh02_step6      6 0  8 0.3  0.2 0.3 "Previous 6-step setting"
run_exp A2 v2c_split08_sh02_step8      8 0  8 0.3  0.2 0.3 "More-step inference upper check"

# A3-1. freq_split_high sweep.
run_exp A3_SPLIT coef_split2_step4      4 0  2 0.3  0.2 0.3 "Frequency split high sweep: 2"
run_exp A3_SPLIT coef_split4_step4      4 0  4 0.3  0.2 0.3 "Frequency split high sweep: 4"
run_exp A3_SPLIT coef_split8_step4      4 0  8 0.3  0.2 0.3 "Frequency split high sweep: 8"
run_exp A3_SPLIT coef_split12_step4     4 0 12 0.3  0.2 0.3 "Frequency split high sweep: 12"
run_exp A3_SPLIT coef_split16_step4     4 0 16 0.3  0.2 0.3 "Frequency split high sweep: 16"

# A3-2. sigma_high sweep.
run_exp A3_SH coef_sh_random_step4      4 0  8 0.3 -1.0 0.3 "High frequencies fully random"
run_exp A3_SH coef_sh0p0_step4          4 0  8 0.3  0.0 0.3 "High frequencies fully preserved"
run_exp A3_SH coef_sh0p1_step4          4 0  8 0.3  0.1 0.3 "Weak high-frequency perturbation"
run_exp A3_SH coef_sh0p2_step4          4 0  8 0.3  0.2 0.3 "Main sigma_high setting"
run_exp A3_SH coef_sh0p3_step4          4 0  8 0.3  0.3 0.3 "Medium high-frequency perturbation"
run_exp A3_SH coef_sh0p5_step4          4 0  8 0.3  0.5 0.3 "Strong high-frequency perturbation"

# A3-3. sigma sweep.
run_exp A3_SIGMA coef_sigma0p1_step4    4 0  8 0.1  0.2 0.3 "Warm-start noise sigma sweep: 0.1"
run_exp A3_SIGMA coef_sigma0p2_step4    4 0  8 0.2  0.2 0.3 "Warm-start noise sigma sweep: 0.2"
run_exp A3_SIGMA coef_sigma0p3_step4    4 0  8 0.3  0.2 0.3 "Main sigma setting"
run_exp A3_SIGMA coef_sigma0p5_step4    4 0  8 0.5  0.2 0.3 "Warm-start noise sigma sweep: 0.5"

# A3-4. cold_start_prob sweep.
run_exp A3_COLD coef_cold0p0_step4      4 0  8 0.3  0.2 0.0 "Cold-start probability sweep: 0.0"
run_exp A3_COLD coef_cold0p1_step4      4 0  8 0.3  0.2 0.1 "Cold-start probability sweep: 0.1"
run_exp A3_COLD coef_cold0p3_step4      4 0  8 0.3  0.2 0.3 "Main cold-start probability"
run_exp A3_COLD coef_cold0p5_step4      4 0  8 0.3  0.2 0.5 "Cold-start probability sweep: 0.5"

echo "================================================================="
echo "All requested experiments for RUN_FILTER=${RUN_FILTER} are done."
echo "Experiment settings : ${EXPERIMENTS_TSV}"
echo "Metrics summary     : ${SUMMARY_TSV}"
echo "Raw metrics         : ${RESULTS_DIR}/raw_metrics"
echo "Train logs          : ${RESULTS_DIR}/train_logs"
echo "================================================================="
