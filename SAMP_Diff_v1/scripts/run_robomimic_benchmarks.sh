#!/usr/bin/env bash
# Robomimic benchmark runner for SAMP-Diff paper experiments.
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/run_robomimic_benchmarks.sh
#
# Common runs:
#   RUN_FILTER=MAIN DEVICE=cuda:0 bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=DIAG DIAG_TASKS="lift_ph can_ph" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=STEPS STEP_TASKS="can_ph square_ph" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=ABLATION ABLATION_TASKS="can_ph" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=SEEDS SEED_TASKS="can_ph square_ph" SEEDS="42 43 44" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=ALL bash scripts/run_robomimic_benchmarks.sh
#
# RUN_FILTER options:
#   MAIN      Main 6-step SAMP-Diff on core robomimic PH tasks.
#   DIAG      Compact Lift/Can diagnostics for action horizon and model capacity.
#   STEPS     Inference-step scaling on Can/Square by default.
#   ABLATION  Small reviewer-facing ablations on Can by default.
#   SEEDS     Extra main-method seeds on Can/Square by default.
#   ALL       Run all of the above.

set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
NUM_EPOCHS="${NUM_EPOCHS:-6000}"
DATA_ROOT="${DATA_ROOT:-data/robomimic/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic_benchmark_6000}"
RUN_FILTER="${RUN_FILTER:-MAIN}"
RESUME="${RESUME:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"

# Main table: prioritize tasks that best show transfer beyond PushT.
MAIN_TASKS="${MAIN_TASKS:-can_ph square_ph transport_ph lift_ph}"

# Diagnostic grid: run this before large benchmark sweeps when Lift/Can plateau.
DIAG_TASKS="${DIAG_TASKS:-lift_ph can_ph}"

# Step scaling: good evidence that the method still works at low inference steps.
STEP_TASKS="${STEP_TASKS:-can_ph square_ph}"
STEP_VALUES="${STEP_VALUES:-2 4 6 8}"

# Small robomimic ablation grid. Keep this compact; PushT remains the main ablation task.
ABLATION_TASKS="${ABLATION_TASKS:-can_ph}"

# Multi-seed confirmation for the main method.
SEED_TASKS="${SEED_TASKS:-can_ph square_ph}"
SEEDS="${SEEDS:-42 43 44}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"

should_run_section() {
    local section="$1"
    [[ "${RUN_FILTER}" == "ALL" || "${RUN_FILTER}" == "${section}" ]]
}

dataset_path() {
    local task="$1"
    local name="${task%_ph}"
    printf "%s/%s/ph/low_dim_abs.hdf5\n" "${DATA_ROOT}" "${name}"
}

check_task() {
    local task="$1"
    local dataset
    dataset="$(dataset_path "${task}")"

    if [[ ! -f "config_task/low_dim/${task}.yaml" ]]; then
        echo "[WARN] missing config for ${task}; skipped"
        return 1
    fi
    if [[ ! -f "${dataset}" ]]; then
        echo "[WARN] missing dataset for ${task}: ${dataset}; skipped"
        return 1
    fi
}

run_train() {
    local section="$1"
    local task="$2"
    local exp_name="$3"
    local seed="$4"
    local purpose="$5"
    shift 5

    check_task "${task}" || return 0

    local dataset output_dir log_file rollout_arg
    dataset="$(dataset_path "${task}")"
    output_dir="${OUTPUT_ROOT}/${exp_name}"
    log_file="${output_dir}/train.log"
    mkdir -p "${output_dir}"

    rollout_arg="training.rollout_every=null"
    if [[ -n "${ROLLOUT_EVERY}" ]]; then
        rollout_arg="training.rollout_every=${ROLLOUT_EVERY}"
    fi

    local args=(
        "task.dataset_path=${dataset}"
        "hydra.run.dir=${output_dir}"
        "training.device=${DEVICE}"
        "training.num_epochs=${NUM_EPOCHS}"
        "training.resume=${RESUME}"
        "${rollout_arg}"
        "data_split.seed=${seed}"
        "training.seed=${seed}"
        "logging.mode=${WANDB_MODE}"
        "logging.project=samp_diff_robomimic_benchmark"
        "logging.name=${exp_name}"
        "policy.sigma=0.3"
        "policy.cold_start_prob=0.3"
        "policy.freq_split_low=0"
        "policy.freq_split_high=8"
        "policy.sigma_high=0.2"
        "$@"
    )

    if [[ -n "${BATCH_SIZE}" ]]; then
        args+=("dataloader.batch_size=${BATCH_SIZE}")
        args+=("val_dataloader.batch_size=${BATCH_SIZE}")
    fi
    if [[ -n "${NUM_WORKERS}" ]]; then
        args+=("dataloader.num_workers=${NUM_WORKERS}")
        args+=("val_dataloader.num_workers=${NUM_WORKERS}")
    fi

    echo "================================================================="
    echo "[${section}] ${exp_name}"
    echo "  task       : ${task}"
    echo "  purpose    : ${purpose}"
    echo "  seed       : ${seed}"
    echo "  target ep  : ${NUM_EPOCHS}"
    echo "  output_dir : ${output_dir}"
    echo "  resume     : ${RESUME}"
    echo "================================================================="

    python train.py --config-name="${task}" "${args[@]}" 2>&1 | tee "${log_file}"
}

if should_run_section "MAIN"; then
    for task in ${MAIN_TASKS}; do
        run_train \
            "MAIN" "${task}" "main_${task}_step6_seed42" "42" \
            "Main SAMP-Diff 6-step robomimic result" \
            "policy.num_inference_steps=6"
    done
fi

if should_run_section "DIAG"; then
    for task in ${DIAG_TASKS}; do
        run_train \
            "DIAG" "${task}" "diag_${task}_step6_act8_depth44_seed42" "42" \
            "Original robomimic config" \
            "policy.num_inference_steps=6" \
            "n_action_steps=8"

        run_train \
            "DIAG" "${task}" "diag_${task}_step6_act4_depth44_seed42" "42" \
            "More frequent closed-loop correction" \
            "policy.num_inference_steps=6" \
            "n_action_steps=4"

        run_train \
            "DIAG" "${task}" "diag_${task}_step6_act4_depth66_seed42" "42" \
            "Closed-loop correction with larger model capacity" \
            "policy.num_inference_steps=6" \
            "n_action_steps=4" \
            "policy.encoder_depth=6" \
            "policy.decoder_depth=6"
    done
fi

if should_run_section "STEPS"; then
    for task in ${STEP_TASKS}; do
        for steps in ${STEP_VALUES}; do
            run_train \
                "STEPS" "${task}" "steps_${task}_step${steps}_seed42" "42" \
                "Inference-step scaling" \
                "policy.num_inference_steps=${steps}"
        done
    done
fi

if should_run_section "ABLATION"; then
    for task in ${ABLATION_TASKS}; do
        run_train \
            "ABLATION" "${task}" "abl_${task}_full_warm_step6_seed42" "42" \
            "Full-frequency warm-start baseline" \
            "policy.num_inference_steps=6" \
            "policy.freq_split_high=16" \
            "policy.sigma_high=0.3"

        run_train \
            "ABLATION" "${task}" "abl_${task}_split08_random_high_step6_seed42" "42" \
            "High-frequency random control" \
            "policy.num_inference_steps=6" \
            "policy.freq_split_high=8" \
            "policy.sigma_high=-1.0"

        run_train \
            "ABLATION" "${task}" "abl_${task}_split08_sh02_step6_seed42" "42" \
            "Main weak high-frequency anchor" \
            "policy.num_inference_steps=6" \
            "policy.freq_split_high=8" \
            "policy.sigma_high=0.2"

        run_train \
            "ABLATION" "${task}" "abl_${task}_split08_sh05_step6_seed42" "42" \
            "Strong high-frequency perturbation control" \
            "policy.num_inference_steps=6" \
            "policy.freq_split_high=8" \
            "policy.sigma_high=0.5"

        run_train \
            "ABLATION" "${task}" "abl_${task}_cold0p0_step6_seed42" "42" \
            "No cold-start control" \
            "policy.num_inference_steps=6" \
            "policy.cold_start_prob=0.0"

        run_train \
            "ABLATION" "${task}" "abl_${task}_cold0p3_step6_seed42" "42" \
            "Main cold-start probability" \
            "policy.num_inference_steps=6" \
            "policy.cold_start_prob=0.3"
    done
fi

if should_run_section "SEEDS"; then
    for task in ${SEED_TASKS}; do
        for seed in ${SEEDS}; do
            run_train \
                "SEEDS" "${task}" "seed_${task}_step6_seed${seed}" "${seed}" \
                "Main SAMP-Diff 6-step multi-seed confirmation" \
                "policy.num_inference_steps=6"
        done
    done
fi

echo "================================================================="
echo "Robomimic benchmark training finished for RUN_FILTER=${RUN_FILTER}"
echo "Outputs: ${OUTPUT_ROOT}"
echo "Target epochs: ${NUM_EPOCHS}"
echo "================================================================="
