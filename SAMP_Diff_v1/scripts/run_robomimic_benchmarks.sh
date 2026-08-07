#!/usr/bin/env bash
# Robomimic benchmark runner for SAMP-Diff paper experiments.
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/run_robomimic_benchmarks.sh
#
# Paper positioning:
#   human-designed semantic structure + learned conditional flow coordination.
#
# Recommended order when seed 42 already exists:
#   RUN_FILTER=MAIN SEEDS="43 44" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=STRUCTURE STRUCTURE_SEEDS="42 43 44" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=DATA DATA_SEEDS="42" bash scripts/run_robomimic_benchmarks.sh
#   RUN_FILTER=STEPS bash scripts/run_robomimic_benchmarks.sh
#
# RUN_FILTER options:
#   MAIN      Semantic-prior SAMP on every task and every requested seed.
#   STRUCTURE Human semantic grouping vs homogeneous and wrong grouping.
#   DATA      Data-efficiency comparison for semantic vs homogeneous priors.
#   DIAG      Compact Lift/Can diagnostics for action horizon and model capacity.
#   STEPS     Secondary inference-step analysis; not the main contribution.
#   PAPER     MAIN + STRUCTURE + DATA.
#   ALL       PAPER + DIAG + STEPS.

set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
NUM_EPOCHS="${NUM_EPOCHS:-6000}"
DATA_ROOT="${DATA_ROOT:-data/robomimic/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/semantic_prior_generalization}"
RUN_FILTER="${RUN_FILTER:-MAIN}"
RESUME="${RESUME:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
TEST_START_SEED="${TEST_START_SEED:-100000}"
N_TEST="${N_TEST:-50}"

# Main table: cross-task applicability of the same hybrid design philosophy.
MAIN_TASKS="${MAIN_TASKS:-can_ph square_ph transport_ph lift_ph}"
SEEDS="${SEEDS:-42 43 44}"

# Diagnostic grid: run this before large benchmark sweeps when Lift/Can plateau.
DIAG_TASKS="${DIAG_TASKS:-lift_ph can_ph}"

# Step scaling: good evidence that the method still works at low inference steps.
STEP_TASKS="${STEP_TASKS:-can_ph square_ph}"
STEP_VALUES="${STEP_VALUES:-2 4 6 8}"

# Structure ablation directly tests the paper claim. Can represents
# pick-and-place; Square represents rotation-sensitive insertion.
STRUCTURE_TASKS="${STRUCTURE_TASKS:-can_ph square_ph}"
STRUCTURE_SEEDS="${STRUCTURE_SEEDS:-42 43 44}"

# Data efficiency tests whether human structure is especially useful when
# demonstrations are scarce. Start with seed 42, then repeat the selected
# budget with 43/44 for the final paper.
DATA_TASKS="${DATA_TASKS:-can_ph square_ph}"
DATA_SEEDS="${DATA_SEEDS:-42}"
DATA_EPISODES="${DATA_EPISODES:-10 25 50}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"

should_run_section() {
    local section="$1"
    if [[ "${RUN_FILTER}" == "ALL" || "${RUN_FILTER}" == "${section}" ]]; then
        return 0
    fi
    if [[ "${RUN_FILTER}" == "PAPER" ]]; then
        [[ "${section}" == "MAIN" \
            || "${section}" == "STRUCTURE" \
            || "${section}" == "DATA" ]]
        return
    fi
    return 1
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
        "task.env_runner.test_start_seed=${TEST_START_SEED}"
        "task.env_runner.n_test=${N_TEST}"
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
    echo "  test bank  : ${TEST_START_SEED}..$((TEST_START_SEED + N_TEST - 1))"
    echo "  output_dir : ${output_dir}"
    echo "  resume     : ${RESUME}"
    echo "================================================================="

    python train.py --config-name="${task}" "${args[@]}" 2>&1 | tee "${log_file}"
}

if should_run_section "MAIN"; then
    for task in ${MAIN_TASKS}; do
        for seed in ${SEEDS}; do
            run_train \
                "MAIN" "${task}" "semantic_${task}_step6_seed${seed}" "${seed}" \
                "Human-designed action semantics with learned flow coordination" \
                "policy.num_inference_steps=6"
        done
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

if should_run_section "STRUCTURE"; then
    for task in ${STRUCTURE_TASKS}; do
        for seed in ${STRUCTURE_SEEDS}; do
            # The semantic result is produced by MAIN with the same task/seed.
            # Homogeneous keeps the learned flow model but removes human action
            # grouping, so all action dimensions share one source prior.
            run_train \
                "STRUCTURE" "${task}" \
                "structure_${task}_homogeneous_step6_seed${seed}" "${seed}" \
                "Homogeneous hand-agnostic source prior control" \
                "policy.num_inference_steps=6" \
                "policy.action_group_spectral_params=null"

            # Parameter count and group sizes match the semantic model, but
            # translation and rotation dimensions are deliberately assigned to
            # the wrong groups. This isolates semantic correctness from merely
            # having multiple parameter groups.
            run_train \
                "STRUCTURE" "${task}" \
                "structure_${task}_wrong_groups_step6_seed${seed}" "${seed}" \
                "Wrong-group control with matched model capacity" \
                "policy.num_inference_steps=6" \
                "policy.action_group_spectral_params.translation.indices=[3,4,5]" \
                "policy.action_group_spectral_params.rotation.indices=[0,1,2,6,7,8]" \
                "policy.action_group_spectral_params.gripper.indices=[9]"
        done
    done
fi

if should_run_section "DATA"; then
    for task in ${DATA_TASKS}; do
        for seed in ${DATA_SEEDS}; do
            for episodes in ${DATA_EPISODES}; do
                run_train \
                    "DATA" "${task}" \
                    "data_${task}_semantic_ep${episodes}_seed${seed}" "${seed}" \
                    "Semantic prior with limited demonstrations" \
                    "policy.num_inference_steps=6" \
                    "data_split.max_train_episodes=${episodes}"

                run_train \
                    "DATA" "${task}" \
                    "data_${task}_homogeneous_ep${episodes}_seed${seed}" "${seed}" \
                    "Homogeneous prior with limited demonstrations" \
                    "policy.num_inference_steps=6" \
                    "data_split.max_train_episodes=${episodes}" \
                    "policy.action_group_spectral_params=null"
            done
        done
    done
fi

echo "================================================================="
echo "Robomimic benchmark training finished for RUN_FILTER=${RUN_FILTER}"
echo "Outputs: ${OUTPUT_ROOT}"
echo "Target epochs: ${NUM_EPOCHS}"
echo "================================================================="
