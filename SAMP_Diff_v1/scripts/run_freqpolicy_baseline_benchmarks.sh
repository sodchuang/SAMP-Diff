#!/usr/bin/env bash
# Run FreqPolicy low-dimensional baselines for SAMP-Diff comparison.
#
# FreqPolicy repos differ more than Diffusion Policy repos, so this script is a
# configurable wrapper. Set FREQ_REPO and the config-name pattern to match the
# checkout your collaborator uses.
#
# Example:
#   FREQ_REPO=/workspace/FreqPolicy \
#   CONFIG_DIR=configs/low_dim \
#   CONFIG_NAME_PATTERN='{task}.yaml' \
#   DEVICE=cuda:0 TASKS="pusht can_ph square_ph" \
#   bash scripts/run_freqpolicy_baseline_benchmarks.sh

set -euo pipefail

FREQ_REPO="${FREQ_REPO:-../FreqPolicy}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train.py}"
DEVICE="${DEVICE:-cuda:0}"
TASKS="${TASKS:-pusht lift_ph can_ph square_ph tool_hang_ph transport_ph}"
SEEDS="${SEEDS:-42 43 44}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/samp_diff_baselines/freqpolicy}"
CONFIG_DIR="${CONFIG_DIR:-config_task/low_dim}"
CONFIG_NAME_PATTERN="${CONFIG_NAME_PATTERN:-{task}.yaml}"
RUN_DEFAULT="${RUN_DEFAULT:-true}"
RUN_LOW_STEP="${RUN_LOW_STEP:-false}"
LOW_STEP_OVERRIDES="${LOW_STEP_OVERRIDES:-policy.num_inference_steps=6}"
COMMON_OVERRIDES="${COMMON_OVERRIDES:-}"
DEFAULT_OVERRIDES="${DEFAULT_OVERRIDES:-}"
RESUME_OVERRIDE="${RESUME_OVERRIDE:-++training.resume=true}"

usage() {
    cat <<'EOF'
Usage:
  FREQ_REPO=/path/to/FreqPolicy bash scripts/run_freqpolicy_baseline_benchmarks.sh

Environment overrides:
  FREQ_REPO             External FreqPolicy repo. Default: ../FreqPolicy
  TRAIN_ENTRY           Training entry inside FREQ_REPO. Default: train.py
  CONFIG_DIR            Hydra config dir inside FREQ_REPO. Default: config_task/low_dim
  CONFIG_NAME_PATTERN   Config file pattern. Use {task}. Default: {task}.yaml
  TASKS                 Space-separated tasks. Default: pusht lift_ph can_ph square_ph tool_hang_ph transport_ph
  SEEDS                 Space-separated seeds. Default: 42 43 44
  DEVICE                Training device. Default: cuda:0
  OUTPUT_ROOT           Output directory inside FREQ_REPO
  RUN_DEFAULT           Run FreqPolicy default setting. Default: true
  RUN_LOW_STEP          Also run low-step setting. Default: false
  LOW_STEP_OVERRIDES    Space-separated Hydra overrides for low-step run.
                        Default: policy.num_inference_steps=6
  COMMON_OVERRIDES      Extra Hydra overrides for every run.
  DEFAULT_OVERRIDES     Extra Hydra overrides for default runs.
  RESUME_OVERRIDE       Resume override. Default: training.resume=true

Expected common data layout inside FREQ_REPO, unless config overrides it:
  data/pusht/pusht_cchi_v7_replay.zarr
  data/robomimic/datasets/{lift,can,square,tool_hang,transport}/ph/low_dim_abs.hdf5

Notes:
  If the FreqPolicy repo does not use Hydra --config-dir/--config-name, set
  TRAIN_ENTRY to a small adapter script in that repo, or copy this wrapper and
  replace run_train().
EOF
}

config_name() {
    local task="$1"
    printf "%s\n" "${CONFIG_NAME_PATTERN//\{task\}/$task}"
}

dataset_path() {
    local task="$1"
    case "$task" in
        pusht)
            printf "data/pusht/pusht_cchi_v7_replay.zarr\n"
            ;;
        *_ph)
            printf "data/robomimic/datasets/%s/ph/low_dim_abs.hdf5\n" "${task%_ph}"
            ;;
    esac
}

dataset_override() {
    local task="$1"
    case "$task" in
        pusht)
            printf "++task.dataset.zarr_path=%s\n" "$(dataset_path "$task")"
            ;;
        *_ph)
            printf "++task.dataset_path=%s\n" "$(dataset_path "$task")"
            ;;
    esac
}

check_task() {
    local task="$1"
    local cfg="$CONFIG_DIR/$(config_name "$task")"
    local data
    data="$(dataset_path "$task")"

    if [[ ! -f "$cfg" ]]; then
        echo "[WARN] Missing config for $task: $FREQ_REPO/$cfg"
        return 1
    fi
    if [[ ! -e "$data" ]]; then
        echo "[WARN] Missing dataset for $task: $FREQ_REPO/$data"
        return 1
    fi
}

split_words() {
    local text="$1"
    if [[ -n "$text" ]]; then
        # shellcheck disable=SC2206
        SPLIT_RESULT=($text)
    else
        SPLIT_RESULT=()
    fi
}

run_train() {
    local task="$1"
    local seed="$2"
    local variant="$3"
    local extra_overrides="$4"

    check_task "$task" || return 0

    local output_dir="$OUTPUT_ROOT/${task}_${variant}_seed${seed}"
    local cfg_name
    cfg_name="$(config_name "$task")"
    mkdir -p "$output_dir"

    split_words "$COMMON_OVERRIDES"
    local common=("${SPLIT_RESULT[@]}")
    split_words "$extra_overrides"
    local extra=("${SPLIT_RESULT[@]}")

    echo "================================================================="
    echo "[FreqPolicy] task=$task variant=$variant seed=$seed"
    echo "  repo       : $FREQ_REPO"
    echo "  config     : $CONFIG_DIR/$cfg_name"
    echo "  output     : $output_dir"
    echo "  device     : $DEVICE"
    echo "================================================================="

    python "$TRAIN_ENTRY" \
        --config-dir="$CONFIG_DIR" \
        --config-name="$cfg_name" \
        "++training.seed=$seed" \
        "++data_split.seed=$seed" \
        "++training.device=$DEVICE" \
        "$RESUME_OVERRIDE" \
        "++logging.mode=$WANDB_MODE" \
        "hydra.run.dir=$output_dir" \
        "$(dataset_override "$task")" \
        "${common[@]}" \
        "${extra[@]}" \
        2>&1 | tee "$output_dir/train.log"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$FREQ_REPO/$TRAIN_ENTRY" ]]; then
    echo "[ERROR] FREQ_REPO/TRAIN_ENTRY not found: $FREQ_REPO/$TRAIN_ENTRY" >&2
    echo "        Set FREQ_REPO=/path/to/FreqPolicy and TRAIN_ENTRY if needed." >&2
    exit 1
fi

cd "$FREQ_REPO"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

for task in $TASKS; do
    for seed in $SEEDS; do
        if [[ "$RUN_DEFAULT" == "true" ]]; then
            run_train "$task" "$seed" "default" "$DEFAULT_OVERRIDES"
        fi
        if [[ "$RUN_LOW_STEP" == "true" ]]; then
            run_train "$task" "$seed" "low_step" "$LOW_STEP_OVERRIDES"
        fi
    done
done

echo "================================================================="
echo "FreqPolicy baseline sweep finished."
echo "Outputs: $FREQ_REPO/$OUTPUT_ROOT"
echo "================================================================="
