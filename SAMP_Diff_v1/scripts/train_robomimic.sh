#!/usr/bin/env bash
# Train SAMP-Diff on supported robomimic low-dimensional PH tasks.
#
# Examples:
#   bash scripts/train_robomimic.sh --list
#   bash scripts/train_robomimic.sh --check lift_ph
#   DEVICE=cuda:1 bash scripts/train_robomimic.sh lift_ph
#   TASKS="lift_ph can_ph square_ph transport_ph" bash scripts/train_robomimic.sh all

set -euo pipefail

SUPPORTED_TASKS=(lift_ph can_ph square_ph transport_ph)
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-data/robomimic/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
RESUME="${RESUME:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
TASKS="${TASKS:-}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_robomimic.sh --list
  bash scripts/train_robomimic.sh --check TASK
  bash scripts/train_robomimic.sh TASK
  bash scripts/train_robomimic.sh all

Supported tasks: lift_ph can_ph square_ph transport_ph

Environment overrides:
  DEVICE, DATA_ROOT, OUTPUT_ROOT, NUM_EPOCHS, BATCH_SIZE,
  NUM_WORKERS, RESUME, WANDB_MODE, TASKS
EOF
}

is_supported() {
    local candidate="$1"
    local task
    for task in "${SUPPORTED_TASKS[@]}"; do
        [[ "$candidate" == "$task" ]] && return 0
    done
    return 1
}

dataset_path() {
    local task="$1"
    local name="${task%_ph}"
    printf '%s/%s/ph/low_dim_abs.hdf5\n' "$DATA_ROOT" "$name"
}

check_task() {
    local task="$1"
    local config="config_task/low_dim/${task}.yaml"
    local dataset
    dataset="$(dataset_path "$task")"

    is_supported "$task" || {
        echo "[ERROR] Unsupported task: $task" >&2
        return 1
    }
    [[ -f "$config" ]] || {
        echo "[ERROR] Missing config: $config" >&2
        return 1
    }
    [[ -f "$dataset" ]] || {
        echo "[ERROR] Missing dataset: $dataset" >&2
        return 1
    }

    python train.py +        --config-name="$task" +        --cfg job +        "task.dataset_path=$dataset" +        "training.device=$DEVICE" +        >/dev/null

    echo "[OK] $task"
    echo "     config : $config"
    echo "     dataset: $dataset"
}

train_task() {
    local task="$1"
    local dataset output_dir log_file
    dataset="$(dataset_path "$task")"
    output_dir="$OUTPUT_ROOT/$task"
    log_file="$output_dir/train.log"

    check_task "$task"
    mkdir -p "$output_dir"

    local args=(
        "task.dataset_path=$dataset"
        "hydra.run.dir=$output_dir"
        "training.device=$DEVICE"
        "training.resume=$RESUME"
        "logging.mode=$WANDB_MODE"
        "logging.name=samp_diff_${task}_v2c_step6"
        "policy.num_inference_steps=6"
        "policy.sigma=0.3"
        "policy.cold_start_prob=0.3"
        "policy.freq_split_low=0"
        "policy.freq_split_high=8"
        "policy.sigma_high=0.2"
    )
    [[ -n "$NUM_EPOCHS" ]] && args+=("training.num_epochs=$NUM_EPOCHS")
    if [[ -n "$BATCH_SIZE" ]]; then
        args+=("dataloader.batch_size=$BATCH_SIZE")
        args+=("val_dataloader.batch_size=$BATCH_SIZE")
    fi
    if [[ -n "$NUM_WORKERS" ]]; then
        args+=("dataloader.num_workers=$NUM_WORKERS")
        args+=("val_dataloader.num_workers=$NUM_WORKERS")
    fi

    echo "[TRAIN] task=$task device=$DEVICE output=$output_dir resume=$RESUME"
    python train.py --config-name="$task" "${args[@]}" 2>&1 | tee "$log_file"
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

command="${1:---help}"
case "$command" in
    --help|-h)
        usage
        ;;
    --list)
        printf '%s\n' "${SUPPORTED_TASKS[@]}"
        ;;
    --check)
        [[ $# -eq 2 ]] || {
            usage >&2
            exit 2
        }
        check_task "$2"
        ;;
    all)
        selected=(${TASKS:-${SUPPORTED_TASKS[*]}})
        for task in "${selected[@]}"; do
            train_task "$task"
        done
        ;;
    *)
        [[ $# -eq 1 ]] || {
            usage >&2
            exit 2
        }
        train_task "$command"
        ;;
esac
