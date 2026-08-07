#!/usr/bin/env bash
# Run official Diffusion Policy low-dimensional baselines for SAMP-Diff comparison.
#
# This script is meant to be launched from this repo, but it trains inside an
# external real-stanford/diffusion_policy checkout.
#
# Example:
#   DP_REPO=/workspace/diffusion_policy \
#   DEVICE=cuda:0 TASKS="pusht can_ph square_ph" \
#   bash scripts/run_dp_baseline_benchmarks.sh
#
# Full paper-style sweep:
#   DP_REPO=/workspace/diffusion_policy SEEDS="42 43 44" bash scripts/run_dp_baseline_benchmarks.sh

set -euo pipefail

DP_REPO="${DP_REPO:-../diffusion_policy}"
DEVICE="${DEVICE:-cuda:0}"
TASKS="${TASKS:-pusht lift_ph can_ph square_ph tool_hang_ph transport_ph}"
SEEDS="${SEEDS:-42 43 44}"
RUN_DEFAULT="${RUN_DEFAULT:-true}"
RUN_STEP6="${RUN_STEP6:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/samp_diff_baselines/dp}"
CONFIG_DIR="${CONFIG_DIR:-configs/samp_diff_dp_lowdim}"
DOWNLOAD_CONFIGS="${DOWNLOAD_CONFIGS:-true}"
RESUME="${RESUME:-true}"

BASE_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim"

usage() {
    cat <<'EOF'
Usage:
  DP_REPO=/path/to/real-stanford/diffusion_policy bash scripts/run_dp_baseline_benchmarks.sh

Environment overrides:
  DP_REPO          External Diffusion Policy repo. Default: ../diffusion_policy
  TASKS            Space-separated tasks. Default: pusht lift_ph can_ph square_ph tool_hang_ph transport_ph
  SEEDS            Space-separated seeds. Default: 42 43 44
  DEVICE           Training device. Default: cuda:0
  RUN_DEFAULT      Run official DP config. Default: true
  RUN_STEP6        Run DP with policy.num_inference_steps=6. Default: true
  WANDB_MODE       offline/online/disabled. Default: offline
  OUTPUT_ROOT      Output directory inside DP_REPO.
  CONFIG_DIR       Config cache directory inside DP_REPO.
  DOWNLOAD_CONFIGS Download official configs if missing. Default: true
  RESUME           Hydra training.resume override. Default: true

Expected data inside DP_REPO:
  data/pusht/pusht_cchi_v7_replay.zarr
  data/robomimic/datasets/{lift,can,square,tool_hang,transport}/ph/low_dim_abs.hdf5
EOF
}

task_config_url() {
    local task="$1"
    case "$task" in
        pusht)
            printf "%s/pusht/diffusion_policy_cnn/config.yaml\n" "$BASE_URL"
            ;;
        lift_ph|can_ph|square_ph|tool_hang_ph|transport_ph)
            printf "%s/%s/diffusion_policy_cnn/config.yaml\n" "$BASE_URL" "$task"
            ;;
        *)
            echo "[ERROR] Unsupported task: $task" >&2
            return 1
            ;;
    esac
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

download_file() {
    local url="$1"
    local dst="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail "$url" -o "$dst"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dst" "$url"
    else
        echo "[ERROR] Need curl or wget to download: $url" >&2
        return 1
    fi
}

ensure_config() {
    local task="$1"
    local config_path="$CONFIG_DIR/${task}.yaml"
    local url

    mkdir -p "$CONFIG_DIR"
    if [[ -f "$config_path" ]]; then
        printf "%s\n" "$config_path"
        return 0
    fi

    if [[ "$DOWNLOAD_CONFIGS" != "true" ]]; then
        echo "[ERROR] Missing config: $config_path" >&2
        return 1
    fi

    url="$(task_config_url "$task")"
    echo "[CONFIG] download $task -> $config_path"
    download_file "$url" "$config_path"
    printf "%s\n" "$config_path"
}

check_task() {
    local task="$1"
    local data
    data="$(dataset_path "$task")"

    ensure_config "$task" >/dev/null
    if [[ ! -e "$data" ]]; then
        echo "[WARN] Missing dataset for $task: $DP_REPO/$data"
        return 1
    fi
}

run_train() {
    local task="$1"
    local seed="$2"
    local variant="$3"
    shift 3

    check_task "$task" || return 0

    local output_dir="$OUTPUT_ROOT/${task}_${variant}_seed${seed}"
    mkdir -p "$output_dir"

    echo "================================================================="
    echo "[DP] task=$task variant=$variant seed=$seed"
    echo "  repo       : $DP_REPO"
    echo "  config_dir : $CONFIG_DIR"
    echo "  output     : $output_dir"
    echo "  device     : $DEVICE"
    echo "================================================================="

    python train.py \
        --config-dir="$CONFIG_DIR" \
        --config-name="${task}.yaml" \
        "training.seed=$seed" \
        "task.dataset.seed=$seed" \
        "training.device=$DEVICE" \
        "training.resume=$RESUME" \
        "logging.mode=$WANDB_MODE" \
        "hydra.run.dir=$output_dir" \
        "$@" \
        2>&1 | tee "$output_dir/train.log"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$DP_REPO/train.py" ]]; then
    echo "[ERROR] DP_REPO does not look like real-stanford/diffusion_policy: $DP_REPO" >&2
    echo "        Set DP_REPO=/path/to/diffusion_policy" >&2
    exit 1
fi

cd "$DP_REPO"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

for task in $TASKS; do
    for seed in $SEEDS; do
        if [[ "$RUN_DEFAULT" == "true" ]]; then
            run_train "$task" "$seed" "default"
        fi
        if [[ "$RUN_STEP6" == "true" ]]; then
            run_train "$task" "$seed" "step6" "policy.num_inference_steps=6"
        fi
    done
done

echo "================================================================="
echo "Diffusion Policy baseline sweep finished."
echo "Outputs: $DP_REPO/$OUTPUT_ROOT"
echo "================================================================="
