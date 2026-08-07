#!/usr/bin/env bash
# Dataset-source generalization for the semantic action-module paper.
#
# Existing PH results are the reference. The default jobs train the missing
# MH / MG controls with identical compute, seed, action horizon and inference
# steps. Official robomimic MG data exists only for Lift and Can.
#
# Required input layout (absolute-action datasets only):
#   data/robomimic/datasets/<task>/<source>/low_dim_abs.hdf5
#
# Default jobs:
#   lift:mh lift:mg can:mh can:mg square:mh transport:mh
#
# Examples:
#   bash scripts/run_dataset_source_generalization.sh --check
#   nohup bash scripts/run_dataset_source_generalization.sh \
#     > dataset_source_generalization_seed42.log 2>&1 &
#   JOBS="transport:mh" bash scripts/run_dataset_source_generalization.sh
#   VARIANTS="HOMO FULL" JOBS="can:mh square:mh" \
#     bash scripts/run_dataset_source_generalization.sh

set -euo pipefail

MODE="${1:-train}"
SEED="${SEED:-42}"
JOBS="${JOBS:-lift:mh lift:mg can:mh can:mg square:mh transport:mh}"
VARIANTS="${VARIANTS:-FULL}"
NUM_EPOCHS="${NUM_EPOCHS:-7000}"
TRAINING_STOP_EPOCH="$((NUM_EPOCHS + 1))"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-200}"
TEST_START_SEED="${TEST_START_SEED:-100000}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-data/robomimic/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/dataset_source_generalization}"
RESUME="${RESUME:-false}"
WANDB_MODE="${WANDB_MODE:-disabled}"
TRANSPORT_SPLIT_HISTORY="${TRANSPORT_SPLIT_HISTORY:-true}"
# Gripper is a discrete phase switch. Reusing an open-state trajectory as its
# next source prior can prevent the policy from closing after it reaches the
# object. Keep pose histories independent per arm, but regenerate both gripper
# priors for every chunk; their phase losses remain isolated per arm.
TRANSPORT_GRIPPER_HISTORY="${TRANSPORT_GRIPPER_HISTORY:-false}"
TRANSPORT_RUN_SUFFIX="${TRANSPORT_RUN_SUFFIX:-precision_grasp_v7}"
# Conservative Transport correction.  Keep the verified v4 pose objective and
# only add a small hold signal.  The stronger v6 group balancing caused pose
# drift because two one-dimensional gripper groups dominated the base loss.
TRANSPORT_GROUP_BALANCED_LOSS="${TRANSPORT_GROUP_BALANCED_LOSS:-false}"
TRANSPORT_PHASE_LOSS_WEIGHT="${TRANSPORT_PHASE_LOSS_WEIGHT:-0.03}"
TRANSPORT_ROBOT0_HOLD_GRIPPER_WEIGHT="${TRANSPORT_ROBOT0_HOLD_GRIPPER_WEIGHT:-0.5}"
TRANSPORT_ROBOT1_HOLD_GRIPPER_WEIGHT="${TRANSPORT_ROBOT1_HOLD_GRIPPER_WEIGHT:-0.8}"
TRANSPORT_ROBOT0_HOLD_POSE_WEIGHT="${TRANSPORT_ROBOT0_HOLD_POSE_WEIGHT:-0.0}"
TRANSPORT_ROBOT1_HOLD_POSE_WEIGHT="${TRANSPORT_ROBOT1_HOLD_POSE_WEIGHT:-0.0}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "${OUTPUT_ROOT}"

dataset_path() {
    local task="$1"
    local source="$2"
    local dir="${DATA_ROOT}/${task}/${source}"
    local candidate

    for candidate in \
        "${dir}/low_dim_abs.hdf5" \
        "${dir}/low_dim_v141_abs.hdf5"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    # Accept another clearly named absolute-action file, but never silently
    # fall back to low_dim.hdf5 (the latter contains delta actions).
    candidate="$(find "${dir}" -maxdepth 1 -type f \
        -iname '*low*abs*.hdf5' -print -quit 2>/dev/null || true)"
    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    return 1
}

validate_dataset() {
    local task="$1"
    local source="$2"
    local path="$3"
    local expected_action_dim="7"
    [[ "${task}" == "transport" ]] && expected_action_dim="14"

    python - "${path}" "${task}" "${expected_action_dim}" <<'PY'
import json
import pathlib
import sys

import h5py

path = pathlib.Path(sys.argv[1])
task = sys.argv[2]
expected_action_dim = int(sys.argv[3])

with h5py.File(path, "r") as f:
    if "data" not in f or "demo_0" not in f["data"]:
        raise SystemExit(f"[ERROR] invalid robomimic HDF5: {path}")
    demos = f["data"]
    actions = demos["demo_0/actions"]
    if actions.shape[-1] != expected_action_dim:
        raise SystemExit(
            f"[ERROR] {path}: action_dim={actions.shape[-1]}, "
            f"expected={expected_action_dim} for {task}"
        )
    obs = demos["demo_0/obs"]
    required = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    if task == "transport":
        required += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
    missing = [key for key in required if key not in obs]
    if missing:
        raise SystemExit(f"[ERROR] {path}: missing observations {missing}")

    env_name = "unknown"
    raw_env_args = demos.attrs.get("env_args", f.attrs.get("env_args", None))
    if raw_env_args is not None:
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        try:
            env_name = json.loads(raw_env_args).get("env_name", "unknown")
        except Exception:
            pass

    print(
        f"[OK] {task}: demos={len(demos)}, action_dim={actions.shape[-1]}, "
        f"env={env_name}, path={path}"
    )
PY
}

configure_task() {
    local task="$1"
    case "${task}" in
        lift|can)
            config_name="${task}_ph"
            translation_indices="[0,1,2]"
            rotation_indices="[3,4,5,6,7,8]"
            gripper_indices="[9]"
            history_training_mode="aligned"
            rotation_split="12"
            rotation_sigma="0.25"
            rotation_sigma_high="0.3"
            phase_loss_weight="0.03"
            ;;
        square)
            config_name="square_ph"
            translation_indices="[0,1,2]"
            rotation_indices="[3,4,5,6,7,8]"
            gripper_indices="[9]"
            # Preserve the verified Square GC result (best 0.68).
            history_training_mode="legacy_shift"
            rotation_split="16"
            rotation_sigma="0.15"
            rotation_sigma_high="0.15"
            phase_loss_weight="0.03"
            ;;
        transport)
            config_name="transport_ph"
            translation_indices="[0,1,2,10,11,12]"
            rotation_indices="[3,4,5,6,7,8,13,14,15,16,17,18]"
            gripper_indices="[9,19]"
            history_training_mode="aligned"
            rotation_split="12"
            rotation_sigma="0.25"
            rotation_sigma_high="0.3"
            phase_loss_weight="${TRANSPORT_PHASE_LOSS_WEIGHT}"
            ;;
        *)
            echo "[ERROR] unsupported task: ${task}" >&2
            return 1
            ;;
    esac
}

# Complete preflight before spending GPU time.
declare -a resolved_jobs=()
for job in ${JOBS}; do
    task="${job%%:*}"
    source="${job##*:}"
    if [[ "${task}" == "${source}" ]]; then
        echo "[ERROR] invalid JOBS item '${job}', expected task:source" >&2
        exit 2
    fi
    if [[ "${source}" == "mg" && "${task}" != "lift" && "${task}" != "can" ]]; then
        echo "[ERROR] official MG data is unavailable for ${task}" >&2
        exit 2
    fi
    configure_task "${task}"
    if ! data="$(dataset_path "${task}" "${source}")"; then
        echo "[ERROR] missing absolute-action dataset for ${task}:${source}" >&2
        echo "        expected under ${DATA_ROOT}/${task}/${source}/" >&2
        echo "        refusing low_dim.hdf5 because it contains delta actions" >&2
        exit 3
    fi
    validate_dataset "${task}" "${source}" "${data}"
    resolved_jobs+=("${task}:${source}:${data}")
done

if [[ "${MODE}" == "--check" || "${MODE}" == "check" ]]; then
    echo "[CHECK] all requested datasets and semantic maps are valid"
    exit 0
fi
if [[ "${MODE}" != "train" ]]; then
    echo "Usage: $0 [train|--check]" >&2
    exit 2
fi

for resolved in "${resolved_jobs[@]}"; do
    task="${resolved%%:*}"
    remainder="${resolved#*:}"
    source="${remainder%%:*}"
    data="${remainder#*:}"
    configure_task "${task}"

    for variant in ${VARIANTS}; do
        variant_upper="$(printf '%s' "${variant}" | tr '[:lower:]' '[:upper:]')"
        run_name="${task}_${source}_${variant_upper,,}_seed${SEED}"
        if [[ "${task}" == "transport" && -n "${TRANSPORT_RUN_SUFFIX}" ]]; then
            run_name="${run_name}_${TRANSPORT_RUN_SUFFIX}"
        fi
        output_dir="${OUTPUT_ROOT}/${run_name}"
        mkdir -p "${output_dir}"

        args=(
            "hydra.run.dir=${output_dir}"
            "task.dataset_path=${data}"
            "task.dataset_type=${source}"
            "training.device=${DEVICE}"
            "training.seed=${SEED}"
            "data_split.seed=${SEED}"
            "data_split.max_train_episodes=${MAX_TRAIN_EPISODES}"
            "training.resume=${RESUME}"
            "training.num_epochs=${TRAINING_STOP_EPOCH}"
            "training.rollout_every=${ROLLOUT_EVERY}"
            "training.checkpoint_every=${CHECKPOINT_EVERY}"
            "task.env_runner.n_envs=${N_ENVS}"
            "task.env_runner.n_test=${N_TEST}"
            "task.env_runner.n_test_vis=${N_TEST_VIS}"
            "task.env_runner.test_start_seed=${TEST_START_SEED}"
            "logging.mode=${WANDB_MODE}"
            "logging.project=samp_dataset_source_generalization"
            "logging.name=${run_name}"
            "++task.dataset.episode_prefix_close_threshold=auto"
            "++task.dataset.episode_prefix_close_is_greater=auto"
            "n_action_steps=4"
            "policy.num_inference_steps=6"
            "policy.sigma=0.3"
            "policy.cold_start_prob=0.3"
            "policy.freq_split_low=0"
            "policy.freq_split_high=8"
            "policy.sigma_high=0.2"
        )

        case "${variant_upper}" in
            FULL)
                if [[ "${task}" == "transport" && "${TRANSPORT_GROUP_BALANCED_LOSS}" == "true" ]]; then
                    # Balance semantic groups before adding the sparse phase
                    # objective.  Robot1 receives a modest priority because
                    # the short red-object pick/place segment was previously
                    # diluted by the longer robot0 lid/hammer trajectory.
                    args+=(
                        "++policy.action_group_loss_weights={robot0_pose:{indices:[0,1,2,3,4,5,6,7,8],weight:1.0},robot0_gripper:{indices:[9],weight:1.7},robot1_pose:{indices:[10,11,12,13,14,15,16,17,18],weight:1.15},robot1_gripper:{indices:[19],weight:2.2}}"
                    )
                else
                    args+=("++policy.action_group_loss_weights=null")
                fi
                args+=(
                    "++policy.action_group_spectral_params.translation.indices=${translation_indices}"
                    "++policy.action_group_spectral_params.translation.freq_split_high=8"
                    "++policy.action_group_spectral_params.translation.sigma=0.3"
                    "++policy.action_group_spectral_params.translation.sigma_high=0.2"
                    "++policy.action_group_spectral_params.rotation.indices=${rotation_indices}"
                    "++policy.action_group_spectral_params.rotation.freq_split_high=${rotation_split}"
                    "++policy.action_group_spectral_params.rotation.sigma=${rotation_sigma}"
                    "++policy.action_group_spectral_params.rotation.sigma_high=${rotation_sigma_high}"
                    "++policy.action_group_spectral_params.gripper.indices=${gripper_indices}"
                    "++policy.action_group_spectral_params.gripper.freq_split_high=16"
                    "++policy.action_group_spectral_params.gripper.sigma=0.5"
                    "++policy.action_group_spectral_params.gripper.sigma_high=0.5"
                    "++policy.action_group_history_params.enabled=true"
                    "++policy.action_group_history_params.shift=4"
                    "++policy.history_training_mode=${history_training_mode}"
                    "++policy.action_phase_loss_params.enabled=true"
                    "++policy.action_phase_loss_params.weight=${phase_loss_weight}"
                    "++policy.action_phase_loss_params.gripper_indices=${gripper_indices}"
                    "++policy.action_phase_loss_params.close_threshold=0.0"
                    "++policy.action_phase_loss_params.close_is_greater=true"
                    "++policy.action_phase_loss_params.transition_radius=2"
                    "++policy.action_phase_loss_params.translation_indices=${translation_indices}"
                    "++policy.action_phase_loss_params.translation_weight=1.5"
                    "++policy.action_phase_loss_params.rotation_indices=${rotation_indices}"
                    "++policy.action_phase_loss_params.rotation_weight=1.3"
                    "++policy.action_phase_loss_params.gripper_weight=1.5"
                    "++policy.action_phase_loss_params.release_enabled=false"
                )

                if [[ "${task}" == "transport" && "${TRANSPORT_SPLIT_HISTORY}" == "true" ]]; then
                    # Transport is asynchronous: keep one independent history
                    # buffer for every arm x action-semantic pair. In particular,
                    # do not replace both grippers with fresh Gaussian noise at
                    # every action chunk unless explicitly requested.
                    args+=(
                        "++policy.action_group_history_params.groups.robot0_translation.indices=[0,1,2]"
                        "++policy.action_group_history_params.groups.robot0_translation.use_history=true"
                        "++policy.action_group_history_params.groups.robot0_translation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.robot0_rotation.indices=[3,4,5,6,7,8]"
                        "++policy.action_group_history_params.groups.robot0_rotation.use_history=true"
                        "++policy.action_group_history_params.groups.robot0_rotation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.robot0_gripper.indices=[9]"
                        "++policy.action_group_history_params.groups.robot0_gripper.use_history=${TRANSPORT_GRIPPER_HISTORY}"
                        "++policy.action_group_history_params.groups.robot0_gripper.update_rate=1.0"
                        "++policy.action_group_history_params.groups.robot1_translation.indices=[10,11,12]"
                        "++policy.action_group_history_params.groups.robot1_translation.use_history=true"
                        "++policy.action_group_history_params.groups.robot1_translation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.robot1_rotation.indices=[13,14,15,16,17,18]"
                        "++policy.action_group_history_params.groups.robot1_rotation.use_history=true"
                        "++policy.action_group_history_params.groups.robot1_rotation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.robot1_gripper.indices=[19]"
                        "++policy.action_group_history_params.groups.robot1_gripper.use_history=${TRANSPORT_GRIPPER_HISTORY}"
                        "++policy.action_group_history_params.groups.robot1_gripper.update_rate=1.0"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.gripper_index=9"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.translation_indices=[0,1,2]"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.rotation_indices=[3,4,5,6,7,8]"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.gripper_weight=1.5"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.closed_hold_gripper_weight=${TRANSPORT_ROBOT0_HOLD_GRIPPER_WEIGHT}"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.closed_hold_translation_weight=${TRANSPORT_ROBOT0_HOLD_POSE_WEIGHT}"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot0.closed_hold_rotation_weight=${TRANSPORT_ROBOT0_HOLD_POSE_WEIGHT}"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.gripper_index=19"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.translation_indices=[10,11,12]"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.rotation_indices=[13,14,15,16,17,18]"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.gripper_weight=1.7"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.closed_hold_gripper_weight=${TRANSPORT_ROBOT1_HOLD_GRIPPER_WEIGHT}"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.closed_hold_translation_weight=${TRANSPORT_ROBOT1_HOLD_POSE_WEIGHT}"
                        "++policy.action_phase_loss_params.per_gripper_groups.robot1.closed_hold_rotation_weight=${TRANSPORT_ROBOT1_HOLD_POSE_WEIGHT}"
                        "++policy.action_phase_loss_params.coordinate_motion_on_any_transition=true"
                        "++policy.action_phase_loss_params.bimanual_hold_enabled=true"
                        "++policy.action_phase_loss_params.bimanual_hold_translation_weight=0.0"
                        "++policy.action_phase_loss_params.bimanual_hold_rotation_weight=0.0"
                        "++policy.action_phase_loss_params.bimanual_hold_gripper_weight=0.4"
                    )
                else
                    args+=(
                        "++policy.action_group_history_params.groups.translation.indices=${translation_indices}"
                        "++policy.action_group_history_params.groups.translation.use_history=true"
                        "++policy.action_group_history_params.groups.translation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.rotation.indices=${rotation_indices}"
                        "++policy.action_group_history_params.groups.rotation.use_history=true"
                        "++policy.action_group_history_params.groups.rotation.update_rate=1.0"
                        "++policy.action_group_history_params.groups.gripper.indices=${gripper_indices}"
                        "++policy.action_group_history_params.groups.gripper.use_history=false"
                        "++policy.action_group_history_params.groups.gripper.update_rate=1.0"
                    )
                fi
                ;;
            HOMO)
                args+=(
                    "++policy.action_group_loss_weights=null"
                    "policy.action_group_spectral_params=null"
                    "++policy.action_group_history_params.enabled=false"
                    "++policy.action_phase_loss_params.enabled=false"
                )
                ;;
            *)
                echo "[ERROR] VARIANTS supports only FULL or HOMO, got ${variant}" >&2
                exit 2
                ;;
        esac

        echo
        echo "================================================================="
        echo "[DATASET_GENERALIZATION] task=${task} source=${source} variant=${variant_upper}"
        echo "[DATASET_GENERALIZATION] seed=${SEED} max_demos=${MAX_TRAIN_EPISODES}"
        echo "[DATASET_GENERALIZATION] output=${output_dir}"
        echo "================================================================="

        python train.py --config-name="${config_name}" "${args[@]}" \
            2>&1 | tee "${output_dir}/train.log"
    done
done

echo
echo "[DATASET_GENERALIZATION] Complete: JOBS=${JOBS} VARIANTS=${VARIANTS}"
