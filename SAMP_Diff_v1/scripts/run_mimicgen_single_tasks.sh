#!/usr/bin/env bash
# Sequential SAMP training for selected MimicGen single-objective tasks.

set -euo pipefail

TASKS="${TASKS:-stack_d1 nut_assembly_d0 threading_d2}"
DATA_ROOT="${DATA_ROOT:-data/mimicgen/core}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/mimicgen_single_tasks}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-4000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-28}"
N_TEST="${N_TEST:-50}"
N_TEST_VIS="${N_TEST_VIS:-6}"
MAX_TRAIN_EPISODES="${MAX_TRAIN_EPISODES:-1000}"
WANDB_MODE="${WANDB_MODE:-disabled}"
GPU_ID="${GPU_ID:-0}"
MODE="${1:-train}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "${OUTPUT_ROOT}"

case_config() {
    local task="$1"
    case "${task}" in
        stack_d1)
            expected_env="Stack_D1"
            obs_dim="32"
            max_steps="400"
            rotation_split="12"
            translation_split="8"
            translation_sigma="0.24"
            translation_sigma_high="0.08"
            rotation_sigma="0.12"
            rotation_sigma_high="0.06"
            gripper_sigma="0.35"
            gripper_sigma_high="0.25"
            translation_history_rate="1.0"
            rotation_history_rate="1.0"
            translation_loss_weight="1.8"
            rotation_loss_weight="1.6"
            gripper_loss_weight="1.6"
            cold_start_prob="0.20"
            phase_weight="0.040"
            release_enabled="true"
            event_oversample_enabled="true"
            event_anchor="first_positive_reward"
            event_before="32"
            event_after="16"
            event_repeats="3"
            event_gate_enabled="false"
            event_gate_indices="auto"
            event_gate_min="[-0.035,-0.035,-0.080]"
            event_gate_max="[0.035,0.035,-0.015]"
            event_gate_radius="3"
            ;;
        nut_assembly_d0)
            expected_env="NutAssembly_D0"
            obs_dim="37"
            max_steps="700"
            translation_split="10"
            rotation_split="16"
            translation_sigma="0.18"
            translation_sigma_high="0.06"
            rotation_sigma="0.10"
            rotation_sigma_high="0.05"
            gripper_sigma="0.30"
            gripper_sigma_high="0.20"
            translation_history_rate="1.0"
            rotation_history_rate="1.0"
            translation_loss_weight="2.0"
            rotation_loss_weight="2.0"
            gripper_loss_weight="1.8"
            cold_start_prob="0.20"
            phase_weight="0.035"
            release_enabled="true"
            event_oversample_enabled="true"
            event_anchor="first_positive_reward"
            event_before="40"
            event_after="12"
            event_repeats="4"
            event_gate_enabled="false"
            event_gate_indices="null"
            event_gate_min="null"
            event_gate_max="null"
            event_gate_radius="0"
            ;;
        threading_d2)
            expected_env="Threading_D2"
            obs_dim="37"
            max_steps="500"
            translation_split="12"
            rotation_split="16"
            translation_sigma="0.15"
            translation_sigma_high="0.05"
            rotation_sigma="0.08"
            rotation_sigma_high="0.04"
            gripper_sigma="0.25"
            gripper_sigma_high="0.18"
            translation_history_rate="1.0"
            rotation_history_rate="1.0"
            translation_loss_weight="2.2"
            rotation_loss_weight="2.4"
            gripper_loss_weight="1.6"
            cold_start_prob="0.20"
            phase_weight="0.030"
            release_enabled="false"
            event_oversample_enabled="true"
            event_anchor="first_positive_reward"
            event_before="48"
            event_after="8"
            event_repeats="4"
            event_gate_enabled="false"
            event_gate_indices="null"
            event_gate_min="null"
            event_gate_max="null"
            event_gate_radius="0"
            ;;
        *)
            echo "[ERROR] unsupported MimicGen task: ${task}" >&2
            return 1
            ;;
    esac
}

validate_dataset() {
    local path="$1"
    local expected="$2"
    local expected_obs_dim="$3"
    python - "${path}" "${expected}" "${expected_obs_dim}" <<'PY'
import h5py
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_env = sys.argv[2]
expected_obs_dim = int(sys.argv[3])
if not path.is_file():
    raise SystemExit(f"[ERROR] missing dataset: {path}")
with h5py.File(path, "r") as f:
    demos = f["data"]
    demo = demos["demo_0"]
    raw = demos.attrs.get("env_args", f.attrs.get("env_args"))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    env_args = json.loads(raw)
    env_name = env_args["env_name"]
    if env_name != expected_env:
        raise SystemExit(
            f"[ERROR] {path}: env_name={env_name}, expected={expected_env}"
        )
    if demo["actions"].shape[-1] != 7:
        raise SystemExit(f"[ERROR] {path}: expected raw action_dim=7")
    controller = env_args.get("env_kwargs", {}).get("controller_configs", {})
    control_delta = controller.get("control_delta", True)
    if control_delta is False:
        raise SystemExit(
            f"[ERROR] {path}: controller is absolute, but this launcher is "
            "for official MimicGen delta-action datasets"
        )
    obs = demo["obs"]
    keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    missing = [key for key in keys if key not in obs]
    if missing:
        raise SystemExit(f"[ERROR] {path}: missing obs keys {missing}")
    obs_dim = sum(obs[key].shape[-1] for key in keys)
    if obs_dim != expected_obs_dim:
        raise SystemExit(
            f"[ERROR] {path}: obs_dim={obs_dim}, expected={expected_obs_dim}"
        )
    print(
        f"[OK] {path.name}: env={env_name} demos={len(demos)} "
        f"raw_action_dim=7 policy_action_dim=7 control_delta={control_delta} "
        f"obs_dim={obs_dim}"
    )
PY
}

# Fail before allocating the GPU if environments or datasets are unavailable.
python - <<'PY'
try:
    import mimicgen.envs.robosuite  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "[ERROR] MimicGen environment registration failed: "
        f"{type(exc).__name__}: {exc}"
    )
print("[OK] MimicGen environments import successfully")
PY

for task in ${TASKS}; do
    case_config "${task}"
    validate_dataset "${DATA_ROOT}/${task}.hdf5" "${expected_env}" "${obs_dim}"
done

# Compose the inherited Hydra config now, before a long sequential run.
python train.py --config-name=mimicgen_single --cfg job >/dev/null
echo "[OK] Hydra config mimicgen_single composes successfully"

if [[ "${MODE}" == "--check" || "${MODE}" == "check" ]]; then
    echo "[CHECK] all MimicGen single-task datasets and environments are valid"
    exit 0
fi
if [[ "${MODE}" != "train" ]]; then
    echo "Usage: $0 [train|--check]" >&2
    exit 2
fi

for task in ${TASKS}; do
    case_config "${task}"
    dataset="${DATA_ROOT}/${task}.hdf5"
    if [[ "${event_gate_enabled}" == "true" && "${event_gate_indices}" == "auto" ]]; then
        event_gate_indices="$(
            python scripts/detect_stack_release_gate.py \
                "${dataset}" --min-coverage 0.75
        )"
    fi
    run_name="${task}_icra_parameter_prior_v1_seed${SEED}"
    output_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${output_dir}"

    echo
    echo "================================================================="
    echo "[MIMICGEN] task=${task} env=${expected_env} obs_dim=${obs_dim}"
    echo "[MIMICGEN] output=${output_dir}"
    echo "================================================================="

    python train.py --config-name=mimicgen_single \
        "hydra.run.dir=${output_dir}" \
        "obs_dim=${obs_dim}" \
        "task_semantics.task_name=${task}" \
        "task.dataset_path=${dataset}" \
        "training.seed=${SEED}" \
        "data_split.seed=${SEED}" \
        "data_split.max_train_episodes=${MAX_TRAIN_EPISODES}" \
        "training.resume=false" \
        "training.num_epochs=$((NUM_EPOCHS + 1))" \
        "training.rollout_every=${ROLLOUT_EVERY}" \
        "training.checkpoint_every=${CHECKPOINT_EVERY}" \
        "task.env_runner.max_steps=${max_steps}" \
        "task.env_runner.n_envs=${N_ENVS}" \
        "task.env_runner.n_test=${N_TEST}" \
        "task.env_runner.n_test_vis=${N_TEST_VIS}" \
        "logging.mode=${WANDB_MODE}" \
        "logging.name=${run_name}" \
        "n_action_steps=2" \
        "policy.num_inference_steps=8" \
        "policy.cold_start_prob=${cold_start_prob}" \
        "policy.action_group_spectral_params.translation.freq_split_high=${translation_split}" \
        "policy.action_group_spectral_params.translation.sigma=${translation_sigma}" \
        "policy.action_group_spectral_params.translation.sigma_high=${translation_sigma_high}" \
        "policy.action_group_spectral_params.rotation.freq_split_high=${rotation_split}" \
        "policy.action_group_spectral_params.rotation.sigma=${rotation_sigma}" \
        "policy.action_group_spectral_params.rotation.sigma_high=${rotation_sigma_high}" \
        "policy.action_group_spectral_params.gripper.freq_split_high=16" \
        "policy.action_group_spectral_params.gripper.sigma=${gripper_sigma}" \
        "policy.action_group_spectral_params.gripper.sigma_high=${gripper_sigma_high}" \
        "policy.action_group_loss_weights.translation.weight=${translation_loss_weight}" \
        "policy.action_group_loss_weights.rotation.weight=${rotation_loss_weight}" \
        "policy.action_group_loss_weights.gripper.weight=${gripper_loss_weight}" \
        "policy.action_group_history_params.enabled=true" \
        "policy.action_group_history_params.shift=2" \
        "policy.action_group_history_params.groups.translation.use_history=true" \
        "policy.action_group_history_params.groups.translation.update_rate=${translation_history_rate}" \
        "policy.action_group_history_params.groups.rotation.use_history=true" \
        "policy.action_group_history_params.groups.rotation.update_rate=${rotation_history_rate}" \
        "policy.action_group_history_params.groups.gripper.use_history=false" \
        "++policy.history_training_mode=aligned" \
        "policy.action_phase_loss_params.enabled=true" \
        "policy.action_phase_loss_params.weight=${phase_weight}" \
        "policy.action_phase_loss_params.release_enabled=${release_enabled}" \
        "++task.dataset.event_oversample_enabled=${event_oversample_enabled}" \
        "++task.dataset.event_oversample_anchor=${event_anchor}" \
        "++task.dataset.event_oversample_before=${event_before}" \
        "++task.dataset.event_oversample_after=${event_after}" \
        "++task.dataset.event_oversample_repeats=${event_repeats}" \
        "++task.dataset.event_oversample_min_anchor_ratio=0.98" \
        "++task.dataset.event_oversample_observation_gate_enabled=${event_gate_enabled}" \
        "++task.dataset.event_oversample_observation_gate_indices=${event_gate_indices}" \
        "++task.dataset.event_oversample_observation_gate_min=${event_gate_min}" \
        "++task.dataset.event_oversample_observation_gate_max=${event_gate_max}" \
        "++task.dataset.event_oversample_observation_gate_radius=${event_gate_radius}" \
        "++task.env_runner.action_clip_by_dataset=true" \
        "++task.env_runner.action_clip_margin_scale=0.10" \
        "++task.env_runner.action_clip_min_margin=0.02" \
        2>&1 | tee "${output_dir}/train.log"
done

echo "[MIMICGEN] complete: ${TASKS}"
