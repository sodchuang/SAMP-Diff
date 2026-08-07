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
MODE="${1:-train}"

export WANDB_MODE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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
            translation_sigma="0.24"
            rotation_sigma="0.12"
            phase_weight="0.020"
            release_enabled="true"
            ;;
        nut_assembly_d0)
            expected_env="NutAssembly_D0"
            obs_dim="37"
            max_steps="700"
            rotation_split="16"
            translation_sigma="0.18"
            rotation_sigma="0.10"
            phase_weight="0.025"
            release_enabled="true"
            ;;
        threading_d2)
            expected_env="Threading_D2"
            obs_dim="37"
            max_steps="500"
            rotation_split="16"
            translation_sigma="0.15"
            rotation_sigma="0.08"
            phase_weight="0.025"
            release_enabled="false"
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
    run_name="${task}_semantic_delta_controlled_v3_seed${SEED}"
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
        "policy.num_inference_steps=6" \
        "policy.action_group_spectral_params.translation.sigma=${translation_sigma}" \
        "policy.action_group_spectral_params.translation.sigma_high=${translation_sigma}" \
        "policy.action_group_spectral_params.rotation.freq_split_high=${rotation_split}" \
        "policy.action_group_spectral_params.rotation.sigma=${rotation_sigma}" \
        "policy.action_group_spectral_params.rotation.sigma_high=${rotation_sigma}" \
        "policy.action_group_history_params.enabled=true" \
        "policy.action_group_history_params.shift=2" \
        "policy.action_group_history_params.groups.translation.use_history=true" \
        "policy.action_group_history_params.groups.rotation.use_history=true" \
        "policy.action_group_history_params.groups.gripper.use_history=false" \
        "++policy.history_training_mode=aligned" \
        "policy.action_phase_loss_params.enabled=true" \
        "policy.action_phase_loss_params.weight=${phase_weight}" \
        "policy.action_phase_loss_params.release_enabled=${release_enabled}" \
        "++task.env_runner.action_clip_by_dataset=true" \
        "++task.env_runner.action_clip_margin_scale=0.10" \
        "++task.env_runner.action_clip_min_margin=0.02" \
        2>&1 | tee "${output_dir}/train.log"
done

echo "[MIMICGEN] complete: ${TASKS}"
