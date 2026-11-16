#!/usr/bin/env bash
# One-H100 completion pipeline for the remaining ICRA A-D experiments.
# Order: missing Can Base -> Can 4x100 eval -> Square 100 eval -> PushT noise.

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
CAN_SEED="${CAN_SEED:-42}"
CAN_EPOCHS="${CAN_EPOCHS:-3000}"
CAN_OUTPUT_ROOT="${CAN_OUTPUT_ROOT:-data/outputs/robomimic_can_source_prior_ablation_3000}"
SEARCH_ROOT="${SEARCH_ROOT:-data/outputs}"
RESULT_ROOT="${RESULT_ROOT:-results/icra_abcd_completion}"
EVAL_ENVS="${EVAL_ENVS:-20}"
CAN_N_TEST="${CAN_N_TEST:-100}"
SQUARE_N_TEST="${SQUARE_N_TEST:-100}"
NOISE_N_TEST="${NOISE_N_TEST:-50}"
NOISE_STDS="${NOISE_STDS:-0.00 0.03 0.10 0.30 0.50}"
NOISE_SEEDS="${NOISE_SEEDS:-0 1 2}"
NOISE_ZERO_SEEDS="${NOISE_ZERO_SEEDS:-0}"
SKIP_DONE="${SKIP_DONE:-true}"
PUSHT_TRAIN_EPOCHS="${PUSHT_TRAIN_EPOCHS:-3000}"
PUSHT_TRAIN_SEED="${PUSHT_TRAIN_SEED:-42}"
PUSHT_OUTPUT_ROOT="${PUSHT_OUTPUT_ROOT:-data/outputs/pusht_icra_best_step6}"
PUSHT_RUN_NAME="${PUSHT_RUN_NAME:-freq_full_warm_step6_retrain_v1_seed42}"
PUSHT_ONLY="${PUSHT_ONLY:-false}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "${RESULT_ROOT}"

resolve_ckpt() {
    local explicit="$1"
    local pattern="$2"
    if [[ -n "${explicit}" ]]; then
        [[ -f "${explicit}" ]] || { echo "[ERROR] checkpoint does not exist: ${explicit}" >&2; return 1; }
        printf '%s\n' "${explicit}"
        return
    fi
    python scripts/find_best_checkpoint.py --root "${SEARCH_ROOT}" --include "${pattern}"
}

eval_ckpt() {
    local label="$1" ckpt="$2" n_test="$3"
    local out="${RESULT_ROOT}/${label}"
    if [[ "${SKIP_DONE}" == "true" && -f "${out}/eval_log.json" ]]; then
        echo "[SKIP DONE] ${label}"
        return
    fi
    rm -rf "${out}"
    echo "[EVAL] ${label} n_test=${n_test} checkpoint=${ckpt}"
    python eval.py -c "${ckpt}" -o "${out}" -d "${DEVICE}" --overwrite \
        --n-envs "${EVAL_ENVS}" --n-test "${n_test}" --n-train 0 \
        --n-test-vis 0 --n-train-vis 0 --profile-actions
}

CAN_BASE_CKPT=""
CAN_UNIFORM_CKPT=""
CAN_SPLIT_CKPT=""
CAN_FULL_CKPT=""
SQUARE_CKPT=""

if [[ "${PUSHT_ONLY}" != "true" ]]; then
echo "[1/4] Resolve or train missing Can Base A2A/frequency checkpoint"
CAN_BASE_PATTERN='can.*base(_a2a|.*freq)|can_ph_base_seed'
CAN_BASE_CKPT="$(resolve_ckpt "${CAN_BASE_CKPT:-}" "${CAN_BASE_PATTERN}" || true)"
if [[ -z "${CAN_BASE_CKPT}" ]]; then
    VARIANTS=base NUM_EPOCHS="${CAN_EPOCHS}" SEED="${CAN_SEED}" GPU_ID="${GPU_ID}" \
        OUTPUT_ROOT="${CAN_OUTPUT_ROOT}" WANDB_MODE="${WANDB_MODE}" \
        bash scripts/run_can_source_prior_ablation.sh
    CAN_BASE_CKPT="$(resolve_ckpt "" "${CAN_BASE_PATTERN}" || true)"
    if [[ -z "${CAN_BASE_CKPT}" ]]; then
        echo "[ERROR] Can Base finished but its checkpoint could not be resolved under ${SEARCH_ROOT}." >&2
        echo "Set CAN_BASE_CKPT=/absolute/path/to/latest.ckpt and rerun." >&2
        exit 1
    fi
fi

CAN_UNIFORM_CKPT="$(resolve_ckpt "${CAN_UNIFORM_CKPT:-}" 'can.*uniform')" || {
    echo "[ERROR] Can uniform checkpoint not found. Set CAN_UNIFORM_CKPT=/absolute/path.ckpt" >&2; exit 1; }
CAN_SPLIT_CKPT="$(resolve_ckpt "${CAN_SPLIT_CKPT:-}" 'can.*split.*history|can.*hybrid.*gripper|can.*_hg_')" || {
    echo "[ERROR] Can split checkpoint not found. Set CAN_SPLIT_CKPT=/absolute/path.ckpt" >&2; exit 1; }
CAN_FULL_CKPT="$(resolve_ckpt "${CAN_FULL_CKPT:-}" 'can.*(samp_full|full_split|hg_gc_fix)|can_ph_full_seed')" || {
    echo "[ERROR] Can full checkpoint not found. Set CAN_FULL_CKPT=/absolute/path.ckpt" >&2; exit 1; }

echo "[2/4] Can source-prior ablation: 100 episodes plus motion profiling"
eval_ckpt can_base_a2a "${CAN_BASE_CKPT}" "${CAN_N_TEST}"
eval_ckpt can_uniform_history "${CAN_UNIFORM_CKPT}" "${CAN_N_TEST}"
eval_ckpt can_split_history "${CAN_SPLIT_CKPT}" "${CAN_N_TEST}"
eval_ckpt can_full_samp "${CAN_FULL_CKPT}" "${CAN_N_TEST}"

echo "[3/4] Square PH stress case: 100 episodes"
SQUARE_CKPT="$(resolve_ckpt "${SQUARE_CKPT:-}" 'square.*(square_ph|hmi_full|gc_fix|hg_gc|best)')" || {
    echo "[ERROR] Square best checkpoint not found. Set SQUARE_CKPT=/absolute/path.ckpt" >&2; exit 1; }
eval_ckpt square_ph_repeat100 "${SQUARE_CKPT}" "${SQUARE_N_TEST}"
else
    echo "[SKIP] PUSHT_ONLY=true: skipping Can and Square phases"
fi

echo "[4/4] PushT best 6-step observation-noise sweep"
if [[ -n "${PUSHT_CKPT:-}" ]]; then
    [[ -f "${PUSHT_CKPT}" ]] || {
        echo "[ERROR] PushT checkpoint does not exist: ${PUSHT_CKPT}" >&2
        exit 1
    }
else
    PUSHT_RUN_DIR="${PUSHT_OUTPUT_ROOT}/${PUSHT_RUN_NAME}"
    PUSHT_COMPLETE_MARKER="${PUSHT_RUN_DIR}/.training_complete"
    PUSHT_CKPT="${PUSHT_RUN_DIR}/checkpoints/latest.ckpt"

    if [[ ! -f "${PUSHT_COMPLETE_MARKER}" || ! -f "${PUSHT_CKPT}" ]]; then
        echo "[PUSHT TRAIN] Rebuilding best full-frequency warm prior, step=6"
        rm -rf "${PUSHT_RUN_DIR}"
        mkdir -p "${PUSHT_RUN_DIR}"
        python train.py --config-name=pusht \
            "hydra.run.dir=${PUSHT_RUN_DIR}" \
            "training.device=${DEVICE}" \
            "training.seed=${PUSHT_TRAIN_SEED}" \
            "task.dataset.seed=${PUSHT_TRAIN_SEED}" \
            "training.num_epochs=${PUSHT_TRAIN_EPOCHS}" \
            training.resume=false \
            training.rollout_every=50 \
            training.checkpoint_every=50 \
            "logging.mode=${WANDB_MODE}" \
            "logging.name=${PUSHT_RUN_NAME}" \
            policy.num_inference_steps=6 \
            policy.freq_split_low=0 \
            policy.freq_split_high=16 \
            policy.sigma=0.3 \
            policy.sigma_high=0.3 \
            policy.cold_start_prob=0.3 \
            task.env_runner.n_test=50 \
            task.env_runner.n_train=6 \
            task.env_runner.max_steps=300 \
            2>&1 | tee "${RESULT_ROOT}/pusht_best_step6_train.log"

        [[ -f "${PUSHT_CKPT}" ]] || {
            echo "[ERROR] PushT training finished without latest.ckpt" >&2
            exit 1
        }
        touch "${PUSHT_COMPLETE_MARKER}"
    else
        echo "[PUSHT TRAIN] completed checkpoint found: ${PUSHT_CKPT}"
    fi
fi
for std in ${NOISE_STDS}; do
    token="${std/./p}"
    seeds="${NOISE_SEEDS}"
    if [[ "${std}" == "0" || "${std}" == "0.0" || "${std}" == "0.00" ]]; then
        seeds="${NOISE_ZERO_SEEDS}"
    fi
    for seed in ${seeds}; do
        label="pusht_noise_std${token}_seed${seed}"
        out="${RESULT_ROOT}/${label}"
        if [[ "${SKIP_DONE}" == "true" && -f "${out}/eval_log.json" ]]; then
            echo "[SKIP DONE] ${label}"
            continue
        fi
        rm -rf "${out}"
        echo "[NOISE] std=${std} seed=${seed} n_test=${NOISE_N_TEST}"
        python eval.py -c "${PUSHT_CKPT}" -o "${out}" -d "${DEVICE}" --overwrite \
            --n-envs "${EVAL_ENVS}" --n-test "${NOISE_N_TEST}" --n-train 0 \
            --n-test-vis 0 --n-train-vis 0 --profile-actions \
            --obs-noise-std "${std}" --obs-noise-seed "${seed}"
    done
done

python scripts/summarize_icra_abcd.py "${RESULT_ROOT}" > "${RESULT_ROOT}/summary.tsv"
cat > "${RESULT_ROOT}/checkpoint_manifest.tsv" <<EOF
item\tcheckpoint
can_base_a2a\t${CAN_BASE_CKPT}
can_uniform_history\t${CAN_UNIFORM_CKPT}
can_split_history\t${CAN_SPLIT_CKPT}
can_full_samp\t${CAN_FULL_CKPT}
square_ph_best\t${SQUARE_CKPT}
pusht_best_step6\t${PUSHT_CKPT}
EOF

echo "[DONE] ${RESULT_ROOT}/summary.tsv"
echo "[DONE] ${RESULT_ROOT}/checkpoint_manifest.tsv"
