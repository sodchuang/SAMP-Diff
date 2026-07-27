#!/usr/bin/env bash
set -euo pipefail

# ToolHang automatic staged curriculum.
#
# Goal:
#   Keep training the current stage until its rollout metric passes a threshold.
#   Stage changes are driven by simulator state, not by epoch or reward alone.
#
# This is ToolHang-specific on purpose. It is meant to get ToolHang working first,
# before generalizing the mechanism to every low-dim task.
#
# Stages:
#   A_HOLD   : contact-then-grasp; align/descend before close, then hold.
#   B_INSERT : resume from A, add insertion guidance.
#   C_FULL   : resume from B, add release/final-settle guidance.
#
# Typical use:
#   bash scripts/run_tool_hang_auto_curriculum.sh
#
# Useful overrides:
#   BASE_RUN_NAME=tool_hang_auto_clean_v3 bash scripts/run_tool_hang_auto_curriculum.sh
#   A_PASS_SCORE=0.80 B_PASS_SCORE=0.50 C_PASS_SCORE=0.50 bash scripts/run_tool_hang_auto_curriculum.sh
#   MAX_STAGE_EPOCHS=3000 CHUNK_EPOCHS=200 bash scripts/run_tool_hang_auto_curriculum.sh

PIPELINE_VERSION="toolhang_direction_guard_v5_rolling"
BASE_RUN_NAME="${BASE_RUN_NAME:-tool_hang_auto_direction_guard_v5_rolling}"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-200}"
FIRST_EVAL_EPOCH="${FIRST_EVAL_EPOCH:-300}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
N_ENVS="${N_ENVS:-8}"
MAX_STAGE_EPOCHS="${MAX_STAGE_EPOCHS:-7000}"
A_MAX_EPOCHS="${A_MAX_EPOCHS:-3000}"

A_PASS_SCORE="${A_PASS_SCORE:-0.80}"
B_PASS_SCORE="${B_PASS_SCORE:-0.50}"
C_PASS_SCORE="${C_PASS_SCORE:-0.50}"
B_PASS_ADDED_EPOCHS="${B_PASS_ADDED_EPOCHS:-2000}"
B_FREEZE_ENCODER_ADDED_EPOCHS="${B_FREEZE_ENCODER_ADDED_EPOCHS:-300}"
REQUIRED_CONSECUTIVE_PASSES="${REQUIRED_CONSECUTIVE_PASSES:-2}"
ROLLING_BEST_CHECKPOINT="${ROLLING_BEST_CHECKPOINT:-true}"
A_MIN_EPOCH="${A_MIN_EPOCH:-300}"
A_DIRECTION_CHECK_EPOCH="${A_DIRECTION_CHECK_EPOCH:-1200}"
A_MIN_CONTACT_RATE="${A_MIN_CONTACT_RATE:-0.02}"
A_LIFT_CHECK_EPOCH="${A_LIFT_CHECK_EPOCH:-1800}"
A_MIN_LIFT_RATE="${A_MIN_LIFT_RATE:-0.02}"
A_GRASP_CHECK_EPOCH="${A_GRASP_CHECK_EPOCH:-2400}"
A_MIN_GRASP_RATE="${A_MIN_GRASP_RATE:-0.02}"
C_MIN_ADDED_EPOCHS="${C_MIN_ADDED_EPOCHS:-500}"
ALLOW_STAGE_B="${ALLOW_STAGE_B:-true}"
ALLOW_STAGE_C="${ALLOW_STAGE_C:-true}"
A_CHECKPOINT_PATH="${A_CHECKPOINT_PATH:-}"


export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

require_project_root() {
  if [[ ! -f train.py || ! -f scripts/run_tool_hang_rescue.sh ]]; then
    echo "[ERROR] Run this from the SAMP-Diff project root." >&2
    exit 2
  fi

  local required_markers=(
    "diffusion_policy/policy/samp_lowdim_policy.py:history_training_mode"
    "diffusion_policy/env_runner/robomimic_lowdim_runner.py:stage_{stage_name}_rate"
    "diffusion_policy/env_runner/robomimic_lowdim_runner.py:recover_after_worker_error"
    "diffusion_policy/workspace/train_samp_lowdim_workspace.py:_is_mujoco_instability_error"
    "diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py:get_tool_hang_stage_flags"
    "diffusion_policy/gym_util/multistep_wrapper.py:get_tool_hang_stage_flags"
    "diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py:normalizer_use_full_dataset"
    "diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py:episode prefix anchor coverage is too low"
    "config_task/low_dim/tool_hang_ph.yaml:normalizer_use_full_dataset: true"
    "config_task/low_dim/tool_hang_ph.yaml:episode_prefix_min_anchor_ratio: 0.9"
    "scripts/run_tool_hang_rescue.sh:policy.history_training_mode"
    "scripts/run_tool_hang_auto_curriculum.sh:A_DIRECTION_CHECK_EPOCH"
  )
  local item
  local file
  local marker
  for item in "${required_markers[@]}"; do
    file="${item%%:*}"
    marker="${item#*:}"
    if [[ ! -f "${file}" ]] || ! grep -Fq "${marker}" "${file}"; then
      echo "[ERROR] ToolHang code bundle is incomplete or stale." >&2
      echo "[ERROR] Missing marker '${marker}' in ${file}" >&2
      echo "[ERROR] Sync all ToolHang Python files and scripts before training." >&2
      exit 6
    fi
  done
  echo "[AUTO] verified pipeline bundle: ${PIPELINE_VERSION}"
}

stage_run_name() {
  local stage="$1"
  case "${stage}" in
    A_HOLD) echo "${BASE_RUN_NAME}_A_hold" ;;
    B_INSERT) echo "${BASE_RUN_NAME}_B_insert" ;;
    C_FULL) echo "${BASE_RUN_NAME}_C_full" ;;
    *) echo "[ERROR] unknown stage ${stage}" >&2; exit 2 ;;
  esac
}

stage_pass_score() {
  local stage="$1"
  case "${stage}" in
    A_HOLD) echo "${A_PASS_SCORE}" ;;
    B_INSERT) echo "${B_PASS_SCORE}" ;;
    C_FULL) echo "${C_PASS_SCORE}" ;;
    *) echo "[ERROR] unknown stage ${stage}" >&2; exit 2 ;;
  esac
}

stage_gate_metric() {
  local stage="$1"
  case "${stage}" in
    A_HOLD) echo "test/stage_grasp_rate" ;;
    B_INSERT) echo "test/stage_insert_rate" ;;
    C_FULL) echo "test/stage_full_rate" ;;
    *) echo "[ERROR] unknown stage ${stage}" >&2; exit 2 ;;
  esac
}

latest_ckpt() {
  local run_name="$1"
  local path="data/outputs/robomimic/${run_name}/checkpoints/latest.ckpt"
  if [[ -f "${path}" ]]; then
    echo "${path}"
  fi
}

current_epoch() {
  local run_name="$1"
  local log_path="data/outputs/robomimic/${run_name}/logs.json.txt"
  if [[ ! -f "${log_path}" ]]; then
    echo 0
    return
  fi
  python -c "import json, pathlib
p=pathlib.Path('${log_path}')
last=None
for line in p.read_text(errors='ignore').splitlines():
    try:
        obj=json.loads(line)
    except Exception:
        continue
    if 'epoch' in obj:
        last=int(obj['epoch'])
print(0 if last is None else last + 1)"
}

checkpoint_epoch() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  python -c "import dill, torch
payload=torch.load('${path}', map_location='cpu', pickle_module=dill)
raw=payload.get('pickles', {}).get('epoch')
print(0 if raw is None else int(dill.loads(raw)))"
}

latest_score() {
  local run_name="$1"
  local metric="$2"
  local log_path="data/outputs/robomimic/${run_name}/logs.json.txt"
  if [[ ! -f "${log_path}" ]]; then
    echo "nan"
    return
  fi
  python -c "import json, math, pathlib
p=pathlib.Path('${log_path}')
metric='${metric}'
score=None
for line in p.read_text(errors='ignore').splitlines():
    try:
        obj=json.loads(line)
    except Exception:
        continue
    if metric in obj:
        score=float(obj[metric])
print('nan' if score is None or math.isnan(score) else score)"
}

latest_metric_epoch() {
  local run_name="$1"
  local metric="$2"
  local log_path="data/outputs/robomimic/${run_name}/logs.json.txt"
  if [[ ! -f "${log_path}" ]]; then
    echo -1
    return
  fi
  python -c "import json, pathlib
p=pathlib.Path('${log_path}')
metric='${metric}'
metric_epoch=-1
for line in p.read_text(errors='ignore').splitlines():
    try:
        obj=json.loads(line)
    except Exception:
        continue
    if metric in obj and 'epoch' in obj:
        metric_epoch=int(obj['epoch'])
print(metric_epoch)"
}

consecutive_passes() {
  local run_name="$1"
  local metric="$2"
  local threshold="$3"
  local log_path="data/outputs/robomimic/${run_name}/logs.json.txt"
  if [[ ! -f "${log_path}" ]]; then
    echo 0
    return
  fi
  python -c "import json, pathlib
p=pathlib.Path('${log_path}')
metric='${metric}'
threshold=float('${threshold}')
values_by_epoch={}
for line in p.read_text(errors='ignore').splitlines():
    try:
        obj=json.loads(line)
    except Exception:
        continue
    if metric in obj and 'epoch' in obj:
        # A resume-time rollout can log the same epoch more than once. Count
        # distinct rollout epochs so one duplicated result cannot satisfy the
        # consecutive-pass gate by itself.
        values_by_epoch[int(obj['epoch'])]=float(obj[metric])
count=0
for _, value in sorted(values_by_epoch.items(), reverse=True):
    if value < threshold:
        break
    count += 1
print(count)"
}

record_best_checkpoint() {
  local stage="$1"
  local run_name="$2"
  local metric="$3"
  local score="$4"
  local metric_epoch="$5"
  local source_ckpt
  local checkpoint_dir
  local best_ckpt
  local state_path
  local improved

  BEST_CKPT=""
  if [[ "${ROLLING_BEST_CHECKPOINT}" != "true" || "${score}" == "nan" ]]; then
    return
  fi

  source_ckpt="$(latest_ckpt "${run_name}")"
  if [[ -z "${source_ckpt}" ]]; then
    return
  fi

  checkpoint_dir="data/outputs/robomimic/${run_name}/checkpoints"
  best_ckpt="${checkpoint_dir}/auto_best_${stage}.ckpt"
  state_path="${checkpoint_dir}/auto_best_${stage}.txt"
  improved="$(python -c "import math, pathlib
score=float('${score}')
p=pathlib.Path('${state_path}')
best=-math.inf
if p.is_file():
    try:
        best=float(p.read_text().split()[0])
    except Exception:
        pass
print('yes' if score > best else 'no')")"

  if [[ "${improved}" == "yes" ]]; then
    cp -f "${source_ckpt}" "${best_ckpt}"
    printf '%s %s %s\n' "${score}" "${metric_epoch}" "${metric}" > "${state_path}"
    echo "[AUTO] ${stage} rolling best updated: ${metric}=${score}, metric_epoch=${metric_epoch}, checkpoint=${best_ckpt}"
  fi
  if [[ -f "${best_ckpt}" ]]; then
    BEST_CKPT="${best_ckpt}"
  fi
}

score_passed() {
  local score="$1"
  local threshold="$2"
  python -c "import math
score=float('${score}') if '${score}' != 'nan' else float('nan')
threshold=float('${threshold}')
print('yes' if (not math.isnan(score) and score >= threshold) else 'no')"
}

next_target_epoch() {
  local run_name="$1"
  local cur
  cur="$(current_epoch "${run_name}")"
  python -c "cur=int('${cur}')
chunk=int('${CHUNK_EPOCHS}')
first=int('${FIRST_EVAL_EPOCH}') + 1
every=int('${ROLLOUT_EVERY}')
desired=max(cur + chunk, first)
# num_epochs is exclusive. Round the final trained epoch up to a rollout
# boundary so every chunk finishes with a fresh simulator metric.
last_epoch=desired - 1
rollout_epoch=((last_epoch + every - 1) // every) * every
print(max(desired, rollout_epoch + 1))"
}

run_stage_chunk() {
  local stage="$1"
  local run_name="$2"
  local target_epoch="$3"
  local resume_mode="$4"
  local resume_path="${5:-}"

  echo
  echo "[AUTO] stage=${stage} run=${run_name} target_epoch=${target_epoch} resume_mode=${resume_mode} resume_path=${resume_path:-none}"

  case "${stage}" in
    A_HOLD)
      # Exact A recipe from the original direction_guard_v4 run that reached
      # stage_grasp_rate >= 0.80. B/C changes must stay in their own cases.
      echo "[AUTO] A recipe=direction_guard_v4_exact phase_loss=false timing=false weights=1.2/1.2/1.6 prefix=64/96/160 variant=HG"
      RUN_NAME="${run_name}" \
      NUM_EPOCHS="${target_epoch}" \
      ROLLOUT_START_EPOCH="${FIRST_EVAL_EPOCH}" \
      ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
      N_ENVS="${N_ENVS}" \
      TRAINING_RESUME="${resume_mode}" \
      RESUME_FROM_PATH="${resume_path:-null}" \
      GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-auto}" \
      GRIPPER_CLOSE_IS_GREATER="${GRIPPER_CLOSE_IS_GREATER:-auto}" \
      N_TRAIN_ROLLOUTS="${A_N_TRAIN_ROLLOUTS:-10}" \
      N_TRAIN_VIDEOS="${A_N_TRAIN_VIDEOS:-10}" \
      N_TEST_ROLLOUTS="${A_N_TEST_ROLLOUTS:-20}" \
      N_TEST_VIDEOS="${A_N_TEST_VIDEOS:-20}" \
      N_ACTION_STEPS="${A_N_ACTION_STEPS:-4}" \
      HISTORY_SHIFT="${A_HISTORY_SHIFT:-4}" \
      ACTION_CLIP_BY_DATASET=true \
      ACTION_CLIP_MARGIN_SCALE="${A_ACTION_CLIP_MARGIN_SCALE:-0.10}" \
      ACTION_CLIP_MIN_MARGIN="${A_ACTION_CLIP_MIN_MARGIN:-0.02}" \
      PHASE_LOSS_ENABLED=false \
      PHASE_LOSS_WEIGHT="${A_PHASE_LOSS_WEIGHT:-0.004}" \
      TRANSLATION_WEIGHT="${A_TRANSLATION_WEIGHT:-1.2}" \
      ROTATION_WEIGHT="${A_ROTATION_WEIGHT:-1.2}" \
      GRIPPER_WEIGHT="${A_GRIPPER_WEIGHT:-1.6}" \
      ROTATION_SIGMA="${A_ROTATION_SIGMA:-0.12}" \
      GRIPPER_TIMING_ENABLED=false \
      EPISODE_PREFIX_ENABLED=true \
      EPISODE_PREFIX_ANCHOR=first_close \
      EPISODE_PREFIX_AFTER="${A_EPISODE_PREFIX_AFTER:-64}" \
      EPISODE_PREFIX_MIN_STEPS="${A_EPISODE_PREFIX_MIN_STEPS:-96}" \
      EPISODE_PREFIX_MAX_STEPS="${A_EPISODE_PREFIX_MAX_STEPS:-160}" \
      REGRASP_WINDOW_ENABLED=false \
      FINAL_RELEASE_WINDOW_ENABLED=false \
      RELEASE_ENABLED=false \
      bash scripts/run_tool_hang_rescue.sh HG
      ;;

    B_INSERT)
      # B v5 learns stable frame assembly while keeping the grasp. Release is
      # deliberately deferred to C; otherwise the insert stage learns to drop
      # the frame near the stand before it is seated.
      echo "[AUTO] B recipe=insert_only_v5 lr=1e-5 action_steps=4 release=false metric=stage_insert_rate"
      RUN_NAME="${run_name}" \
      NUM_EPOCHS="${target_epoch}" \
      ROLLOUT_START_EPOCH="${FIRST_EVAL_EPOCH}" \
      ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
      N_ENVS="${N_ENVS}" \
      TRAINING_RESUME="${resume_mode}" \
      RESUME_FROM_PATH="${resume_path:-null}" \
      GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-auto}" \
      GRIPPER_CLOSE_IS_GREATER="${GRIPPER_CLOSE_IS_GREATER:-auto}" \
      ROLLOUT_BEFORE_TRAINING=true \
      FREEZE_ENCODER_UNTIL_EPOCH="${B_FREEZE_ENCODER_UNTIL_EPOCH:-null}" \
      TRAIN_LR="${B_TRAIN_LR:-0.00001}" \
      N_ACTION_STEPS="${B_N_ACTION_STEPS:-4}" \
      HISTORY_SHIFT="${B_HISTORY_SHIFT:-4}" \
      PHASE_LOSS_WEIGHT="${B_PHASE_LOSS_WEIGHT:-0.012}" \
      TRANSLATION_WEIGHT="${B_TRANSLATION_WEIGHT:-1.6}" \
      ROTATION_WEIGHT="${B_ROTATION_WEIGHT:-0.8}" \
      GRIPPER_WEIGHT="${B_GRIPPER_WEIGHT:-3.6}" \
      GRIPPER_TIMING_ENABLED=true \
      MIN_CLOSE_STEPS="${B_MIN_CLOSE_STEPS:-12}" \
      MAX_LATCH_CHUNKS_AFTER_CLOSE="${B_MAX_LATCH_CHUNKS_AFTER_CLOSE:-20}" \
      PICK_WINDOW_BEFORE="${B_PICK_WINDOW_BEFORE:-8}" \
      PICK_WINDOW_AFTER="${B_PICK_WINDOW_AFTER:-24}" \
      PICK_WINDOW_TRANSLATION_WEIGHT="${B_PICK_WINDOW_TRANSLATION_WEIGHT:-3.0}" \
      PICK_WINDOW_ROTATION_WEIGHT="${B_PICK_WINDOW_ROTATION_WEIGHT:-0.35}" \
      PICK_WINDOW_GRIPPER_WEIGHT="${B_PICK_WINDOW_GRIPPER_WEIGHT:-4.8}" \
      EPISODE_PREFIX_ENABLED=true \
      EPISODE_PREFIX_ANCHOR=first_release \
      EPISODE_PREFIX_AFTER="${B_EPISODE_PREFIX_AFTER:-24}" \
      EPISODE_PREFIX_MIN_STEPS="${B_EPISODE_PREFIX_MIN_STEPS:-180}" \
      EPISODE_PREFIX_MAX_STEPS="${B_EPISODE_PREFIX_MAX_STEPS:-320}" \
      HANG_WINDOW_ANCHOR=release \
      HANG_WINDOW_OCCURRENCE=first \
      HANG_WINDOW_BEFORE="${B_HANG_WINDOW_BEFORE:-40}" \
      HANG_WINDOW_AFTER="${B_HANG_WINDOW_AFTER:-0}" \
      HANG_WINDOW_TRANSLATION_WEIGHT="${B_HANG_WINDOW_TRANSLATION_WEIGHT:-3.2}" \
      HANG_WINDOW_ROTATION_WEIGHT="${B_HANG_WINDOW_ROTATION_WEIGHT:-3.4}" \
      HANG_WINDOW_GRIPPER_WEIGHT="${B_HANG_WINDOW_GRIPPER_WEIGHT:-0.0}" \
      REGRASP_WINDOW_ENABLED=false \
      FINAL_RELEASE_WINDOW_ENABLED=false \
      RELEASE_ENABLED=false \
      bash scripts/run_tool_hang_rescue.sh FLOW
      ;;

    C_FULL)
      RUN_NAME="${run_name}" \
      NUM_EPOCHS="${target_epoch}" \
      ROLLOUT_START_EPOCH="${FIRST_EVAL_EPOCH}" \
      ROLLOUT_EVERY="${ROLLOUT_EVERY}" \
      CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
      N_ENVS="${N_ENVS}" \
      TRAINING_RESUME="${resume_mode}" \
      RESUME_FROM_PATH="${resume_path:-null}" \
      PHASE_LOSS_WEIGHT="${C_PHASE_LOSS_WEIGHT:-0.0025}" \
      TRANSLATION_WEIGHT="${C_TRANSLATION_WEIGHT:-1.6}" \
      ROTATION_WEIGHT="${C_ROTATION_WEIGHT:-0.95}" \
      GRIPPER_WEIGHT="${C_GRIPPER_WEIGHT:-2.3}" \
      HANG_WINDOW_ANCHOR=close \
      HANG_WINDOW_OCCURRENCE=first \
      HANG_WINDOW_BEFORE=0 \
      HANG_WINDOW_AFTER=8 \
      HANG_WINDOW_TRANSLATION_WEIGHT="${C_HANG_WINDOW_TRANSLATION_WEIGHT:-0.12}" \
      HANG_WINDOW_ROTATION_WEIGHT="${C_HANG_WINDOW_ROTATION_WEIGHT:-0.35}" \
      HANG_WINDOW_GRIPPER_WEIGHT="${C_HANG_WINDOW_GRIPPER_WEIGHT:-0.0}" \
      REGRASP_WINDOW_ENABLED=true \
      FINAL_RELEASE_WINDOW_ENABLED=true \
      RELEASE_ENABLED=false \
      bash scripts/run_tool_hang_rescue.sh FLOW \
        "++policy.action_phase_loss_params.extra_windows.regrasp_insert.translation_weight=${C_REGRASP_TRANSLATION_WEIGHT:-0.08}" \
        "++policy.action_phase_loss_params.extra_windows.regrasp_insert.rotation_weight=${C_REGRASP_ROTATION_WEIGHT:-0.18}" \
        "++policy.action_phase_loss_params.extra_windows.regrasp_insert.gripper_weight=${C_REGRASP_GRIPPER_WEIGHT:-0.05}" \
        "++policy.action_phase_loss_params.extra_windows.final_release.translation_weight=${C_FINAL_RELEASE_TRANSLATION_WEIGHT:-0.06}" \
        "++policy.action_phase_loss_params.extra_windows.final_release.rotation_weight=${C_FINAL_RELEASE_ROTATION_WEIGHT:-0.12}" \
        "++policy.action_phase_loss_params.extra_windows.final_release.gripper_weight=${C_FINAL_RELEASE_GRIPPER_WEIGHT:-0.08}"
      ;;

    *)
      echo "[ERROR] unknown stage ${stage}" >&2
      exit 2
      ;;
  esac
}

run_until_pass() {
  local stage="$1"
  local init_resume_path="${2:-}"
  local run_name
  local threshold
  local resume_mode
  local resume_path
  local cur
  local target
  local score
  local passed
  local ckpt
  local metric
  local metric_epoch=-1
  local expected_metric_epoch=-1
  local minimum_epoch=0
  local maximum_epoch=0
  local pass_count=0
  local BEST_CKPT=""

  run_name="$(stage_run_name "${stage}")"
  threshold="$(stage_pass_score "${stage}")"
  metric="$(stage_gate_metric "${stage}")"

  case "${stage}" in
    A_HOLD)
      minimum_epoch="${A_MIN_EPOCH}"
      # num_epochs is exclusive; +1 ensures the final budget epoch itself
      # performs rollout (for example epoch 3000 with A_MAX_EPOCHS=3000).
      maximum_epoch=$((A_MAX_EPOCHS + 1))
      ;;
    B_INSERT)
      source_epoch="$(checkpoint_epoch "${init_resume_path}")"
      minimum_epoch=$((source_epoch + B_PASS_ADDED_EPOCHS))
      maximum_epoch=$((source_epoch + MAX_STAGE_EPOCHS + 1))
      ;;
    C_FULL)
      source_epoch="$(checkpoint_epoch "${init_resume_path}")"
      minimum_epoch=$((source_epoch + C_MIN_ADDED_EPOCHS))
      maximum_epoch=$((source_epoch + MAX_STAGE_EPOCHS + 1))
      ;;
  esac

  echo
  echo "[AUTO] ===== ${stage} (${run_name}) threshold=${threshold} metric=${metric} min_epoch=${minimum_epoch} max_epoch=${maximum_epoch} consecutive=${REQUIRED_CONSECUTIVE_PASSES} ====="
  while true; do
    cur="$(current_epoch "${run_name}")"
    score="$(latest_score "${run_name}" "${metric}")"
    passed="$(score_passed "${score}" "${threshold}")"
    pass_count="$(consecutive_passes "${run_name}" "${metric}" "${threshold}")"
    ckpt="$(latest_ckpt "${run_name}")"
    metric_epoch="$(latest_metric_epoch "${run_name}" "${metric}")"
    record_best_checkpoint "${stage}" "${run_name}" "${metric}" "${score}" "${metric_epoch}"
    if [[ "${passed}" == "yes" && "${cur}" -ge "${minimum_epoch}" && "${pass_count}" -ge "${REQUIRED_CONSECUTIVE_PASSES}" && -n "${ckpt}" ]]; then
      echo "[AUTO] ${stage} gate was already passed: ${metric}=${score}, epoch=${cur}, consecutive=${pass_count}."
      PASSED_CKPT="${BEST_CKPT:-${ckpt}}"
      echo "[AUTO] ${stage} transfer checkpoint: ${PASSED_CKPT}"
      return 0
    fi

    if [[ "${cur}" -ge "${maximum_epoch}" ]]; then
      score="$(latest_score "${run_name}" "${metric}")"
      if [[ "${stage}" == "A_HOLD" ]]; then
        contact_score="$(latest_score "${run_name}" "test/frame_grasp_contact_rate")"
        lift_score="$(latest_score "${run_name}" "test/frame_lift_rate")"
        hold_steps="$(latest_score "${run_name}" "test/mean_max_grasp_hold_steps")"
        echo "[AUTO] ${stage} exhausted its ${A_MAX_EPOCHS}-epoch budget at epoch=${maximum_epoch}."
        echo "[AUTO] grasp=${score} contact=${contact_score} lift=${lift_score} mean_hold_steps=${hold_steps}"
        echo "[AUTO] Stop: this is a pipeline/model failure, not a reason to add more epochs."
      else
        echo "[AUTO] ${stage} exhausted its ${MAX_STAGE_EPOCHS}-epoch budget at epoch=${maximum_epoch}, score=${score}. Stop for inspection."
      fi
      exit 3
    fi

    ckpt="$(latest_ckpt "${run_name}")"
    if [[ -n "${ckpt}" ]]; then
      resume_mode=true
      resume_path=null
    elif [[ -n "${init_resume_path}" ]]; then
      resume_mode=false
      resume_path="${init_resume_path}"
    else
      resume_mode=false
      resume_path=null
    fi

    target="$(next_target_epoch "${run_name}")"
    if [[ -z "${ckpt}" && -n "${init_resume_path}" ]]; then
      source_epoch="$(checkpoint_epoch "${init_resume_path}")"
      # The checkpoint epoch is inclusive while num_epochs is exclusive.
      # +1 makes the transfer chunk end on source_epoch + CHUNK_EPOCHS.
      source_target=$((source_epoch + CHUNK_EPOCHS + 1))
      if [[ "${target}" -lt "${source_target}" ]]; then
        echo "[AUTO] transfer checkpoint epoch=${source_epoch}; raising target_epoch from ${target} to ${source_target}"
        target="${source_target}"
      fi
    fi

    if [[ "${target}" -gt "${maximum_epoch}" ]]; then
      target="${maximum_epoch}"
    fi

    run_stage_chunk "${stage}" "${run_name}" "${target}" "${resume_mode}" "${resume_path}"

    score="$(latest_score "${run_name}" "${metric}")"
    cur="$(current_epoch "${run_name}")"
    metric_epoch="$(latest_metric_epoch "${run_name}" "${metric}")"
    # Rollout is scheduled only on multiples of ROLLOUT_EVERY. A chunk may be
    # capped at a non-rollout epoch, so compare against the latest epoch where
    # rollout was actually due, not blindly against cur - 1.
    expected_metric_epoch=$(( ((cur - 1) / ROLLOUT_EVERY) * ROLLOUT_EVERY ))
    if [[ "${score}" == "nan" ]]; then
      echo "[ERROR] ${stage} completed a training chunk but '${metric}' was never logged." >&2
      echo "[ERROR] This is an eval/code-bundle failure, not a zero success rate." >&2
      echo "[ERROR] Stop now instead of spending more epochs without a usable gate." >&2
      exit 5
    fi
    if [[ "${metric_epoch}" -ne "${expected_metric_epoch}" ]]; then
      echo "[ERROR] ${stage} latest metric is stale: metric_epoch=${metric_epoch}, expected=${expected_metric_epoch}." >&2
      echo "[ERROR] The current rollout failed or was skipped; refusing to train or switch stages using an older score." >&2
      exit 5
    fi
    passed="$(score_passed "${score}" "${threshold}")"
    pass_count="$(consecutive_passes "${run_name}" "${metric}" "${threshold}")"
    record_best_checkpoint "${stage}" "${run_name}" "${metric}" "${score}" "${metric_epoch}"
    echo "[AUTO] ${stage} latest ${metric}=${score}, threshold=${threshold}, epoch=${cur}/${minimum_epoch}, consecutive=${pass_count}/${REQUIRED_CONSECUTIVE_PASSES}"
    if [[ "${stage}" == "A_HOLD" ]]; then
      contact_score="$(latest_score "${run_name}" "test/frame_grasp_contact_rate")"
      lift_score="$(latest_score "${run_name}" "test/frame_lift_rate")"
      lift_delta="$(latest_score "${run_name}" "test/mean_max_frame_lift_delta")"
      hold_steps="$(latest_score "${run_name}" "test/mean_max_grasp_hold_steps")"
      echo "[AUTO] A direction: contact=${contact_score}, lift=${lift_score}, mean_lift_delta=${lift_delta}, mean_hold_steps=${hold_steps}, stable_grasp=${score}"
      if [[ "${metric_epoch}" -ge "${A_DIRECTION_CHECK_EPOCH}" ]]; then
        contact_passed="$(score_passed "${contact_score}" "${A_MIN_CONTACT_RATE}")"
        if [[ "${contact_passed}" != "yes" ]]; then
          echo "[ERROR] A_HOLD failed its direction guard at epoch=${metric_epoch}: contact=${contact_score}, required>=${A_MIN_CONTACT_RATE}." >&2
          echo "[ERROR] The policy is not reliably reaching/grasping the frame; more epochs would not justify switching stages." >&2
          exit 6
        fi
      fi
      if [[ "${metric_epoch}" -ge "${A_LIFT_CHECK_EPOCH}" ]]; then
        lift_passed="$(score_passed "${lift_score}" "${A_MIN_LIFT_RATE}")"
        if [[ "${lift_passed}" != "yes" ]]; then
          echo "[ERROR] A_HOLD failed its lift guard at epoch=${metric_epoch}: lift=${lift_score}, required>=${A_MIN_LIFT_RATE}." >&2
          echo "[ERROR] Contact exists but the frame is not being lifted; stop before wasting the remaining A budget." >&2
          exit 6
        fi
      fi
      if [[ "${metric_epoch}" -ge "${A_GRASP_CHECK_EPOCH}" ]]; then
        grasp_started="$(score_passed "${score}" "${A_MIN_GRASP_RATE}")"
        if [[ "${grasp_started}" != "yes" ]]; then
          echo "[ERROR] A_HOLD failed its stable-grasp guard at epoch=${metric_epoch}: grasp=${score}, required>=${A_MIN_GRASP_RATE}." >&2
          echo "[ERROR] The policy has not produced even one stable grasp and should be inspected instead of extended." >&2
          exit 6
        fi
      fi
    fi

    if [[ "${passed}" == "yes" && "${cur}" -ge "${minimum_epoch}" && "${pass_count}" -ge "${REQUIRED_CONSECUTIVE_PASSES}" ]]; then
      ckpt="$(latest_ckpt "${run_name}")"
      if [[ -z "${ckpt}" ]]; then
        echo "[ERROR] ${stage} passed but latest checkpoint is missing for ${run_name}." >&2
        exit 4
      fi
      PASSED_CKPT="${BEST_CKPT:-${ckpt}}"
      echo "[AUTO] ${stage} passed. transfer_checkpoint=${PASSED_CKPT}"
      return 0
    fi

    echo "[AUTO] ${stage} not passed; continuing same stage."
  done
}

require_project_root

if [[ -n "${A_CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${A_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] A_CHECKPOINT_PATH not found: ${A_CHECKPOINT_PATH}" >&2
    exit 4
  fi
  A_CKPT="${A_CHECKPOINT_PATH}"
  echo "[AUTO] Using confirmed A checkpoint override: ${A_CKPT}"
else
  PASSED_CKPT=""
  run_until_pass A_HOLD
  A_CKPT="${PASSED_CKPT}"
fi
if [[ "${ALLOW_STAGE_B}" != "true" ]]; then
  echo
  echo "[AUTO] A_HOLD passed its simulator grasp gate. Stopping before B_INSERT."
  echo "[AUTO] A checkpoint: ${A_CKPT}"
  echo "[AUTO] Resume with ALLOW_STAGE_B=true using the same BASE_RUN_NAME."
  exit 0
fi

echo
echo "[AUTO] A simulator grasp rate passed automatically; continuing to B_INSERT."

if [[ -z "${B_FREEZE_ENCODER_UNTIL_EPOCH:-}" ]]; then
  A_SOURCE_EPOCH="$(checkpoint_epoch "${A_CKPT}")"
  B_FREEZE_ENCODER_UNTIL_EPOCH=$((A_SOURCE_EPOCH + B_FREEZE_ENCODER_ADDED_EPOCHS))
fi
echo "[AUTO] B encoder freeze boundary: epoch=${B_FREEZE_ENCODER_UNTIL_EPOCH}"

PASSED_CKPT=""
run_until_pass B_INSERT "${A_CKPT}"
B_CKPT="${PASSED_CKPT}"
if [[ "${ALLOW_STAGE_C}" != "true" ]]; then
  echo
  echo "[AUTO] B_INSERT passed its metric gate. Stopping before C_FULL."
  echo "[AUTO] B checkpoint: ${B_CKPT}"
  echo "[AUTO] Resume with ALLOW_STAGE_B=true ALLOW_STAGE_C=true using the same BASE_RUN_NAME."
  exit 0
fi

echo
echo "[AUTO] ALLOW_STAGE_C=true; continuing from confirmed B checkpoint."


PASSED_CKPT=""
run_until_pass C_FULL "${B_CKPT}"
C_CKPT="${PASSED_CKPT}"

echo
echo "[AUTO] ToolHang auto curriculum completed."
echo "[AUTO] Final checkpoint: ${C_CKPT}"
