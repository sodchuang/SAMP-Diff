#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${TOOL_HANG_LOG_FILE:-${PROJECT_ROOT}/toolhang.log}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniforge3}"
CONDA_ENV="${CONDA_ENV:-robodiff310}"
EXPECTED_PIPELINE="toolhang_direction_guard_v8_isolated"

cd "${PROJECT_ROOT}"

if [[ ! -f train.py || ! -f scripts/run_tool_hang_auto_curriculum.sh ]]; then
  echo "[RESTART] invalid project root: ${PROJECT_ROOT}" >&2
  exit 2
fi

actual_pipeline="$(
  sed -n 's/^PIPELINE_VERSION="\([^"]*\)"/\1/p' \
    scripts/run_tool_hang_auto_curriculum.sh | head -1
)"
if [[ "${actual_pipeline}" != "${EXPECTED_PIPELINE}" ]]; then
  echo "[RESTART] wrong code version: ${actual_pipeline:-missing}" >&2
  echo "[RESTART] expected: ${EXPECTED_PIPELINE}" >&2
  echo "[RESTART] run 'git -C ${PROJECT_ROOT}/.. pull --ff-only origin main' first." >&2
  exit 3
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  conda_script="${CONDA_ROOT}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_script}" ]]; then
    echo "[RESTART] conda activation script not found: ${conda_script}" >&2
    exit 4
  fi
  # shellcheck disable=SC1090
  source "${conda_script}"
  conda activate "${CONDA_ENV}"
fi

echo "[RESTART] stopping previous ToolHang processes..."
pkill -TERM -f '[r]un_tool_hang_auto_curriculum.sh' 2>/dev/null || true
pkill -TERM -f '[r]un_tool_hang_rescue.sh' 2>/dev/null || true
pkill -TERM -f '[t]rain.py' 2>/dev/null || true
pkill -TERM -f '[w]andb-service' 2>/dev/null || true
sleep 3
pkill -KILL -f '[r]un_tool_hang_auto_curriculum.sh' 2>/dev/null || true
pkill -KILL -f '[r]un_tool_hang_rescue.sh' 2>/dev/null || true
pkill -KILL -f '[t]rain.py' 2>/dev/null || true
pkill -KILL -f '[w]andb-service' 2>/dev/null || true

if [[ -f "${LOG_FILE}" ]]; then
  archive="${LOG_FILE%.log}_$(date +%Y%m%d_%H%M%S).log"
  mv "${LOG_FILE}" "${archive}"
  echo "[RESTART] previous log archived: ${archive}"
fi

echo "[RESTART] starting ${EXPECTED_PIPELINE}..."
nohup env \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  bash scripts/run_tool_hang_auto_curriculum.sh \
  > "${LOG_FILE}" 2>&1 &
controller_pid=$!

sleep 8
if ! kill -0 "${controller_pid}" 2>/dev/null; then
  echo "[RESTART] startup failed. Last log lines:" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 5
fi

active_run="pending"
active_pipeline="pending"
[[ -f data/outputs/robomimic/.tool_hang_active_run ]] \
  && active_run="$(tr -d '\r\n' < data/outputs/robomimic/.tool_hang_active_run)"
[[ -f data/outputs/robomimic/.tool_hang_active_pipeline ]] \
  && active_pipeline="$(tr -d '\r\n' < data/outputs/robomimic/.tool_hang_active_pipeline)"

echo "[RESTART] running"
echo "[RESTART] controller_pid=${controller_pid}"
echo "[RESTART] pipeline=${active_pipeline}"
echo "[RESTART] base_run=${active_run}"
echo "[RESTART] log=${LOG_FILE}"
echo "[RESTART] monitor: bash scripts/monitor_tool_hang.sh"
