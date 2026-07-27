#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/robomimic}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-${OUTPUT_ROOT}/.tool_hang_active_run}"
ACTIVE_PIPELINE_FILE="${ACTIVE_PIPELINE_FILE:-${OUTPUT_ROOT}/.tool_hang_active_pipeline}"

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "[MONITOR] output root not found: ${OUTPUT_ROOT}" >&2
  echo "[MONITOR] run this script from the SAMP_Diff_v1 project root." >&2
  exit 2
fi

while true; do
  if [[ -t 1 ]]; then
    clear
  fi

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  pipeline="none"
  base_run="none"
  [[ -f "${ACTIVE_PIPELINE_FILE}" ]] && pipeline="$(tr -d '\r\n' < "${ACTIVE_PIPELINE_FILE}")"
  [[ -f "${ACTIVE_RUN_FILE}" ]] && base_run="$(tr -d '\r\n' < "${ACTIVE_RUN_FILE}")"

  echo "ToolHang monitor  ${now}"
  echo "pipeline: ${pipeline}"
  echo "base_run: ${base_run}"

  if [[ "${base_run}" == "none" ]]; then
    echo
    echo "[WAIT] no active run manifest yet."
    sleep "${INTERVAL}"
    continue
  fi

  latest_log="$(
    find "${OUTPUT_ROOT}" -maxdepth 2 -type f -name logs.json.txt \
      -path "${OUTPUT_ROOT}/${base_run}_*/logs.json.txt" \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -n | tail -1 | cut -d' ' -f2-
  )"

  if [[ -z "${latest_log}" || ! -f "${latest_log}" ]]; then
    echo
    echo "[WAIT] active run exists but logs.json.txt has not been created."
    sleep "${INTERVAL}"
    continue
  fi

  run_dir="$(dirname "${latest_log}")"
  stage="$(basename "${run_dir}")"
  age_seconds="$(( $(date +%s) - $(stat -c %Y "${latest_log}") ))"
  train_processes="$(pgrep -fc '[t]rain.py' || true)"
  controller_processes="$(pgrep -fc '[r]un_tool_hang_auto_curriculum.sh' || true)"

  echo "stage:    ${stage}"
  echo "log:      ${latest_log}"
  echo "updated:  ${age_seconds}s ago"
  echo "process:  controller=${controller_processes} train_family=${train_processes}"
  echo

  python - "${latest_log}" "${run_dir}" <<'PY'
import json
import math
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])

# Read only the recent tail. Rollout records can be large, so keep enough room
# for several complete evaluation lines without repeatedly parsing the full log.
tail_bytes = 16 * 1024 * 1024
with log_path.open("rb") as f:
    size = f.seek(0, 2)
    f.seek(max(0, size - tail_bytes))
    raw = f.read().decode("utf-8", errors="ignore")
if size > tail_bytes:
    raw = raw.split("\n", 1)[-1]

records = []
for line in raw.splitlines():
    try:
        records.append(json.loads(line))
    except Exception:
        pass

if not records:
    print("[WAIT] no complete JSON record available.")
    raise SystemExit(0)

last = records[-1]
epoch = last.get("epoch", "n/a")
step = last.get("global_step", "n/a")
loss = last.get("train_loss", float("nan"))
lr = last.get("lr", float("nan"))
print(f"train: epoch={epoch} global_step={step} loss={loss} lr={lr}")

if isinstance(lr, (int, float)) and math.isfinite(lr) and lr < 1e-7:
    print("[ALERT] LR is effectively zero (<1e-7); learning is stalled.")

metric_names = [
    "test/stage_grasp_rate",
    "test/stage_insert_rate",
    "test/stage_full_rate",
    "test/frame_grasp_contact_rate",
    "test/frame_lift_rate",
    "test/mean_max_frame_lift_delta",
    "test/mean_max_grasp_hold_steps",
]
latest = {}
metric_epochs = {}
for record in records:
    for name in metric_names:
        if name in record:
            latest[name] = record[name]
            metric_epochs[name] = record.get("epoch", "n/a")

if latest:
    print()
    print("latest evaluation:")
    for name in metric_names:
        if name in latest:
            short = name.removeprefix("test/")
            print(f"  epoch={metric_epochs[name]} {short}={latest[name]}")
else:
    print()
    print("[WAIT] no recent evaluation metric in the log tail.")

best_files = sorted((run_dir / "checkpoints").glob("auto_best_*.txt"))
if best_files:
    print()
    print("rolling best:")
    for path in best_files:
        print(f"  {path.stem}: {path.read_text(errors='ignore').strip()}")
PY

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits \
      | awk '{print "gpu: utilization=" $1 "% memory=" $2 "/" $3 " MiB"}'
  fi

  if [[ "${age_seconds}" -gt 180 ]]; then
    echo
    echo "[ALERT] logs have not changed for more than 180 seconds."
  fi

  sleep "${INTERVAL}"
done
