#!/usr/bin/env bash
# Submit with your account, for example:
# sbatch --account=ChingWenMa scripts/slurm_fixed_phase.sh --config can_ph ...
#SBATCH --job-name=samp_fixed_phase
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=2-00:00:00
#SBATCH --output=fixed_phase-%j.out
#SBATCH --error=fixed_phase-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from SAMP_Diff_v1}"
PYTHON="${SAMP_PYTHON:-$HOME/.conda/envs/samp-robodiff310/bin/python}"
test -x "$PYTHON" || { echo "Set SAMP_PYTHON to your environment's python" >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Train-only: no simulator workers or phase detector are started by this job.
exec "$PYTHON" -u scripts/train_fixed_phase_task.py "$@"
