#!/usr/bin/env bash
# Run eval.py (with video output) for all 6 ablation experiments.
# Usage:  cd SAMP_Diff_v1 && bash scripts/run_eval_videos.sh
# Output: eval_output/<exp_name>/  (eval_log.json + mp4 videos)

set -e
BASE=/workspace/Spectral-Adaptive-Modulated-Prior-Diffusion/SAMP_Diff_v1
DEVICE=cuda:0

declare -A CKPTS=(
  [v1_baseline]="outputs/2026-05-23/07-37-48/checkpoints/epoch=3050_test_score=0.988_train_score=0.934.ckpt"
  [v2a_split04]="outputs/2026-05-23/15-06-39/checkpoints/epoch=3050_test_score=0.958_train_score=0.990.ckpt"
  [v2b_split08]="outputs/2026-05-23/22-31-12/checkpoints/epoch=3400_test_score=0.989_train_score=0.810.ckpt"
  [v2c_split08_sh02]="outputs/2026-05-24/05-56-21/checkpoints/epoch=3000_test_score=0.995_train_score=0.997.ckpt"
  [v2d_split04_sh02]="outputs/2026-05-24/13-24-18/checkpoints/epoch=2950_test_score=0.989_train_score=0.983.ckpt"
  [v2e_split08_sh05]="outputs/2026-05-25/06-13-00/checkpoints/epoch=3050_test_score=0.985_train_score=0.998.ckpt"
)

# Run order: baseline → a → b → c → d → e
ORDER=(v1_baseline v2a_split04 v2b_split08 v2c_split08_sh02 v2d_split04_sh02 v2e_split08_sh05)

cd "$BASE"
for EXP in "${ORDER[@]}"; do
  CKPT="${CKPTS[$EXP]}"
  OUT="eval_output/${EXP}"

  echo "================================================================="
  echo "[$(date '+%H:%M:%S')] Running eval: $EXP"
  echo "  ckpt : $CKPT"
  echo "  out  : $OUT"
  echo "================================================================="

  # Remove old output if exists (eval.py will ask interactively otherwise)
  rm -rf "$OUT"

  python eval.py \
    -c "$CKPT" \
    -o "$OUT" \
    -d "$DEVICE"

  echo "[$(date '+%H:%M:%S')] Done: $EXP  →  $OUT/eval_log.json"
  echo ""
done

echo "All experiments finished. Results in eval_output/"
