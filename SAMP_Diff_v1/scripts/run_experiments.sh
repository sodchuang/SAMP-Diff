#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — SAMP-Diff 完整消融實驗腳本
#
# 使用方式:
#   chmod +x scripts/run_experiments.sh
#   cd SAMP_Diff_v1
#   conda activate robodiff310
#   bash scripts/run_experiments.sh
#
# 每個 run 結束後會自動計算 motion metrics。
# 所有結果記錄在 results/metrics_summary.txt
#
# 實驗清單:
#   [V1] v1_baseline         — 全頻 warm-start (原始 SAMP-Diff)
#   [V2] v2a_split04         — freq_split=[0,4],  sigma_high=-1  (低頻warm,高頻隨機)
#   [V2] v2b_split08         — freq_split=[0,8],  sigma_high=-1  (中低頻warm,高頻隨機)
#   [V2] v2c_split08_sh02    — freq_split=[0,8],  sigma_high=0.2 (★ 最佳)
#   [V2] v2d_split04_sh02    — freq_split=[0,4],  sigma_high=0.2 (消融: split比較)
#   [V2] v2e_split08_sh05    — freq_split=[0,8],  sigma_high=0.5 (消融: sigma_high比較)
#   [V3] (已註解) v3_image   — ResNet18 + freq-split (待後續)
# =============================================================================

set -e  # 遇到錯誤即停止

# ---- 設定 ---------------------------------------------------------------
DEVICE="${DEVICE:-cuda:0}"                        # 預設 GPU，可用環境變數覆蓋
BASE_CFG="pusht"                                  # config_task/low_dim/pusht.yaml
METRICS_SCRIPT="compute_motion_metrics.py"
RESULTS_DIR="results"
SUMMARY_FILE="${RESULTS_DIR}/metrics_summary.txt"
N_EPISODES=50
NUM_EPOCHS=4000                                    # 每個 run 訓練 epoch 數

mkdir -p "${RESULTS_DIR}"
echo "=================================================================" | tee -a "${SUMMARY_FILE}"
echo "SAMP-Diff 實驗摘要 — $(date)" | tee -a "${SUMMARY_FILE}"
echo "=================================================================" | tee -a "${SUMMARY_FILE}"

# ---- 輔助函數 -----------------------------------------------------------

# run_exp <exp_name> <extra_overrides...>
# 訓練 + 評估，結果寫入 SUMMARY_FILE
run_exp() {
    local EXP_NAME="$1"
    shift
    local OVERRIDES="$*"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] ${EXP_NAME}"
    echo "  overrides: ${OVERRIDES}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 訓練
    python train.py \
        --config-name="${BASE_CFG}" \
        training.device="${DEVICE}" \
        training.num_epochs="${NUM_EPOCHS}" \
        training.resume=false \
        multi_run.run_dir="data/outputs/${EXP_NAME}" \
        logging.name="${EXP_NAME}" \
        logging.mode=online \
        ${OVERRIDES}

    # 找最新 checkpoint
    CKPT=$(ls -t "data/outputs/${EXP_NAME}/checkpoints/"*.ckpt 2>/dev/null | head -1)
    if [[ -z "${CKPT}" ]]; then
        echo "[WARN] ${EXP_NAME}: checkpoint not found, skipping metrics"
        return
    fi

    echo "[METRICS] ${EXP_NAME}: ${CKPT}"

    # 計算 motion metrics
    METRICS_OUT="${RESULTS_DIR}/${EXP_NAME}_metrics.txt"
    python "${METRICS_SCRIPT}" \
        -c "${CKPT}" \
        -d "${DEVICE}" \
        --n_episodes "${N_EPISODES}" \
        2>&1 | tee "${METRICS_OUT}"

    # 摘要寫入
    echo "" | tee -a "${SUMMARY_FILE}"
    echo "[ ${EXP_NAME} ]" | tee -a "${SUMMARY_FILE}"
    grep -E "Mean Score|Path Length|Jerk Cost|Discontinuity" "${METRICS_OUT}" \
        | tee -a "${SUMMARY_FILE}"
}

# =========================================================================
# V1 — 全頻 warm-start (原始 SAMP-Diff 設計)
#   sigma_high 設超大值 → 近似 v1 全頻統一先驗
#   或: freq_split_low=0, freq_split_high=16 → 全部落在 mid band
# =========================================================================

run_exp "v1_baseline" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=16" \
    "policy.sigma=0.3" \
    "policy.sigma_high=0.3" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v1,lowdim,full_warmstart]"

# =========================================================================
# V2a — freq_split=[0,4], sigma_high=-1 (低4頻 warm, 其餘純隨機)
# =========================================================================

run_exp "v2a_split04" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=4" \
    "policy.sigma=0.3" \
    "policy.sigma_high=-1.0" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v2a,lowdim,split04,no_sigma_high]"

# =========================================================================
# V2b — freq_split=[0,8], sigma_high=-1 (低8頻 warm, 其餘純隨機)
# =========================================================================

run_exp "v2b_split08" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=8" \
    "policy.sigma=0.3" \
    "policy.sigma_high=-1.0" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v2b,lowdim,split08,no_sigma_high]"

# =========================================================================
# V2c ★ — freq_split=[0,8], sigma_high=0.2 (最佳設定)
# =========================================================================

run_exp "v2c_split08_sh02" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=8" \
    "policy.sigma=0.3" \
    "policy.sigma_high=0.2" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v2c,lowdim,split08,sigma_high_02,best]"

# =========================================================================
# V2d — freq_split=[0,4], sigma_high=0.2 (消融: split 範圍影響)
# =========================================================================

run_exp "v2d_split04_sh02" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=4" \
    "policy.sigma=0.3" \
    "policy.sigma_high=0.2" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v2d,lowdim,split04,sigma_high_02]"

# =========================================================================
# V2e — freq_split=[0,8], sigma_high=0.5 (消融: sigma_high 大小影響)
# =========================================================================

run_exp "v2e_split08_sh05" \
    "policy._target_=diffusion_policy.policy.samp_lowdim_policy.SampLowdimPolicy" \
    "policy.obs_dim=20" \
    "policy.action_dim=2" \
    "policy.obs_as_global_cond=true" \
    "policy.obs_as_local_cond=false" \
    "policy.freq_split_low=0" \
    "policy.freq_split_high=8" \
    "policy.sigma=0.3" \
    "policy.sigma_high=0.5" \
    "policy.cold_start_prob=0.3" \
    "task.dataset._target_=diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset" \
    "task.env_runner._target_=diffusion_policy.env_runner.pusht_keypoints_runner.PushTKeypointsRunner" \
    "logging.project=samp_diff_ablation" \
    "logging.tags=[v2e,lowdim,split08,sigma_high_05]"

# =========================================================================
# V3 — Image Policy (ResNet18 + freq-split prior) — 待後續
# =========================================================================

# run_exp "v3_image_split08_sh02" \
#     "policy._target_=diffusion_policy.policy.samp_image_policy.SampImagePolicy" \
#     "policy.freq_split_low=0" \
#     "policy.freq_split_high=8" \
#     "policy.sigma=0.3" \
#     "policy.sigma_high=0.2" \
#     "policy.cold_start_prob=0.3" \
#     "policy.crop_shape=[76,76]" \
#     "task.dataset._target_=diffusion_policy.dataset.pusht_image_dataset.PushTImageDataset" \
#     "task.env_runner._target_=diffusion_policy.env_runner.pusht_image_runner.PushTImageRunner" \
#     "logging.project=samp_diff_v3" \
#     "logging.tags=[v3,image,resnet18,split08,sigma_high_02]"

# =========================================================================
# 完成
# =========================================================================

echo ""
echo "================================================================="
echo "全部實驗完成 — $(date)"
echo "結果摘要: ${SUMMARY_FILE}"
echo "================================================================="
cat "${SUMMARY_FILE}"
