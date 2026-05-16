# Spectral-Adaptive Modulated Prior Diffusion (SAMP-Diff)

結合 **A2A 知情初始化**、**頻域（DCT）動作生成** 與 **Flow Matching** 的機器人控制策略。
將動作預測移至頻域，以 6 步 Euler ODE 取代標準 50 步去噪，實現 50Hz+ 即時控制。

---

## 文件索引

| 文件 | 說明 |
| :--- | :--- |
| [Plan.md](./PLAN.md) | 架構概覽、研究計畫、安裝、訓練、評估（本頁） |
| [thesis/A2A.md](./thesis/A2A.md) | A2A Flow Matching 深度技術分析 |
| [thesis/DP4.md](./thesis/DP4.md) | Diffusion Policy 4 深度技術分析 |

---

## 核心創新與技術組合

本研究融合三個技術主幹，缺一不可：

| 技術 | 來源 | 本研究的整合方式 |
| :--- | :--- | :--- |
| **DCT 頻域動作編碼** | FreqPolicy | 所有動作在 DCT 空間生成，降低 token 數量 |
| **A2A 知情熱啟動** | A2A Flow Matching | 以 DCT(A_{t-1})+σε 作為 x_0，縮短 ODE 路徑 |
| **Conditional Flow Matching** | torchcfm | 以 6 步 Euler ODE 取代 50 步 DDPM 去噪 |

整體流程（訓練與推論均在 DCT 空間進行）：

| 步驟 | 領域 | 操作 | 來源 |
| :---: | :--- | :--- | :--- |
| ① | 時域 | 觀測 → 條件向量 c | Diffusion Policy |
| ② | 頻域 | A_prev → DCT → x_0 = F_prev + σε | A2A |
| ③ | 頻域 | MAE Transformer 預測速度 v_θ(x_t, t, c) | FreqPolicy |
| ④ | 頻域 | Euler ODE 6步：x_1 = x_0 + ∫v_θ dt | Flow Matching |
| ⑤ | 時域 | x_1 → iDCT → A_t，執行並存為 A_{t-1} | — |

---

## 系統架構圖 (System Architecture)

```mermaid
graph TB
    subgraph INPUT["輸入層"]
        OBS["觀測 o_t\nstate / image\n(B, T_o, obs_dim)"]
        PREV["前一動作 A_prev\n(B, H, Da)\n★ A2A warm-start buffer"]
    end

    subgraph ENCODER["條件編碼器"]
        OBS --> OBSENC["obs_encoder\nLinear 投影"]
        OBSENC --> COND["global_cond c\n(B, T_o × obs_dim)"]
    end

    subgraph FREQ["頻域變換 (DCT)"]
        PREV --> DCT_P["DCT(A_prev)"]
        DCT_P --> X0["x_0 = F_prev + σ·ε\n★ 知情初始化"]
        GT["GT 動作 A (訓練)"] --> DCT_GT["DCT(A) = x_1"]
    end

    subgraph SAMPNET["SampNet — MAE Transformer"]
        direction TB
        TEMB["time_proj  t → t_emb\n正弦位置編碼"]
        TOK["token_embed\nx_t tokens"]
        ENC["Transformer Encoder × L\nBasicTransformerBlock\ncross-attn 注入 c"]
        DEC["Transformer Decoder × L"]
        FHEAD["flow_head\nLinear→SiLU→Linear"]
        VPRED["v_pred  (B, H, Da)"]
        TOK --> ENC
        TEMB --> ENC
        ENC --> DEC
        DEC --> FHEAD
        FHEAD --> VPRED
    end

    subgraph LOSS_OR_ODE["訓練 / 推論分支"]
        direction LR
        TRAIN["訓練\nx_t = (1-t)x_0 + t·x_1\nu = x_1 − x_0\nL = ‖v_pred − u‖²"]
        INFER["推論 Euler 6 步\nx_{t+dt} = x_t + dt·v_pred\n僅需 6 次 forward pass"]
    end

    subgraph OUTPUT["輸出"]
        X1["頻域結果 x_1"]
        IDCT["iDCT"]
        ACTION["動作 A_t\n(B, n_action_steps, Da)\n→ 存回 A_prev"]
    end

    COND --> ENC
    X0 --> TOK
    X0 --> TRAIN
    DCT_GT --> TRAIN
    VPRED --> TRAIN
    VPRED --> INFER
    INFER --> X1
    X1 --> IDCT
    IDCT --> ACTION
    ACTION --> PREV

    style INPUT fill:#e8f4fd,stroke:#4a90d9
    style ENCODER fill:#fff9e6,stroke:#f0a500
    style FREQ fill:#e8f8e8,stroke:#4caf50
    style SAMPNET fill:#f3e5f5,stroke:#9c27b0
    style LOSS_OR_ODE fill:#fce4ec,stroke:#e91e63
    style OUTPUT fill:#e0f2f1,stroke:#009688
```

---

## 技術基準對比 (Literature Review)

詳細文獻探討請見：
* [**A2A Flow Matching**](./thesis/A2A.md)：知情初始化效率優勢與時域過度平滑局限。
* [**Diffusion Policy 4 (DP4)**](./thesis/DP4.md)：潛在空間擴散穩健性及工業級實時控制運算壓力。

| 方法 | 生成空間 | 推論步數 | Warm-start | 頻域結構 |
| :--- | :--- | :---: | :---: | :---: |
| Diffusion Policy (DDPM) | 時域 | 50 | ✗ | ✗ |
| A2A Flow Matching | 時域 | 10 | ✓ | ✗ |
| FreqPolicy | 頻域 | 50 | ✗ | ✓ |
| **SAMP-Diff v1（本研究）** | **頻域** | **6** | **✓** | **✓** |

---

## 訓練流程圖 (Training Pipeline)

```mermaid
flowchart LR
    subgraph DATA["資料準備"]
        DS["LeRobot / Robomimic\nHuggingFace Hub / HDF5"]
        NORM["LinearNormalizer  [-1, 1]"]
        DS --> NORM
    end

    subgraph BUILD["先驗建構 (A2A warm-start)"]
        NORM --> OBS_T["obs (B, T_o, obs_dim)"]
        NORM --> ACT_T["action (B, H, Da)"]
        ACT_T --> SHIFT["shift → prev_action"]
        SHIFT --> DCT_PREV["DCT(prev_action)"]
        DCT_PREV --> X0T["x_0 = DCT_prev + σ·ε"]
        ACT_T --> DCT_ACT["DCT(action) = x_1"]
    end

    subgraph FM["Flow Matching"]
        X0T --> INTERP["x_t = (1-t)·x_0 + t·x_1\nu = x_1 − x_0"]
        DCT_ACT --> INTERP
    end

    subgraph MODEL["SampNet Forward"]
        OBS_T --> OBSENC2["obs_encoder → global_cond"]
        INTERP --> SAMP["MAE Transformer + flow_head"]
        OBSENC2 --> SAMP
        SAMP --> VPT["v_pred"]
    end

    subgraph OPTIM["最佳化"]
        VPT --> LOSS["L = ‖v_pred − u‖²"]
        LOSS --> BACK["AdamW + cosine LR + EMA"]
    end

    style DATA fill:#e8f4fd,stroke:#4a90d9
    style BUILD fill:#fff3e0,stroke:#f5a623
    style FM fill:#e8f8e8,stroke:#4caf50
    style MODEL fill:#f3e5f5,stroke:#9c27b0
    style OPTIM fill:#fce4ec,stroke:#e91e63
```

---

## 推論流程圖 (Inference Pipeline)

```mermaid
flowchart LR
    subgraph ENV["環境 (MuJoCo / LeRobot Gym)"]
        OT["觀測 o_t  (state ± image)"]
        EXEC["執行動作  8 steps / call"]
    end

    subgraph WARM["A2A 初始化"]
        BUF["warm-start buffer  A_prev"]
        BUF --> DCT2["DCT(A_prev) + σ·ε = x_0"]
        INIT["第一幀: x_0 ~ N(0,I)"]
    end

    subgraph ODE["Euler ODE  6 步"]
        OT --> COND2["obs_encoder → c"]
        DCT2 --> EULER["x_{t+dt} = x_t + dt · v_θ(x_t, t, c)"]
        COND2 --> EULER
        EULER --> X1OUT["x_1"]
    end

    subgraph DECODE["解碼"]
        X1OUT --> IDCT2["iDCT → actions (B, H, Da)"]
        IDCT2 --> SLICE["slice [:, T_o-1 : T_o-1+8]"]
    end

    SLICE --> EXEC
    SLICE --> BUF

    style ENV fill:#e8f4fd,stroke:#4a90d9
    style WARM fill:#fff3e0,stroke:#f5a623
    style ODE fill:#e8f8e8,stroke:#4caf50
    style DECODE fill:#f3e5f5,stroke:#9c27b0
```

---

## 實驗設計 (Experiments)

### Exp-1：頻率先驗的來源 (Frequency Prior Source)

**研究問題**：頻域先驗中各頻率成分的「來源」應是歷史動作還是即時觀測？

| 方案 | 低頻先驗 | 高頻先驗 | 核心概念 |
| :--- | :--- | :--- | :--- |
| **方案 A**（A2A 原版） | A_{t-1}（歷史） | A_{t-1}（歷史） | 全頻靠記憶 |
| **方案 B**（本研究 v1） | A_{t-1}（歷史） | N(0,I)（自由雜訊） | 低頻記憶 / 高頻自由 |
| **方案 C** | A_{t-1}（歷史） | obs 編碼 z_t | 低頻記憶 / 高頻視覺引導 |

**情境對照**：
- **Robomimic Lift**（慣性主導）：動作平滑，歷史先驗應有優勢
- **LeRobot PushT**（視覺主導）：目標隨機位移，需即時修正

**評估指標**：Task Success Rate、Frequency Band Error、Action Jerk、Perturbation Recovery Time

---

## 支援環境與資料集 (Supported Benchmarks)

| 分類 | 名稱 | 資料格式 | Config |
| :--- | :--- | :--- | :--- |
| **基礎驗證** | `LeRobot PushT` | HuggingFace Hub | `lerobot_pusht` |
| **靈巧操作** | `LeRobot ALOHA` 雙臂 | HuggingFace Hub | `lerobot_aloha` |
| **模仿學習** | `Robomimic` lift / can / square | HDF5 (MuJoCo) | `lift_ph` |
| **數據增強** | `MimicGen` | HDF5 | `mimicgen_lift_d0` |
| **工業大數據** | `Bridge V2` | RLDS / zarr | — |
| **高頻控制** | `DROID` | RLDS | — |
| **幾何精度** | `ManiSkill2` | HDF5 | — |
| **多任務通用** | `Meta-World` | 即時生成 | — |
| **實體落地** | `UR_Real_Data`（自行錄製） | zarr | — |

---

## 安裝

**需求**：Linux、CUDA 11.6 驅動、conda（Miniconda / Anaconda）

```bash
cd SAMP_Diff_v1

# 1. 建立 conda 環境（Python 3.9 + PyTorch 1.12.1 + CUDA 11.6）
conda env create -f conda_environment.yaml
conda activate robodiff

# 2. 安裝本地套件
pip install -e .

# 3. 安裝額外依賴（conda_environment.yaml 未包含）
pip install torchcfm torch-dct lerobot

# 4. 安裝 LeRobot gym 模擬環境（依需求選擇）
pip install gym-pusht          # PushT
pip install gym-aloha           # ALOHA 雙臂
```

> conda 環境名稱為 `robodiff`（定義於 `conda_environment.yaml`）。
> 無 conda 環境時，可改用 `bash scripts/install.sh`（建立 `.venv`）。

---

## 資料集準備

### LeRobot（自動下載，無需手動操作）

```bash
# 首次執行 train.py 時自動從 HuggingFace Hub 下載
# 若要預先快取，手動執行：
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/pusht')"
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('lerobot/aloha_sim_transfer_cube_human')"
```

### Robomimic（MuJoCo，需手動下載）

```bash
# 下載原始資料集
python download_data.py --task lift     # 夾取方塊（推薦）
python download_data.py --task can      # 撿鋁罐
python download_data.py --task square   # 螺帽套柱

# 僅做格式轉換（已有 hdf5 時）
python download_data.py --task lift --skip-download
```

輸出路徑：`SAMP_Diff_v1/data/robomimic/datasets/<task>/ph/low_dim_abs.hdf5`

---

## 訓練

```bash
cd SAMP_Diff_v1
conda activate robodiff

# ── LeRobot ──────────────────────────────────────────────────────────
python train.py --config-name=lerobot_pusht
python train.py --config-name=lerobot_aloha

# 換 ALOHA 子任務：覆蓋 repo_id 即可，不需改 yaml
python train.py --config-name=lerobot_aloha task.repo_id=lerobot/aloha_sim_insertion_human

# ── Robomimic / MuJoCo ───────────────────────────────────────────────
python train.py --config-name=lift_ph
python train.py --config-name=can_ph
python train.py --config-name=square_ph

# ── 常用覆蓋參數（Hydra 語法，空格分隔）─────────────────────────────
python train.py --config-name=lerobot_pusht training.device=cuda:1 dataloader.batch_size=128
python train.py --config-name=lift_ph training.num_epochs=1000 training.debug=true
```

訓練輸出：

```
data/outputs/<run_name>/
├── checkpoints/
│   ├── latest.ckpt                         ← 斷點續訓用
│   └── epoch=xxxx-test_mean_score=x.xxx.ckpt
└── wandb/                                  ← 離線 log（mode: offline）
```

> **續訓**：`training.resume: true`（預設），直接重跑同一指令即自動從 `latest.ckpt` 繼續。

---

## 推論 / 評估

`eval.py` 參數：`-c` / `--checkpoint`（必填）、`-o` / `--output_dir`（必填）、`-d` / `--device`（預設 `cuda:0`）

```bash
conda activate robodiff

# Robomimic / MuJoCo
python eval.py \
    -c data/outputs/samp_lowdim_lift_ph/checkpoints/latest.ckpt \
    -o data/eval_output/lift_ph \
    -d cuda:0

# LeRobot PushT
python eval.py \
    -c data/outputs/samp_lowdim_lerobot_pusht/checkpoints/latest.ckpt \
    -o data/eval_output/lerobot_pusht

# CPU 執行（無 GPU）
python eval.py \
    -c data/outputs/samp_lowdim_lerobot_pusht/checkpoints/latest.ckpt \
    -o data/eval_output/lerobot_pusht \
    -d cpu
```

評估結果寫入 `<output_dir>/eval_log.json`（含 `test/mean_score`）。

**部署呼叫週期（Python API）**：

```python
policy.reset()                            # 切換 episode 前清除 warm-start buffer
while not done:
    obs_dict = {'obs': obs_tensor}        # (1, n_obs_steps, obs_dim)
    result   = policy.predict_action(obs_dict)
    action   = result['action']           # (1, n_action_steps, action_dim)
    env.step(action[0])
```

---

## 路線圖 (Roadmap)

| 版本 | 重點 | 狀態 |
| :--- | :--- | :--- |
| **v1**（本版） | DCT + FM + A2A warm-start，全頻統一先驗，LeRobot + Robomimic | ✅ 進行中 |
| **v2** | 高低頻分離先驗（Exp-1 方案 B/C） | 🔲 計畫中 |
| **v3** | 視覺輸入（ResNet18 encoder），Image Policy | 🔲 計畫中 |

