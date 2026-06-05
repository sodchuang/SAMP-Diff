# Spectral-Adaptive Modulated Prior Diffusion (SAMP-Diff)

結合 **A2A 知情初始化**、**頻域（DCT）動作生成** 與 **Flow Matching** 的機器人控制策略。
本論文專注於 low-dim（低維狀態）控制，不討論影像觀測，所有架構、流程與實驗皆針對 low-dim。
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
---
![flow](./low-dimm-SAMP-Difff_flow.jpg)
![training flow](./train_path(1).png)
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

## 實驗設計 (Experiments)


### Exp-1：頻段分離先驗消融（PushT lowdim）

#### Diffusion Policy (DP) 評估結果（50 episodes）

| 指標 | Mean Score | Path Length | Jerk Cost | Discontinuity |
| :--- | :---: | :---: | :---: | :---: |
| DP (50ep) | 95.1% ± 1.8% | 10.6 | -3.800e-02 | 0.04 |
| ACT (50ep) | 79.0% ± 4.2% | 19.8 | 1.381e+04 | 0.07 |


**研究問題**：Flow Matching 的 x₀ 應如何按頻段差異化設計，才能同時兼顧任務成功率與動作平滑度？

**先驗建構（v2 改進）**

```
DCT(prev) → 分三段：
  [ :fl ]   帶外低頻  →  DCT(prev) + σ_high · ε  （σ_high<0 → 純 N(0,I)）
  [ fl:fh ] 目標帶    →  DCT(prev) + σ · ε         ← A2A warm-start
  [ fh: ]  帶外高頻  →  DCT(prev) + σ_high · ε
```

**消融實驗清單**（腳本：`scripts/run_experiments.sh`）

| 實驗 ID | freq_split | sigma | sigma_high | 說明 | 狀態 |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `v1_baseline` | [0, 16]（全帶） | 0.3 | 0.3 | 原始 v1，全頻統一 warm-start | ✅ Score 0.986, Disc 18.20, Jerk 3.79e9 |
| `v2a_split04` | [0, 4] | 0.3 | −1 | 低4頻 warm，其餘純隨機 | ✅ Score 0.948, Disc 37.88, Jerk 2.89e10 |
| `v2b_split08` | [0, 8] | 0.3 | −1 | 低8頻 warm，其餘純隨機 | ✅ Score 0.942, Disc 37.72, Jerk 3.37e10 |
| `v2c_split08_sh02` ★ | [0, 8] | 0.3 | 0.2 | 帶外弱錨定，**當前最佳** | ✅ Score 0.984, Disc 15.84, Jerk 2.53e9 |
| `v2d_split04_sh02` | [0, 4] | 0.3 | 0.2 | 消融：split 寬度對比 | ✅ Score 0.987, Disc 15.87, Jerk 2.79e9 |
| `v2e_split08_sh05` | [0, 8] | 0.3 | 0.5 | 消融：sigma_high 強弱對比 | ✅ Score 0.937, Disc 21.23, Jerk 7.41e9 |

**已完成實驗結果（官方跑出）**

| 指標 | v1 baseline | v2a（split04, sh=−1） | v2b（split08, sh=−1） | v2c ★（split08, sh=0.2） | v2d（split04, sh=0.2） | v2e（split08, sh=0.5） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Score | 0.9861 ± 0.067 | 0.9479 ± 0.103 | 0.9418 ± 0.155 | **0.9835 ± 0.072** | 0.9872 ± 0.019 | 0.9371 ± 0.207 |
| Discontinuity (px) | 18.20 ± 2.36 | 37.88 ± 5.24 | 37.72 ± 5.34 | **15.84 ± 3.27** | 15.87 ± 1.54 | 21.23 ± 3.20 |
| Jerk Cost | 3.79e9 ± 1.58e9 | 2.89e10 ± 7.40e9 | 3.37e10 ± 8.00e9 | **2.53e9 ± 1.22e9** | 2.79e9 ± 8.21e8 | 7.41e9 ± 1.95e9 |
| Path Length (px) | 3919 ± 1215 | 11554 ± 2813 | 11722 ± 2888 | **3397 ± 1120** | 4037 ± 1084 | 5602 ± 1719 |

**評估指標**：Mean Score（任務成功率）、Action Discontinuity（chunk 邊界跳躍量）、Action Jerk、Path Length

---
#### Consistency Policy 與 Diffusion Policy 指標對比

| 評估指標 (Metrics) | 傳統 Diffusion Policy (K=16) | Consistency Policy (1-Step) | Consistency Policy (2-Step) |
| :--- | :---: | :---: | :---: |
| 成功率 (Mean Score) | 95.1% ± 1.8% | 85.6% ± 3.2% | 93.4% ± 2.1% |
| 推理速度 (Latency / FPS) | 約 10 ~ 15 FPS (慢) | 約 70+ FPS (極快) | 約 40+ FPS (快) |
| 軌跡流暢度 (Jerk Cost) | 低 (平滑) | 輕微粗糙 | 接近 Diffusion (平滑) |

---


### Exp-1.1：環境與觀測魯棒性實驗（Robustness to Imperfect World）


**1. 觀測雜訊注入實驗 (Observation Noise Injection)**
    - 在 eval.py 內，於每次環境 obs 讀取時，對所有狀態維度加上 $\epsilon \sim \mathcal{N}(0, \sigma^2)$，其中 $\sigma$ 取 $[0.01, 0.03, 0.05, 0.10]$，每組獨立重複 10 次以取平均。
    - 評估指標：任務成功率（Mean Score）隨雜訊強度變化，繪製「觀測雜訊強度 vs 成功率」折線圖。
    - 理論亮點：SAMP-Flow 頻域生成具備天然低通濾波，對高頻觀測雜訊有顯著抑制效果，成功率曲線遠優於時域 baseline，展現 Inherent Robustness。

**2. 動態模擬擾動與外力推擠 (Simulated External Perturbation)**
    - 在 MuJoCo (Push-T) 環境中，定期對方塊施加隨機外力，模擬現實工廠干擾。
    - 指標：統計模型被干擾後的 Recovery Rate（回正成功率）。

### Exp-1.2：控制系統極限壓測（System Stress Testing）

**3. 動作分塊長度 $H$ (Action Horizon) 的極限消融與壓測**
    - 訓練時強行改變 Action Horizon，$H = [8, 16, 32, 64, 128]$。
    - 對比時域模型（ACT, Diffusion）在 $H$ 極短/極長時的崩潰現象。
    - SAMP-Flow 頻域生成，iDCT 可保證極長序列下依舊平滑。
    - 預期圖表：雙軸圖（橫軸 $H$，左：成功率，右：Jerk Cost）。
    - 論文亮點：突破時域模型在 Horizon 長度上的物理限制。

### Exp-1.3：推理時效與硬體延遲體檢（Inference Latency Profile）

**4. 推理延遲分解壓測**
    - profile_time.py 壓測推論 1000 次平均時間，拆解為：
        1. 觀測編碼 (Obs Encoding)
        2. 6步 Euler ODE 頻域求解 (SampNet Forward)
        3. 逆餘弦變換解碼 (iDCT)
    - 預期圖表：堆疊條形圖，總耗時 < 15ms 即可滿足 50Hz 工業控制循環。
    - 論文亮點：量化證明 SAMP-Flow 可即時落地於真實機器人。

---

### Exp-2：視覺輸入（PushT Image，v3）

**（本論文專注於 low-dim，影像輸入相關內容略）**

---

### Exp-2：ALOHA 雙臂操作（v4）

**目標**：將 SAMP-Diff 的頻段分離先驗推廣到高維度（14-dim）、長 horizon（100）的雙臂操作任務，驗證跨任務泛化。

| 項目 | 設定 |
| :--- | :--- |
| 任務 | `AlohaTransferCube-v0`（sim），可換 `AlohaInsertion-v0` |
| obs / action dim | 14 / 14（關節角） |
| horizon / n_action_steps | 100 / 100（ACT 風格，一次規劃整個 chunk） |
| 資料集 | `lerobot/aloha_sim_transfer_cube_scripted`（HuggingFace 自動下載） |
| Config | `config_task/low_dim/lerobot_aloha.yaml` |
| 狀態 | **計畫中**，等 PushT 消融完成後啟動 |

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

**需求**：Linux、CUDA 11.6 驅動、git、curl

```bash
cd SAMP_Diff_v1

# 一鍵安裝（自動處理 Python 3.9、PyTorch、MuJoCo 依賴、lerobot）
bash scripts/install.sh

# 啟用環境
source .venv/bin/activate
```

`install.sh` 自動完成：

| 步驟 | 內容 |
| :--- | :--- |
| 1 | `apt` 裝系統套件（`libosmesa6-dev libglfw3 patchelf` 等 MuJoCo / OpenGL 依賴） |
| 2 | 偵測或編譯 Python 3.9，建立 `.venv` |
| 3 | PyTorch 1.12.1 + CUDA 11.6 |
| 4 | `requirements.txt` |
| 5 | `pip install -e .` |
| 6 | `pip install torchcfm torch-dct lerobot` |
| 7 | `pip install gym-pusht gym-aloha gym-xarm` |

> conda 用戶可改用：`conda env create -f conda_environment.yaml && conda activate robodiff`，再補裝 `pip install -e . torchcfm torch-dct lerobot gym-pusht gym-aloha`。

---

## 資料集準備

**無需任何操作**，直接執行訓練指令，首次啟動時自動從 HuggingFace Hub 下載並快取到 `~/.cache/huggingface/`。

---

## 訓練

```bash
cd SAMP_Diff_v1
source .venv/bin/activate

python train.py --config-name=lerobot_pusht
python train.py --config-name=lerobot_aloha

# 換 ALOHA 子任務：覆蓋 repo_id 即可，不需改 yaml
python train.py --config-name=lerobot_aloha task.repo_id=lerobot/aloha_sim_insertion_human

# 常用覆蓋參數（Hydra 語法，空格分隔）
python train.py --config-name=lerobot_pusht training.device=cuda:1 dataloader.batch_size=128
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
source .venv/bin/activate

# LeRobot PushT
python eval.py \
    -c data/outputs/samp_lowdim_lerobot_pusht/checkpoints/latest.ckpt \
    -o data/eval_output/lerobot_pusht

# LeRobot ALOHA
python eval.py \
    -c data/outputs/samp_lowdim_lerobot_aloha/checkpoints/latest.ckpt \
    -o data/eval_output/lerobot_aloha

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
| **v1** | DCT + FM + A2A warm-start，全頻統一先驗，PushT lowdim | ✅ 完成 |
| **v2** | 頻段分離先驗（`freq_split` + `sigma_high`），PushT lowdim 消融 | 🔄 實作完成，消融實驗執行中 |
| **v3** | 視覺輸入（ResNet18 encoder），PushT Image Policy | ✅ 架構實作完成，待訓練 |
| **v4** | ALOHA 雙臂操作（14-dim），LeRobot Hub 資料 | 🔲 計畫中（v2 消融完成後） |

