#!/usr/bin/env bash
# =============================================================================
# SAMP_Diff_v1 Linux 安裝腳本（無 conda）
# 需求：Python 3.9-3.11、CUDA 11.6 驅動、git
# 用法：cd SAMP_Diff_v1 && bash scripts/install.sh
# =============================================================================
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_VERSION="3.9.18"
PYTHON_TARBALL="Python-${PYTHON_VERSION}.tgz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}"
BUILD_DIR="/tmp/python_build"

# ── 顏色輸出 ─────────────────────────────────────────────────────────────────
info()  { echo -e "\e[32m[INFO]\e[0m  $*"; }
warn()  { echo -e "\e[33m[WARN]\e[0m  $*"; }
error() { echo -e "\e[31m[ERROR]\e[0m $*" >&2; exit 1; }

# ── 0. 前置檢查 ───────────────────────────────────────────────────────────────
command -v git  &>/dev/null || error "未找到 git，請先安裝：sudo apt install git"
command -v curl &>/dev/null || error "未找到 curl，請先安裝：sudo apt install curl"

# ── 1. 系統套件（編譯依賴 + MuJoCo / OpenGL）─────────────────────────────────
info "安裝系統套件（需要 sudo）..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    wget \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    python3-venv

sudo update-ca-certificates

# ── 2. 取得 Python（優先 3.9，次選 3.10/3.11）────────────────────────────────
find_suitable_python() {
    # 優先找 3.9，次選 3.10 / 3.11（Debian 12 系統預設）
    for cmd in python3.9 python3.10 python3.11 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            case "$ver" in 3.9|3.10|3.11) echo "$cmd"; return 0 ;; esac
        fi
    done
    return 1
}

if PYTHON_BIN=$(find_suitable_python); then
    PY_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "已找到 Python ${PY_VER}：$PYTHON_BIN"
    [ "$PY_VER" != "3.9" ] && warn "使用 Python ${PY_VER}（非 3.9），大部分功能正常，若遇依賴問題可手動安裝 Python 3.9"
    # 確保 venv 模組可用（Debian 需要獨立安裝）
    sudo apt-get install -y "python${PY_VER}-venv" python3-venv 2>/dev/null || true
    "$PYTHON_BIN" -m venv --help &>/dev/null \
        || error "python${PY_VER}-venv 安裝失敗，請手動執行：sudo apt install python${PY_VER}-venv"
else
    # 快速路徑：apt
    info "嘗試透過 apt 安裝 python3.9..."
    APT_PY_OK=false
    for pkg in python3.9 python3.11 python3.10; do
        if sudo apt-get install -y "${pkg}" "${pkg}-venv" "${pkg}-dev" 2>/dev/null \
           && command -v "${pkg%%-*}" &>/dev/null; then
            PYTHON_BIN="${pkg%%-*}"
            APT_PY_OK=true
            info "apt 安裝成功：$(${PYTHON_BIN} --version)"
            break
        fi
    done

    if ! $APT_PY_OK; then
        # 嘗試從原始碼編譯 Python 3.9
        info "apt 無合適 Python，改從原始碼編譯（約 5-10 分鐘）..."
        mkdir -p "$BUILD_DIR"
        pushd "$BUILD_DIR" >/dev/null

        if [ ! -f "$PYTHON_TARBALL" ]; then
            info "下載 Python ${PYTHON_VERSION} 原始碼（依序嘗試多種來源）..."
            # 自動偵測 Nexus raw proxy（與 apt 同一主機）
            NEXUS_RAW=$(grep -oP 'http://[^/]+' /etc/apt/sources.list \
                /etc/apt/sources.list.d/*.list 2>/dev/null | grep -v '^#' | head -1 || true)
            DL_OK=false
            for url in \
                "${NEXUS_RAW}/repository/raw-proxy/https/www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}" \
                "${NEXUS_RAW}/repository/files/${PYTHON_TARBALL}" \
                "ftp://ftp.python.org/pub/python/${PYTHON_VERSION}/${PYTHON_TARBALL}" \
                "$PYTHON_URL" \
            ; do
                [ -z "$url" ] && continue
                wget -q --show-progress "$url" -O "$PYTHON_TARBALL" 2>/dev/null \
                || wget -q --show-progress --no-check-certificate "$url" -O "$PYTHON_TARBALL" 2>/dev/null \
                || curl -fkL "$url" -o "$PYTHON_TARBALL" 2>/dev/null \
                && { DL_OK=true; break; } || true
            done
            if ! $DL_OK || [ ! -s "$PYTHON_TARBALL" ]; then
                error "所有下載方式均失敗（外部網路被封鎖）。
  選項 A — 手動上傳 tarball 後重跑：
    scp Python-3.9.18.tgz root@<此主機>:/tmp/python_build/Python-3.9.18.tgz
    bash scripts/install.sh
  選項 B — 直接用系統 Python 3.11（Debian 12 已內建）：
    python3 -m venv .venv && source .venv/bin/activate"
            fi
        fi

        tar -xzf "$PYTHON_TARBALL"
        cd "Python-${PYTHON_VERSION}"
        ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local 2>&1 | tail -5
        make -j"$(nproc)" 2>&1 | tail -5
        sudo make altinstall 2>&1 | tail -5
        popd >/dev/null

        PYTHON_BIN="python3.9"
        command -v python3.9 &>/dev/null || error "編譯後仍找不到 python3.9，請檢查 /usr/local/bin"
        info "Python 3.9 編譯完成：$(python3.9 --version)"
    fi
fi

# ── 2b. 建立 'python' 系統指令（Debian 12 預設不提供）───────────────────────
if ! command -v python &>/dev/null; then
    PYTHON39_PATH="$(command -v python3.9 2>/dev/null || true)"
    if [ -n "$PYTHON39_PATH" ]; then
        sudo ln -sf "$PYTHON39_PATH" /usr/local/bin/python
        sudo ln -sf "$PYTHON39_PATH" /usr/local/bin/python3
        info "已建立 symlink：/usr/local/bin/python -> $PYTHON39_PATH"
    fi
fi

# ── 3. 建立虛擬環境 ───────────────────────────────────────────────────────────
# 若目錄存在但 activate 不存在，代表上次建立失敗，先清除再重建
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    warn "偵測到不完整的虛擬環境（缺少 activate），清除後重建..."
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    info "建立虛擬環境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

info "啟用虛擬環境..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 3b. 自動偵測內部 PyPI proxy（從 apt sources.list 推斷 Nexus 位址）────────
NEXUS_BASE=$(grep -oP 'http://[^/]+' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null \
    | grep -v '^#' | head -1)
if [ -n "$NEXUS_BASE" ]; then
    PYPI_PROXY_CANDIDATES=(
        "${NEXUS_BASE}/repository/pypi-proxy/simple/"
        "${NEXUS_BASE}/repository/pypi/simple/"
    )
    for candidate in "${PYPI_PROXY_CANDIDATES[@]}"; do
        if curl -sf --max-time 3 "$candidate" >/dev/null 2>&1; then
            PYPI_PROXY="$candidate"
            NEXUS_HOST="${NEXUS_BASE#http://}"
            pip config set global.index-url "$PYPI_PROXY"
            pip config set global.trusted-host "$NEXUS_HOST"
            info "偵測到內部 PyPI proxy：$PYPI_PROXY"
            break
        fi
    done
    [ -z "${PYPI_PROXY:-}" ] && warn "未找到可用的 PyPI proxy，pip 將直接連線 PyPI（若網路受限可能失敗）"
fi

pip install --upgrade pip setuptools wheel

# ── 4. PyTorch（CUDA 11.6）────────────────────────────────────────────────────
info "安裝 PyTorch 1.12.1 + CUDA 11.6..."
pip install \
    torch==1.12.1+cu116 \
    torchvision==0.13.1+cu116 \
    --index-url https://download.pytorch.org/whl/cu116

# ── 4. PyTorch3D（需配合 CUDA 版本）─────────────────────────────────────────
info "安裝 PyTorch3D 0.7.0..."
pip install \
    "fvcore==0.1.5.post20221221" \
    "iopath==0.1.9"
pip install \
    pytorch3d==0.7.0 \
    --index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1121/download.html \
    || warn "PyTorch3D 預編譯輪子下載失敗，嘗試從源碼編譯（較慢）..." \
    && pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.0"

# ── 5. MuJoCo 依賴 ────────────────────────────────────────────────────────────
info "安裝 free-mujoco-py..."
pip install free-mujoco-py==2.1.6

# ── 6. robosuite（自訂分支）──────────────────────────────────────────────────
info "安裝 robosuite（cheng-chi 分支）..."
pip install "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"

# ── 7. 主要依賴 ───────────────────────────────────────────────────────────────
info "安裝 requirements.txt..."
pip install -r requirements.txt

# ── 8. 本地套件 + SAMP 專用 ──────────────────────────────────────────────────
info "安裝本地 diffusion_policy 套件..."
pip install -e .

info "安裝 SAMP-Diff 專用套件（torchcfm, torch-dct, lerobot）..."
pip install torchcfm torch-dct
pip install lerobot

info "安裝 LeRobot gym 環境（選填，可依需求安裝）..."
pip install gym-pusht gym-aloha gym-xarm || true   # 失敗不中斷

# ── 完成 ──────────────────────────────────────────────────────────────────────
info "================================================================"
info " 安裝完成！啟用環境："
info "   source $VENV_DIR/bin/activate"
info " 訓練（LeRobot）："
info "   python train.py --config-name=lerobot_pusht"
info " 訓練（Robomimic）："
info "   python train.py --config-name=lift_ph"
info "================================================================"
