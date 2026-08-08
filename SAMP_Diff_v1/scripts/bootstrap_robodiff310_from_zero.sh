#!/usr/bin/env bash
# Bootstrap the SAMP-Diff H100 training + MuJoCo evaluation environment on a
# fresh Ubuntu/Debian machine that does not have conda or mamba installed.
#
# Usage (from the repository root):
#   bash scripts/bootstrap_robodiff310_from_zero.sh
#
# Optional:
#   ENV_NAME=robodiff310 RECREATE=1 bash scripts/bootstrap_robodiff310_from_zero.sh

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-robodiff310}"
MINIFORGE_HOME="${MINIFORGE_HOME:-${HOME}/miniforge3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RECREATE="${RECREATE:-0}"
INSTALL_MIMICGEN="${INSTALL_MIMICGEN:-1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INSTALL_MIMICGEN

log() { printf '\n[BOOTSTRAP] %s\n' "$*"; }
die() { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" != "Linux" ]]; then
    die "This installer is for Linux only."
fi

if [[ "${EUID}" -eq 0 ]]; then
    APT=(apt-get)
else
    command -v sudo >/dev/null 2>&1 || die "sudo is required for system packages."
    APT=(sudo apt-get)
fi

log "Installing Linux build, FFmpeg, and headless MuJoCo dependencies"
export DEBIAN_FRONTEND=noninteractive
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends \
    build-essential ca-certificates curl wget git \
    ffmpeg pkg-config ninja-build patchelf \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    libgl1 libgl1-mesa-dev libgl1-mesa-dri libglfw3 \
    libglew-dev libosmesa6-dev libegl1 libgles2 \
    libsm6 libxext6 libxrender1 libxi6 \
    libxrandr2 libxinerama1 libxcursor1

if [[ ! -x "${MINIFORGE_HOME}/bin/conda" ]]; then
    log "Installing Miniforge into ${MINIFORGE_HOME}"
    installer="$(mktemp --suffix=.sh)"
    curl -fL --retry 5 --retry-delay 2 \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o "${installer}"
    bash "${installer}" -b -p "${MINIFORGE_HOME}"
    rm -f "${installer}"
else
    log "Using existing Miniforge at ${MINIFORGE_HOME}"
fi

# shellcheck disable=SC1091
source "${MINIFORGE_HOME}/etc/profile.d/conda.sh"
conda config --set solver classic >/dev/null 2>&1 || true
conda init bash >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    if [[ "${RECREATE}" == "1" ]]; then
        log "Removing existing ${ENV_NAME} because RECREATE=1"
        conda deactivate >/dev/null 2>&1 || true
        conda env remove -n "${ENV_NAME}" -y
    else
        die "Conda environment ${ENV_NAME} already exists. Use RECREATE=1 to replace it."
    fi
fi

log "Creating ${ENV_NAME} with Python ${PYTHON_VERSION}"
conda create -n "${ENV_NAME}" -y \
    "python=${PYTHON_VERSION}" \
    pip=24.0 setuptools=65.5.0 wheel=0.38.4 ninja
conda activate "${ENV_NAME}"

# robosuite v1.2 still imports abstract collection classes from `collections`.
# Python 3.10 moved them to `collections.abc`. Provide an environment-only
# compatibility shim without editing robosuite or the SAMP-Diff source tree.
site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' \
    'import collections, collections.abc; collections.Iterable=collections.abc.Iterable; collections.Mapping=collections.abc.Mapping; collections.MutableMapping=collections.abc.MutableMapping; collections.Sequence=collections.abc.Sequence; collections.MutableSequence=collections.abc.MutableSequence; collections.Callable=collections.abc.Callable' \
    > "${site_packages}/robosuite_py310_compat.pth"

python -m pip install --upgrade \
    pip==24.0 setuptools==65.5.0 wheel==0.38.4

log "Installing PyTorch 2.3.1 with CUDA 12.1 runtime"
python -m pip install \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

cd "${PROJECT_ROOT}"
[[ -f requirements_train_h100.txt ]] \
    || die "requirements_train_h100.txt is missing from ${PROJECT_ROOT}"
[[ -f requirements_runtime_h100.txt ]] \
    || die "requirements_runtime_h100.txt is missing from ${PROJECT_ROOT}"

log "Installing pinned SAMP-Diff runtime packages"
python -m pip install -r requirements_runtime_h100.txt

log "Installing pinned SAMP-Diff training packages"
python -m pip install -r requirements_train_h100.txt
python -m pip install \
    hydra-core==1.2.0 omegaconf==2.2.3 antlr4-python3-runtime==4.9.3 \
    PyYAML==6.0.1 packaging==23.2 cloudpickle==2.2.1 pycparser==2.21 \
    llvmlite==0.39.1

# Install the small W&B / HTTP dependency tree normally. Installing requests
# with --no-deps leaves urllib3, certifi, idna, and charset-normalizer absent,
# which makes Hydra report a misleading workspace import failure.
python -m pip install --upgrade-strategy only-if-needed \
    requests==2.28.1 wandb==0.13.3 \
    pyparsing==3.1.2 beautifulsoup4==4.12.3 \
    contourpy==1.0.7 cycler==0.11.0 fonttools==4.38.0 \
    kiwisolver==1.4.4 python-dateutil==2.8.2

# torchdyn declares its notebook and differential-equation stack as required
# metadata even though SAMP training only calls torchcfm. Install compatible
# releases so pip check and direct torchdyn imports are both clean.
python -m pip install --upgrade-strategy only-if-needed \
    ipykernel==6.29.5 ipywidgets==8.1.3 poethepoet==0.10.0 \
    pytorch-lightning==1.8.6 torchmetrics==0.11.4 \
    torchcde==0.2.5 torchsde==0.2.6

# Gym 0.21 requires the older wheel metadata parser. Keep this after the main
# requirements install so another package cannot silently upgrade wheel first.
python -m pip install wheel==0.38.4 setuptools==65.5.0
python -m pip install --no-build-isolation gym==0.21.0

# torchcfm 1.0.7 imports cleanly in this project, but its package metadata also
# declares torchdyn. Install the compatible torchdyn release explicitly.
python -m pip install torchcfm==1.0.7 torchdyn==1.0.6 torch-dct==0.1.6

log "Installing legacy robosuite / mujoco_py evaluation stack"
python -m pip install robomimic==0.2.0
python -m pip install --no-deps free-mujoco-py==2.1.6
python -m pip install --no-deps \
    "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
python -m pip install --no-deps \
    "git+https://github.com/ARISE-Initiative/robosuite-task-zoo.git@74eab7f88214c21ca1ae8617c2b2f8d19718a9ed"

# These exact versions satisfy free-mujoco-py. They are normal Python packages,
# so install them normally; the isolation boundary is only MuJoCo / robosuite.
python -m pip install --upgrade-strategy only-if-needed \
    glfw==1.12.0 fasteners==0.15 Cython==0.29.36 cffi==1.15.1

log "Installing dm-control and its modern MuJoCo binding without changing legacy pins"
python -m pip install --no-deps \
    dm-control==1.0.9 mujoco==2.3.7 \
    dm-env==1.6 dm-tree==0.1.8 labmaze==1.0.6 \
    lxml==4.9.4 PyOpenGL==3.1.7

if [[ "${INSTALL_MIMICGEN}" == "1" ]]; then
    log "Installing MimicGen from the official NVIDIA repository"
    python -m pip install \
        "git+https://github.com/NVlabs/mimicgen.git@main"
    python -m pip install gdown==5.2.0 chardet==5.2.0
fi

log "Installing this checkout in editable mode"
python -m pip install -e .

# Re-assert compatibility pins after every source package is installed.
python -m pip install \
    numpy==1.23.5 numba==0.56.4 \
    setuptools==65.5.0 wheel==0.38.4
python -m pip install --upgrade-strategy only-if-needed glfw==1.12.0 fasteners==0.15

log "Writing headless MuJoCo activation variables"
activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${activate_dir}"
printf '%s\n' \
    'export MUJOCO_GL="${MUJOCO_GL:-osmesa}"' \
    'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"' \
    'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"' \
    > "${activate_dir}/samp_diff_mujoco.sh"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

log "Running GPU, training, and simulator import checks"
python - <<'PY'
import os
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] PyTorch cannot see the GPU")
print("gpu:", torch.cuda.get_device_name(0))

import pkg_resources
import hydra
import h5py
from diffusion_policy.policy.samp_lowdim_policy import SampLowdimPolicy
print("training imports: OK")

import mujoco_py
import mujoco
import robosuite
import robosuite_task_zoo
from dm_control import mujoco as dm_mujoco
print("MuJoCo imports: OK")

try:
    import mimicgen.envs.robosuite
    print("MimicGen registration: OK")
except ModuleNotFoundError as exc:
    if exc.name == "mimicgen" and os.environ.get("INSTALL_MIMICGEN") == "0":
        print("MimicGen registration: skipped")
    else:
        raise
PY

log "Checking Hydra composition"
python train.py --config-name=mimicgen_single --cfg job >/dev/null

log "Environment is ready"
printf '%s\n' \
    "source ${MINIFORGE_HOME}/etc/profile.d/conda.sh" \
    "conda activate ${ENV_NAME}" \
    "cd ${PROJECT_ROOT}"
