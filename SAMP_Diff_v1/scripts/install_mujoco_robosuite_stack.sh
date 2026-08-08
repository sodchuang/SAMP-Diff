#!/usr/bin/env bash
# Install only the MuJoCo / robosuite simulator stack after the base SAMP-Diff
# training environment is already working.
#
# Intentionally narrow isolation rule:
#   - normal packages install normally
#   - legacy MuJoCo / robosuite packages install with --no-deps
#
# Usage:
#   cd SAMP_Diff_v1
#   bash scripts/install_mujoco_robosuite_stack.sh

set -euo pipefail

INSTALL_MIMICGEN="${INSTALL_MIMICGEN:-1}"

python -m pip install --upgrade-strategy only-if-needed \
    robomimic==0.2.0

# Isolate legacy simulator packages. Their metadata can request dependency
# versions that are incompatible with the Python 3.10 H100 training env, even
# though the runtime works with the pinned packages below.
python -m pip install --no-deps free-mujoco-py==2.1.6
python -m pip install --no-deps \
    "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
python -m pip install --no-deps \
    "git+https://github.com/ARISE-Initiative/robosuite-task-zoo.git@74eab7f88214c21ca1ae8617c2b2f8d19718a9ed"

# Explicit compatible runtime pins for the isolated simulator packages.
python -m pip install --upgrade-strategy only-if-needed \
    glfw==1.12.0 fasteners==0.15 Cython==0.29.36 cffi==1.15.1

python -m pip install --no-deps \
    dm-control==1.0.9 mujoco==2.3.7 \
    dm-env==1.6 dm-tree==0.1.8 labmaze==1.0.6 \
    lxml==4.9.4 PyOpenGL==3.1.7

if [[ "${INSTALL_MIMICGEN}" == "1" ]]; then
    python -m pip install \
        "git+https://github.com/NVlabs/mimicgen.git@main"
    python -m pip install --upgrade-strategy only-if-needed \
        gdown==5.2.0 chardet==5.2.0
fi

site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' \
    'import collections, collections.abc; collections.Iterable=collections.abc.Iterable; collections.Mapping=collections.abc.Mapping; collections.MutableMapping=collections.abc.MutableMapping; collections.Sequence=collections.abc.Sequence; collections.MutableSequence=collections.abc.MutableSequence; collections.Callable=collections.abc.Callable' \
    > "${site_packages}/robosuite_py310_compat.pth"

python - <<'PY'
import mujoco_py
import mujoco
import robosuite
import robosuite_task_zoo
from dm_control import mujoco as dm_mujoco
print("MuJoCo / robosuite imports: OK")

try:
    import mimicgen.envs.robosuite
    print("MimicGen registration: OK")
except ModuleNotFoundError:
    print("MimicGen registration: skipped")
PY
