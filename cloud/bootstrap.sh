#!/usr/bin/env bash
# Reproducible trainer setup for a fresh Ubuntu NVIDIA CUDA development image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
MAX_JOBS="${MAX_JOBS:-3}"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is missing; select an NVIDIA GPU image." >&2
  exit 1
}
command -v nvcc >/dev/null 2>&1 || {
  echo "ERROR: nvcc is missing; use a CUDA *-devel image, not a runtime-only image." >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is missing." >&2
  exit 1
}

python3 - <<'PY'
import sys

supported = {(3, 10), (3, 11)}
current = sys.version_info[:2]
if current not in supported:
    raise SystemExit(
        f"ERROR: Python {current[0]}.{current[1]} is unsupported; "
        "use Python 3.10 or 3.11."
    )
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  build-essential cmake git ninja-build python3-dev python3-venv
apt-get clean

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "${REPO_ROOT}/trainer/requirements.txt"

# gsplat compiles CUDA kernels lazily. Throttling Ninja avoids the host-memory
# OOM observed on 16-32 GB cloud machines.
MAX_JOBS="${MAX_JOBS}" python - <<'PY'
from importlib.metadata import version

from gsplat.cuda._backend import _C

print("gsplat CUDA extension ready:", version("gsplat"))
PY

SPLATFUSE_DOCTOR_REPORT="$(python "${REPO_ROOT}/trainer/train.py" --doctor)"
printf '%s\n' "${SPLATFUSE_DOCTOR_REPORT}"
SPLATFUSE_DOCTOR_REPORT="${SPLATFUSE_DOCTOR_REPORT}" python - <<'PY'
import json
import os

report = json.loads(os.environ["SPLATFUSE_DOCTOR_REPORT"])
if not report.get("ready_for_cuda_training"):
    raise SystemExit(
        "ERROR: trainer doctor did not report ready_for_cuda_training=true."
    )
PY
python "${REPO_ROOT}/trainer/train.py" --self-test

echo "SplatFuse trainer environment is ready at ${VENV_DIR}."
echo "Activate it with: source '${VENV_DIR}/bin/activate'"
