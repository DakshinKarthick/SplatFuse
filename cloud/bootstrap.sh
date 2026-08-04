#!/usr/bin/env bash
# One-shot setup for a fresh cloud NVIDIA GPU box (RunPod / vast.ai / Lambda).
# Assumes a CUDA 12.x base image. STUB — adjust to your chosen provider/image.
set -euo pipefail

echo "[*] GPU check (must show an NVIDIA card):"
nvidia-smi || { echo "No NVIDIA GPU visible — wrong instance type."; exit 1; }

echo "[*] CUDA toolkit / nvcc:"
nvcc --version || echo "nvcc missing — pick a -devel CUDA image or install the toolkit."

echo "[*] Build deps:"
apt-get update -y && apt-get install -y --no-install-recommends cmake ninja-build git

# TODO: clone your repo (push to GitHub first), then:
#   cd GaussianSplat/renderer-cuda && cmake -B build -G Ninja && cmake --build build
#
# TODO: for Phase 3 training, install the CUDA build of torch (NOT rocm):
#   pip install torch numpy pillow tqdm

echo "[*] Done. See ../renderer-cuda/README.md and ../trainer/README.md."
