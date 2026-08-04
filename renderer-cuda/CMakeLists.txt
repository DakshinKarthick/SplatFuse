# Phase 2 · CUDA renderer build (STUB)
# NOTE: CUDA targets NVIDIA GPUs only — will not build/run on an AMD RX 7600.
# See README.md for the ROCm/HIP and WebGPU alternatives.
cmake_minimum_required(VERSION 3.20)
project(gaussiansplat_renderer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(renderer
  src/main.cpp
  src/forward.cu
  src/backward.cu
)
