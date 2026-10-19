#!/usr/bin/env bash
# 拉取并编译 M1 参照用的 llama.cpp。
#
# 版本固定，保证 M1 的数字可复现。llama.cpp 只作为外部参照工具使用，
# 不修改、也不纳入本仓库的版本管理。
#
# 用法：
#   scripts/setup_llama_cpp.sh
#   CUDA_ENV=/path/to/cuda scripts/setup_llama_cpp.sh
set -euo pipefail

LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT="f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb"  # 2026-10-19
# CUDA toolkit：本机使用 micromamba 的 cuda13 环境（nvcc 13.0，支持 sm_120a）。
CUDA_ENV="${CUDA_ENV:-/home/xzq/micromamba/envs/cuda13}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/llama.cpp"

# 首次：只取目标 commit，避免克隆完整历史。
if [ ! -d "${SRC}/.git" ]; then
  git init -q "${SRC}"
  git -C "${SRC}" remote add origin "${LLAMA_REPO}"
  git -C "${SRC}" fetch --depth 1 origin "${LLAMA_COMMIT}"
  git -C "${SRC}" checkout -q FETCH_HEAD
else
  git -C "${SRC}" checkout -q "${LLAMA_COMMIT}"
fi

# 120a-real：Blackwell 的 arch-specific SASS，llama.cpp 对 CUDA >= 12.8 默认就会加。
cmake -S "${SRC}" -B "${SRC}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCMAKE_CUDA_COMPILER="${CUDA_ENV}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_ENV}"

cmake --build "${SRC}/build" --config Release -j --target llama-bench llama-server
