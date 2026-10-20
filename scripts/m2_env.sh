# 在 bash/zsh 里 source 本文件（从仓库根目录执行）：
#   source scripts/m2_env.sh
#
# 作用：让 FlashInfer JIT 找到 nvcc、CUDA 头文件、curand 头文件和 cache。

export CUDA_HOME="${CUDA_HOME:-$HOME/micromamba/envs/cuda13}"
export VENV="${VENV:-$PWD/.venv}"
export CPATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$PWD/.tmp/flashinfer}"
export PATH="$CUDA_HOME/bin:$PATH"
