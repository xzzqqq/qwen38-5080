# 在 fish 里 source 本文件（从仓库根目录执行）：
#   source scripts/m2_env.fish
#
# 作用：让 FlashInfer JIT 找到 nvcc、CUDA 头文件、curand 头文件和 cache。
# system nvcc 是 12.4，且 curand 头文件在 pip 的 nvidia/cu13 包里。

set -gx CUDA_HOME $HOME/micromamba/envs/cuda13
set -gx VENV $PWD/.venv
set -gx CPATH "$VENV/lib/python3.12/site-packages/nvidia/cu13/include:$CUDA_HOME/targets/x86_64-linux/include"
set -gx LIBRARY_PATH "$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib"
set -gx FLASHINFER_WORKSPACE_BASE "$PWD/.tmp/flashinfer"
set -gx PATH "$CUDA_HOME/bin" $PATH
