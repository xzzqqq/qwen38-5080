# 决策记录

格式：日期、决定、理由、被否决的选项。

## 2026-10-19 llama.cpp 作为外部工具，不入库

- 决定：`llama.cpp/` 不纳入版本管理；用 `scripts/setup_llama_cpp.sh` 按固定
  commit `f174553` 拉取并编译，commit 和编译参数记录在 `results/`。
- 理由：它只用于 M1 参照，不修改。固定 commit + 脚本已经能保证可复现。
- 被否决：
  - git submodule：会强制递归 clone 一个历史很大的仓库，收益只有钉版本。
  - 把源码 copy 进仓库：体积大、更新麻烦。
- 以后如果必须打 patch，放 `scripts/patches/`，仍然不引入 submodule。

## 2026-10-19 CUDA 工具链用 micromamba env，不动系统

- 决定：编译使用 `/home/xzq/micromamba/envs/cuda13`（nvcc 13.0.88 +
  cudart + cublas 13.0.2.14）；系统的 `/usr/local/cuda` 仍是 12.4，不改。
- 理由：12.4 不支持 `sm_120`；装系统级 CUDA 需要 root，而且会动到驱动相关的
  环境。conda env 可以精确固定版本，也方便复现。
- 被否决：apt/runfile 装系统级 CUDA 13；用 12.4 编 `compute_90` PTX 兜底
  （不符合 `sm_120a` 目标，性能会偏低）。
