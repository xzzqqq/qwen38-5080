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

## 2026-10-20 M1 后修订 baseline 与 spec 的验收口径

- 决定：
  - M1（IQ4_XS）只作外部参照，不参与 baseline 验收；Phase 1 为 M1–M3。
  - 显存预算改用实测可用 16.61 GB（15841 MiB）；DFlash2 是否装得下待 Phase 3 验证。
  - Phase 2 以约 80% 逻辑带宽（FP8 KV@8K 约 51 tok/s）为进入 spec 的门槛，
    55 tok/s 留作后续目标。
  - Spec 先测容量、τ 和相对同一 target AR 的净加速，按有效输出速度选择
    MTP / DFlash2；量化后 τ 低于 BF16 的 75% 只作诊断。
  - 8K 下先不做层 offload；M2 实测峰值后再定。
- 理由：M1 的 55.9 tok/s 来自 13.35 GB 的 IQ4_XS，不能外推到 14.41 GB 的
  NVFP4；AR 从 80% 提到 85% 带宽只多约 6.25% 吞吐，先验证 spec 的容量和净加速
  收益更高。
- 被否决：
  - 把 M1 的 tok/s 当 baseline 或实现效率参照。
  - 按 16303 MiB 总量而不是 15841 MiB 可用量算显存余量。
  - Phase 2 硬性追 55 tok/s 后再进 spec。

## 2026-10-20 M2 checkpoint 固定与分片策略

- 决定：
  - 固定 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 到 revision
    `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`；下载后校验 `crc32.txt`。
  - 下载两个 shard。vision 和文本混在 shard 1，不能靠“跳过 vision”跳掉整个文件；
    加载时按 `model.visual.` 前缀过滤。
  - embedding 保持 host BF16；文本权重和 `lm_head` 常驻 GPU。
- 理由：文件树和 index 实测 shard 1 有 979 个文本 + 333 个 vision tensor；只下
  shard 2 会缺一半文本权重。固定 revision + CRC 才能保证 M2 和后续阶段读的是同一份权重。
- 被否决：
  - 从 `main` 拉取：不可复现。
  - 用 HTTP Range 拼一个 text-only safetensors：M2 不需要，增加实现和校验复杂度；
    完整下载 17.92 GB 在 371 GB 可用磁盘内可接受。

## 2026-10-20 M2 NVFP4Linear 固定 cutlass 与对拍容差

- 决定：
  - `NVFP4Linear` 的 backend 固定 `cutlass`，不用未记录的 `auto`；后续比较
    `b12x`/`cudnn` 时单独跑。
  - 权重传 `weight_packed.T`，scale 传 128x4 swizzled 后的转置；`alpha` 用
    `input_scale × weight_scale_2`。
  - 模块只保留 swizzled scale，构造后释放原始 linear scale；否则全模型会多占约 1.6 GB。
  - 对拍容差固定为 `nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 1e-2`、输出 finite。
  - FlashInfer JIT 的 `CPATH` 必须同时包含 `cuda13/targets/x86_64-linux/include`
    和 pip 的 `nvidia/cu13/include`，用 `scripts/m2_env.*` 固化。
- 理由：真实权重 M=1/8/128/3/17/129 的 nrmse ≈ 0.00166、cosine ≈ 0.9999986；
  容差留了约 3× 余量，又能抓住 layout/scale 错误。
- 被否决：
  - 先用 `backend="auto"`：不同 shape 可能选不同 backend，M2 需要单一可复现路径。
  - 用「输出看起来差不多」代替固定容差：无法区分 BF16 舍入和 layout 错误。
