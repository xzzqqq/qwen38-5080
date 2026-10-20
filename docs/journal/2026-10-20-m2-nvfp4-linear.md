# 2026-10-20 M2 §2：NVFP4Linear 与对拍

## 目的

把 checkpoint 的 NVFP4 权重接到 FlashInfer `mm_fp4`，得到可替换 `nn.Linear`
的 W4A4 模块，并用独立 reference 验证 layout、activation quant 和 GEMM。

## 实现

- `runner/ops/nvfp4/reference.py`
  - E2M1 查表；低 nibble = 偶数下标，高 nibble = 奇数下标；
  - `dequant_nvfp4`：`e2m1 * weight_scale(F8) * weight_scale_2(F32)`；
  - `quantize_activation_reference`：`input_scale` 是反量化乘数，量化用
    `1/input_scale` 作为 global divisor；
  - `swizzle_scale_128x4`：线性 scale → 128x4 swizzled，纯 PyTorch。
- `runner/ops/nvfp4/linear.py`
  - `NVFP4Linear` 只保留 packed weight、swizzled F8 scale 和 F32 global scale；
    原始 linear scale 构造后释放，避免全模型多占约 1.6 GB；
  - forward：`nvfp4_quantize` → `mm_fp4(backend="cutlass")` → BF16；
  - `b` 传 `weight_packed.T`，`b_descale` 传 swizzled scale 的转置。
- `runner/ops/nvfp4/bytes.py`：权重 / activation / output 字节模型。
- `tests/ops/test_nvfp4.py`：layout、quant、真实权重和尾块对拍。

## 环境坑

第一次实际调用 `mm_fp4` 时，FlashInfer JIT 编译失败：

```text
fatal error: curand_kernel.h: No such file or directory
fatal error: cuda_fp16.h: No such file or directory
```

原因：

- `cuda13` env 的运行时头文件在 `targets/x86_64-linux/include`；
- `curand_kernel.h` 在 pip 的 `nvidia/cu13/include`，不在 `CUDA_HOME/include`。

修复：把两个目录都放进 `CPATH`，库目录放进 `LIBRARY_PATH`，并保持
`FLASHINFER_WORKSPACE_BASE` 在仓库 `.tmp/flashinfer`。已固化到
`scripts/m2_env.sh` / `scripts/m2_env.fish` 和预检脚本。

## 验证

### Layout / quant

- `swizzle_scale_128x4` 与 `block_scale_interleave` 逐字节一致；
- activation 量化与 FlashInfer `nvfp4_quantize` 的 code diff = 0；
- dequant 最大差 `1.2e-7`（浮点舍入）。

### `NVFP4Linear` 对拍

真实权重 `layers.0.linear_attn.in_proj_qkv`，M = 1/8/128/3/17/129：

| M | nrmse | cosine | max_rel |
|---|---|---|---|
| 1 | 0.001654 | 0.9999986 | 0.00223 |
| 8 | 0.001667 | 0.9999986 | 0.00240 |
| 128 | 0.001662 | 0.9999986 | 0.00246 |
| 3 | 0.001659 | 0.9999986 | 0.00304 |
| 17 | 0.001656 | 0.9999986 | 0.00259 |
| 129 | 0.001661 | 0.9999986 | 0.00384 |

固定容差：`nrmse ≤ 5e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 1e-2`、输出 finite。
13 个 pytest 全通过（含 E2M1 边界、padding、真实权重和尾块）。

### Microbench（M2 诊断，不是 tok/s 门槛）

backend `cutlass`，warmup 3 / 10 次中位数：

| preset | M | full ms | quant ms | gemm ms | logical GB/s | peak MiB | ws ≤ MiB |
|---|---|---|---|---|---|---|---|
| in_proj_qkv | 1 | 0.126 | 0.027 | 0.066 | 235 | 69 | 40 |
| in_proj_qkv | 8 | 0.112 | 0.026 | 0.064 | 264 | 69 | 40 |
| in_proj_qkv | 128 | 0.104 | 0.027 | 0.061 | 311 | 74 | 42 |
| gate_proj | 1 | 0.115 | 0.027 | 0.064 | 435 | 88 | 0 |
| gate_proj | 8 | 0.111 | 0.025 | 0.063 | 454 | 88 | 0 |
| gate_proj | 128 | 0.113 | 0.025 | 0.062 | 485 | 94 | 2 |
| lm_head | 1 | 0.813 | 0.026 | 0.802 | 881 | 723 | 0 |
| lm_head | 8 | 0.819 | 0.025 | 0.811 | 878 | 726 | 0 |
| lm_head | 128 | 0.995 | 0.025 | 0.979 | 783 | 785 | 2 |

`ws ≤` 是上界（peak − 模块权重 − activation/output），0 表示 allocator 复用了已有块，
不是“没有 workspace”。`lm_head` 的大 N 路径接近 0.9 TB/s；小 `in_proj_qkv` 的完整
forward 被 quant / Python / launch 空隙拉低，这是 M3 时间线要拆的部分。

## 结论

§2 完成：`NVFP4Linear` 数值正确，M=1/8/128 和尾块都通过；backend 固定
`cutlass`。M2 的 §3 可以开始拼 eager runner。

没有做：整模型加载、KV/state、8K 容量、CUDA Graph、性能验收。

## 产物

- `runner/ops/nvfp4/{reference,linear,bytes}.py`
- `tests/ops/test_nvfp4.py`、`tests/ops/conftest.py`
- `bench/micro/nvfp4_linear.py`
- `scripts/m2_env.{sh,fish}`
- `results/m2/20261020T224231/`（预检 10/10，含 CUDA header 检查）
- `results/m2/20261020T224659-nvfp4-microbench/`（含 workspace 上界）
- `results/m2/20261020T224541-nvfp4-test/`（13 passed）
