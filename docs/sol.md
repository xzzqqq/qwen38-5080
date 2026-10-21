# SOL 模型（M3 实际字节）

- 生成时间：2026-10-22T02:08:31.390810+08:00
- commit：`bff10d3dae0087552a697bc69b1617e8974ef9e6`（dirty=True）
- checkpoint：`models/Qwen3.8-27B-NVFP4-RTX5090` revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：torch 2.14.1+cu130、transformers 5.20.2、FlashInfer 0.7.2、driver 580.105.08
- allocator：`expandable_segments:True`（见 §5 口径说明）
- 生成命令：`.venv/bin/python scripts/m3_sol.py` --bench results/m3/20261022T003024-m3-bench/bench.json

## 1. 权重字节（运行时唯一 storage）

| 类别 | 字节 | GB | 说明 |
|---|---|---|---|
| `nvfp4_weight_packed` | 12,799,180,800 | 12.799 | NVFP4 E2M1 packed（U8，[N, K/2]） |
| `nvfp4_weight_scale_swizzled` | 1,599,897,600 | 1.600 | 每 16 元素一个 UE4M3 scale，128x4 swizzled |
| `bf16_params` | 52,476,928 | 0.052 | norm / conv1d / in_proj_a,b / A_log / dt_bias 等 BF16 参数 |
| `nvfp4_scalars` | 3,208 | 0.000 | weight_scale_2 / input_scale（FP32 标量） |
| `bf16_buffers` | 256 | 0.000 | 非持久 buffer（rotary inv_freq 等） |
| **合计（GPU）** | **14,451,558,792** | **14.452** | 401 个 NVFP4Linear |
| host embedding（不计入 GPU） | 2,542,796,800 | 2.543 | BF16，常驻 host |

vision tower 未加载；embedding 在 host，均不计入每步 GPU 流量。

## 2. KV / GDN / conv state 字节

| capacity | KV（16 层 BF16 K+V） | GDN recurrent（48 层 FP32） | conv（48 层 BF16） | 合计 |
|---|---|---|---|---|
| 1024 | 67,108,864 (0.067 GB) | 150,994,944 | 3,932,160 | 0.222 GB |
| 8192 | 536,870,912 (0.537 GB) | 150,994,944 | 3,932,160 | 0.692 GB |

## 3. 每步理想字节与 SOL

```text
B_ideal(t) = B_weights
           + 2 * 16 * 4 * 256 * kv_bytes * L_t   # K/V 读
           + 2 * 16 * 4 * 256 * kv_bytes         # 新 K/V 写
           + 2 * B_gdn_state                     # recurrent 读写
           + 2 * B_conv_state                    # conv 读写
B_mean = mean(B_ideal(t)),  t = 1..511,  L_t = depth + t
SOL_tok_s = 960e9 / B_mean
```

- **depth 512 / BF16 KV**：B_mean 14.812 GB/step （首步 14.795、末步 14.829），其中 KV 平均 0.050 GB；**SOL 64.8 tok/s**
- **depth 7680 / BF16 KV**：B_mean 15.282 GB/step （首步 15.265、末步 15.298），其中 KV 平均 0.520 GB；**SOL 62.8 tok/s**

> eager / static-exact 的 decode attention 只读有效前缀 `L_t`；static-cap / graph 在 decode 扫满 capacity，按 capacity 代入（见下表 scan 列）。两者都按各自实际扫描长度计算 SOL。

## 4. 实测 vs SOL（BF16 KV，M3 P0）

| scenario | execution | depth | capacity | decode 实际扫描 | B_mean GB（L_t） | capacity 额外流量 GB/step | SOL tok/s | 模型 tok/s | 模型 %SOL | 模型逻辑 GB/s | 系统 tok/s | 系统 %SOL | 系统逻辑 GB/s | peak GB | min free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eager-512 | eager | 512 | dynamic | scan=L_t~767 | 14.812 | 0.000 | 64.8 | 11.89 | 18.4% | 176.2 | 11.78 | 18.2% | 174.5 | 14.780 | 1.357 |
| eager-8192 | eager | 7680 | dynamic | scan=L_t~7935 | 15.282 | 0.000 | 62.8 | 11.68 | 18.6% | 178.5 | 11.53 | 18.3% | 176.2 | 15.428 | 0.413 |
| exact-512-1024 | static-exact | 512 | 1024 | scan=L_t~767 | 14.812 | 0.000 | 64.8 | 11.78 | 18.2% | 174.4 | 11.73 | 18.1% | 173.7 | 14.780 | 1.374 |
| exact-512-8192 | static-exact | 512 | 8192 | scan=L_t~767 | 14.812 | 0.000 | 64.8 | 11.50 | 17.7% | 170.3 | 11.46 | 17.7% | 169.7 | 15.250 | 0.912 |
| exact-8192 | static-exact | 7680 | 8192 | scan=L_t~7935 | 15.282 | 0.000 | 62.8 | 11.58 | 18.4% | 177.0 | 11.54 | 18.4% | 176.3 | 15.428 | 0.514 |
| cap-512-1024 | static-cap | 512 | 1024 | scan=1024（额外 0.017 GB/step） | 14.812 | 0.017 | 64.8 | 11.94 | 18.4% | 176.9 | 11.81 | 18.2% | 175.0 | 14.780 | 1.374 |
| cap-512-8192 | static-cap | 512 | 8192 | scan=8192（额外 0.487 GB/step） | 14.812 | 0.487 | 64.8 | 11.75 | 18.1% | 174.0 | 11.58 | 17.9% | 171.5 | 15.428 | 0.744 |
| cap-8192 | static-cap | 7680 | 8192 | scan=8192（额外 0.017 GB/step） | 15.282 | 0.017 | 62.8 | 11.91 | 19.0% | 182.1 | 11.77 | 18.7% | 179.8 | 15.428 | 0.514 |
| graph-512-1024 | graph | 512 | 1024 | scan=1024（额外 0.017 GB/step） | 14.812 | 0.017 | 64.8 | 31.41 | 48.5% | 465.2 | 31.38 | 48.4% | 464.8 | 14.822 | 1.170 |
| graph-512-8192 | graph | 512 | 8192 | scan=8192（额外 0.487 GB/step） | 14.812 | 0.487 | 64.8 | 22.71 | 35.0% | 336.4 | 22.70 | 35.0% | 336.2 | 15.292 | 0.520 |
| graph-8192 | graph | 7680 | 8192 | scan=8192（额外 0.017 GB/step） | 15.282 | 0.017 | 62.8 | 22.53 | 35.9% | 344.3 | 22.52 | 35.9% | 344.2 | 15.292 | 0.119 |

模型 tok/s 与系统 tok/s 的定义见 `docs/plans/m3.md` §2.3；模型 %SOL 和系统 %SOL 使用同一 B_mean，因此系统百分比包含 sampling / host 传输 / 同步的损失。

## 5. 口径说明

1. **逻辑带宽不是 DRAM 实测**：这里用“每步字节模型 × tok/s”估算；要拿物理字节需要 `ncu` 的 `dram__bytes`。Nsight Systems 的时间线只解释 kernel 与空隙，不提供这份字节。
2. **B_ideal 只含必需访问**：activation quant、临时 tensor、GQA 展开（`repeat_kv`）、static-cap 的 masked 槽访问、ShapeStableLinear 的 padding 都单列，不计入 B_ideal。额外流量不等于有效模型字节变大。
3. **BF16 vs FP8**：M3 只实测 BF16 KV。FP8 的理论值见 §6，标注“未实测”，不为了填表提前引入 FP8 attention kernel。
4. **allocator**：M3 全流程用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，它不改变数值，但会影响 peak/reserved 的观测值。
5. **单位**：本文 GB = 10^9 字节。

## 6. FP8 KV 理论值（未实测）

| 场景 | B_mean GB（BF16→FP8） | SOL tok/s（BF16→FP8） |
|---|---|---|
| depth 512 | 14.812 → 14.787 | 64.8 → 64.9 |
| depth 7680 | 15.282 → 15.021 | 62.8 → 63.9 |

## 7. eager attention 的额外流量（`repeat_kv` 展开）

```text
O(C) = H_kv * C * D * kv_bytes        # 单层、单个 K/V 的紧凑字节
E(C) = H_q  * C * D * kv_bytes        # H_q=24，展开后的字节
B_expand_model(C) = 2 * 16 * (O(C) + 2 * E(C))
```

| 扫描长度 C | 全模型紧凑 KV 读（16 层 K+V） | 展开路径总量估算 | 相比紧凑 KV 的额外量 |
|---|---|---|---|
| 1024 | 0.067 GB | 0.872 GB/step | 0.805 GB/step |
| 8192 | 0.537 GB | 6.979 GB/step | 6.442 GB/step |

`O(C)`/`E(C)` 是**单层单个 K 或 V** 的字节；表里的“紧凑 KV 读”是乘过 `2 * 16`（K+V、16 个 full attention 层）的全模型值。展开路径总量按 `2 * 16 * (O(C) + 2 * E(C))` 算。

这些是数组访问量估算，不是实测 DRAM 流量或毫秒数；cache 命中、kernel 实现和调度都会影响实际成本。static-cap 的 decode 用 capacity 代入，eager/static-exact 用 `L_t` 代入。

## 8. 复现

```bash
make m3-fixtures
make m3-check
make m3-bench
make m3-memory
make m3-sol
```

结果目录：`results/m3/<run-id>-{check,bench,memory}/`；本文件由 `scripts/m3_sol.py` 生成。
