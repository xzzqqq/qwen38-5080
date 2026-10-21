# 2026-10-21 M3 decode baseline：eager / static-exact / static-cap / CUDA Graph

## 状态

M3（P0）完成。四执行方式在同一 checkpoint、dtype、prefill chunk 和 workload 下
建立了可复现的 decode baseline；CUDA Graph 把单步时间从 86 ms 降到 31.8 ms
（depth 512，capacity 1024），8K 下从 87.2 ms 降到 44.4 ms。

- 代码：本次改动（runner/model/{static_cache,padded_reference,graph_runner,
  decode_runner,snapshots}.py、bench/e2e/decode.py、scripts/m3_*.py、
  bench/micro/attn_length_effect.py）
- checkpoint：`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
  revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 环境：torch 2.14.1+cu130、transformers 5.20.2、FlashInfer 0.7.2、
  driver 580.105.08、CUDA 13.0.88、nsys 2026.2.1
- allocator：全部进程 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  （理由见 `docs/decisions.md` 2026-10-21）
- 口径：bs=1、NVFP4 W4A4、KV BF16、GDN state FP32、conv state BF16、
  embedding host BF16、greedy、prefill chunk 128、固定深度 depth 512 / 7680

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 非 graph 的三条路径（eager / static-exact / static-cap） | 11.3–12.0 tok/s，与 capacity、depth、attention 扫描长度基本无关 |
| graph（depth 512，capacity 1024） | **31.41 tok/s**（48.5% SOL，465.7 GB/s 逻辑带宽） |
| graph（depth 512，capacity 8192） | 22.71 tok/s（36.2% SOL）——多出的 12.2 ms/step 全部是扫 capacity 的 GQA 展开 |
| graph（depth 7680，capacity 8192） | 22.53 tok/s（35.9% SOL，344.6 GB/s） |
| 模型速度 vs 系统速度 | graph 相差 <0.1%；eager 相差约 0.8%，采样和 host 传输不是瓶颈 |
| 单步 kernel 数 | eager/static 7242–7290；graph 展开后 7290 个 node |
| eager 的 86.4 ms/step 里 | kernel 累计只有 30.2 ms，其余约 56 ms 是 CPU 提交/间隙 |
| graph 的 31.8 ms/step 里 | NVFP4 GEMM 20.7 ms（696 GB/s，72% 峰值带宽），其余 kernel 9.0 ms |
| 8K 的 graph 42.3 ms kernel 里 | NVFP4 GEMM 21.1 ms、`repeat_kv` 展开 9.4 ms、attention matmul 4.3 ms、其他 7.5 ms |
| graph 常驻后的 8K 峰值 | **15.722 GB**（reserved 16.054 GB），free 仅 **0.119 GB**，无 OOM、无增长 |

一句话：**M3 的瓶颈不是模型带宽，而是 7290 个 tiny kernel 的提交开销；录成整步
graph 后，剩下的瓶颈依次是 NVFP4 GEMV（21 ms）、8K 的 GQA 展开（9.4 ms）和
attention matmul（4.3 ms）。**

## 2. 实现与入口

| 入口 | 文件 | 作用 |
|---|---|---|
| `make m3-fixtures` | `scripts/m3_fixtures.py` | 固定深度 prompt、greedy trace、三条 smoke |
| `make m3-check` | `scripts/m3_check.py` | 四模式 + padded reference + reset/边界/smoke |
| `make m3-bench` | `bench/e2e/decode.py` | depth×capacity 矩阵、两套计时、5 次重复 |
| `make m3-memory` | `scripts/m3_memory.py` | 分阶段显存 + graph 常驻后的完整生命周期 |
| `make m3-profile` | `scripts/m3_profile.sh` + `m3_profile_stats.py` | nsys 采集与 kernel 统计 |
| `make m3-sol` | `scripts/m3_sol.py` | 实际字节账 + `docs/sol.md` |
| `make m3-attn-effect` | `bench/micro/attn_length_effect.py` | padded/unpadded attention 的 FP reduction 差异 |

- `runner/model/static_cache.py`：hybrid static cache 适配。KV 预分配、原地写入；
  `exact` 只把有效前缀 `[:, :, :L]` 交给 attention，`cap` 在 decode 交整块
  capacity、prefill 仍用有效前缀。`cumulative_length` 是 device tensor，graph
  内自行推进。
- `runner/model/padded_reference.py`：独立的 padded DynamicCache reference
  （自己的位置计数、自己的 mask 尺寸），不复用 static adapter 的索引逻辑。
- `runner/model/decode_runner.py`：四模式共用 host gather → pinned → 固定 GPU
  输入 buffer 的 staging，以及模型/系统两套计时。
- `runner/model/graph_runner.py`：整步 capture/replay。**必须显式传
  `past_key_values`**，并加了两道 guard（warmup 必须推进 cache、capture 不推进
  位置）。
- `runner/model/snapshots.py`：逐层 hidden / 有效 KV / GDN+conv state / logits
  的 CPU 快照与 bitwise 比较。快照放 CPU，8K 时也只多占约 1.3 GB 内存。

## 3. 计时协议与 fixtures

- prompt：`bench/workloads/m3-p0/fixed_{512,7680}.json`，来自
  `scripts/m2_capacity.py:fixed_ids` 的重复文本；trace 由 **eager runner** 生成
  （greedy，忽略 EOS，512 个 token）。
- 计时区间：prefill 产生首 token（计入 TTFT），随后 **511 次 decode forward /
  511 个输出 token**；最终 cache 长度 = depth + 511（depth 512 → 1023，
  depth 7680 → 8191）。模型和系统 tok/s 的分子都是 511。
- 模型 tok/s：冻结 trace 驱动，逐步 CUDA events；D2D copy 在 start event 之前。
- 系统 tok/s：真实贪心闭环，墙钟包住 host gather / pinned copy / H2D / 模型 /
  sampling / D2H。
- request tok/s：512 个输出 token ÷（prefill + decode 墙钟）。
- 每个场景先完整 warmup 一次（和正式测量同样的两步），再测 5 次，每次原地 reset
  并重新 prefill；原始 511 个逐步延迟全部保留。
- fixtures hash 在 `bench/workloads/m3-p0/manifest.json`，bench/check 启动时校验。

## 4. 正确性验收（gate / diagnostic 分开）

命令：`make m3-check`（5 个场景，独立子进程）。
结果目录：`results/m3/20261021T180905-m3-check/`。

| 场景 | gate | 结果 |
|---|---|---|
| numeric-512（depth 512 / cap 1024 / 16 步） | 110 | 全通过 |
| numeric-8192（depth 7680 / cap 8192 / 16 步） | 50 | 全通过 |
| graph-1024 | 21 | 全通过 |
| graph-8192 | 21 | 全通过 |
| smoke-graph（三条真实 prompt，2048 上限） | 9 | 全通过 |

gate 项（同 shape，全部 **bitwise** 通过）：

- eager → static-exact：logits、逐层 hidden（64 层 × 17 次 forward）、16 层有效
  KV、48 层 GDN recurrent（FP32）、48 层 conv state、cache 长度。两档 depth 与
  全部边界长度（1/127/128/129/300）都一致。
- padded DynamicCache reference → static-cap：同 shape 的 logits / hidden / KV /
  state 全部 bitwise；fill=0 与 fill=3 的输出也 bitwise 一致（未写入槽不污染结果）。
- static-cap → graph：逐步 logits、最终 KV/state bitwise；logits 地址稳定；
  replay 确实写入 static KV 槽；reset 后重复 replay 结果一致；换输入 token 输出
  会变（证明 graph 读的是输入 buffer）。
- reset：原地 reset 重跑 A、A→B→A、短→长→短、与**独立新建 cache** 对比，全部
  bitwise。
- capacity：1024 / 8192 两档都能填满，越界由 host 侧 guard 拒绝（不让 kernel
  越界），reset 后可继续写。
- smoke：三条都正常 EOS，长度合理，数学答案 72 clips、代码含函数和测试、
  chat 解释到 memory-bandwidth。

diagnostic（不作门槛）：

- eager → padded（attention 长度效应）：首处分叉在 depth 512 的第 3 个 decode
  step（forward 6，layer 31 起）、depth 7680 的第 1 个 decode step（forward 60，
  layer 15 起）；logits 差最大 5.9（512 档 2.875）。
- 该差异的来源用 `make m3-attn-effect` 隔离：同一个 attention op，K/V 长度 513
  vs 补到 1024（mask 后有效 key 相同），BF16 输出最大差 **9.77e-4**（中位
  0.22 ulp，只有 0.5% 元素变化），两条路径与 float64 参考的距离**相同**
  （3.87e-3）。即：FP reduction 顺序不同，不是实现错误。
- free-running greedy（smoke）与冻结 eager trace 的差异：math 首处分叉
  step 30（margin 7.0）、code step 37（0.25）、chat step 27（3.125）。sub-ulp 的
  注意力差异经过 64 层 BF16 残差和 401 个 W4A4 GEMM 被放大到 O(1) logits，
  所以真实生成的 trace 不能当 bitwise gate；逐 step bitwise 由 graph-1024/8192
  的 16 步检查负责。

## 5. 性能结果

### 5.1 两套速度（5 次重复的中位数 [min, max]）

结果目录：`results/m3/20261021T163844-m3-bench/`。

| scenario | execution | depth | capacity | 模型 tok/s | 系统 tok/s | request tok/s | step p50/p95 ms | prefill s | peak GB | reserved GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager-512 | eager | 512 | dynamic | 11.55 [11.47, 11.59] | 11.46 [11.29, 11.50] | 11.37 | 86.40 / 88.51 | 0.43 | 14.780 | 14.915 | 1.357 |
| eager-8192 | eager | 7680 | dynamic | 11.47 [11.41, 11.48] | 11.27 [11.26, 11.31] | 9.85 | 87.16 / 88.07 | 6.69 | 15.428 | 15.859 | 0.413 |
| exact-512-1024 | static-exact | 512 | 1024 | 11.64 [11.60, 11.72] | 11.62 [11.59, 11.64] | 11.53 | 85.76 / 86.66 | 0.43 | 14.780 | 14.898 | 1.374 |
| exact-512-8192 | static-exact | 512 | 8192 | 11.55 [11.51, 11.58] | 11.50 [11.50, 11.52] | 11.41 | 86.52 / 87.30 | 0.43 | 15.250 | 15.360 | 0.912 |
| exact-8192 | static-exact | 7680 | 8192 | 11.62 [11.59, 11.64] | 11.59 [11.54, 11.61] | 10.10 | 85.91 / 86.83 | 6.61 | 15.428 | 15.758 | 0.514 |
| cap-512-1024 | static-cap | 512 | 1024 | 11.27 [11.24, 11.35] | 11.17 [11.15, 11.19] | 11.09 | 88.56 / 89.46 | 0.44 | 14.780 | 14.898 | 1.374 |
| cap-512-8192 | static-cap | 512 | 8192 | 11.95 [11.93, 11.97] | 11.77 [11.72, 11.79] | 11.68 | 83.55 / 84.37 | 0.42 | 15.428 | 15.527 | 0.744 |
| cap-8192 | static-cap | 7680 | 8192 | 11.78 [11.74, 11.83] | 11.61 [11.53, 11.62] | 10.11 | 84.73 / 85.57 | 6.61 | 15.428 | 15.758 | 0.514 |
| **graph-512-1024** | graph | 512 | 1024 | **31.41** [31.41, 31.41] | **31.38** [31.37, 31.38] | 30.65 | 31.84 / 31.86 | 0.42 | 14.822 | 15.005 | 1.170 |
| **graph-512-8192** | graph | 512 | 8192 | 22.71 [22.69, 22.71] | 22.70 [22.67, 22.70] | 22.32 | 44.04 / 44.09 | 0.43 | 15.292 | 15.655 | 0.520 |
| **graph-8192** | graph | 7680 | 8192 | 22.53 [22.53, 22.55] | 22.51 [22.51, 22.54] | 17.49 | 44.38 / 44.42 | 6.58 | 15.292 | 16.054 | 0.122 |

要点：

1. **static cache 本身不带来速度收益**：exact/cap/容量/深度都在 11.3–12.0 tok/s，
   说明当前瓶颈是 CPU 提交而不是 cache 存储或 attention shape。static-exact 的
   `int(cumulative_length)` 每步同步 16 次（每个 KV 层一次），比 cap 略慢。
2. **整步 CUDA Graph 是唯一的数量级变化**：depth 512 从 86.4 ms 降到 31.8 ms
   （2.7×），depth 7680 从 87.2 ms 降到 44.4 ms（2.0×）。
3. **capacity 只影响扫满 capacity 的模式**：exact 在 cap 1024 和 8192 下速度相同
   （11.64 vs 11.55），因为只读有效前缀；graph 从 cap 1024 的 31.8 ms 涨到
   cap 8192 的 44.0 ms，多出的 12.2 ms 就是 8K 的 GQA 展开。
4. 单步延迟分布很窄（p95/p50 < 1.02），5 次 run 之间也稳定；没有观察到热降频。

### 5.2 Nsight Systems：每步 kernel 数与时长分布

结果目录：`results/m3/20261021T182309-m3-profile/`（nsys 2026.2.1，
`--capture-range=cudaProfilerApi` 只采 8 个 decode step，
`--cuda-graph-trace=node` 展开 graph node）。

| case | 每步 kernel/node | 每步 kernel 累计 ms | 每步墙钟 ms | <5 µs kernel 数量/时间占比 |
|---|---|---|---|---|
| eager-512 | 7242 | 30.2 | 110.1 | 93.9% / 27.8% |
| eager-7680 | 7242 | 44.1 | 110.5 | 92.6% / 18.6% |
| static-exact-512 | 7290 | 30.3 | 111.1 | 93.9% / 27.9% |
| static-cap-512 | 7290 | 31.0 | 111.7 | 93.6% / 27.0% |
| **graph-512** | 7290 | 29.7 | **32.6** | 93.6% / 25.9% |
| **graph-7680** | 7290 | 42.3 | **45.0** | 93.6% / 18.2% |

- 表里的“每步墙钟”是**开着 profiler 且每步 synchronize** 的数字，比 bench 的
  86.4 / 87.2 ms 大；kernel 累计不受 host 同步影响。把两者对齐到 bench 的
  wall：非 graph 在 depth 512 是 86.4 ms 墙钟 vs 30.2 ms kernel（≈56 ms 的 CPU
  提交/间隙），depth 7680 是 87.2 vs 44.1（≈43 ms）。graph 把这段压到 3–4 ms。
- 93% 的 kernel 小于 5 µs，但只占 18–28% 的 kernel 时间：**数量上全是 tiny
  kernel，时间上仍是少数大 kernel 主导**。这直接说明 Phase 2 不能只靠 graph，
  必须减少/融合 kernel 或提高大 kernel 的效率。

graph-512 的 top kernel（8 步合计 → 每步）：

| kernel | 每步次数 | 每步 ms | 占比 |
|---|---|---|---|
| `cutlass::device_kernel<...GemmUniversal...>`（FlashInfer `mm_fp4`） | 401 | 20.7 | 69.5% |
| `elementwise_kernel<128,4,direct_copy>` | 192 | 1.2 | 4.0% |
| `unrolled_elementwise_kernel<direct_copy>` | 706 | 1.1 | 3.7% |
| `elementwise_kernel<128,2,BinaryFunctor>` | 578 | 0.94 | 3.2% |
| `tensorrt_llm::quantize_with_block_size`（activation quant） | 401 | 0.52 | 1.8% |
| `wmma bf16 gemm 32x32_128x2_tn`（ShapeStableLinear，M=128） | 96 | 0.40 | 1.7% |

graph-7680 相对 graph-512 多出来的部分：

| 项 | 每步次数 | 每步 ms |
|---|---|---|
| `repeat_kv` 展开（`direct_copy`，16 层 × K/V，各 100 MB） | 32 | **9.4** |
| QK^T matmul（`wmma 16x16_32x1_nn`） | 16 | 2.2 |
| attn@V matmul（`wmma 16x16_128x1_tn`） | 16 | 2.1 |

这与 `docs/sol.md` §7 的估算一致：8K 的展开路径比紧凑 KV 多 6.44 GB/step，
实测 13.7 ms/step，折合约 509 GB/s 的数组访问带宽。

## 6. 显存

命令：`make m3-memory`。结果目录：`results/m3/20261021T181311-m3-memory/`。

| scenario | load peak GB | cache 初始化 GB | 首次 prefill peak GB | capture pool GB | graph 常驻后 prefill peak GB | decode peak GB | 全程 max peak GB | min free GB | reset→prefill 增量 | KV ptr 稳定 |
|---|---|---|---|---|---|---|---|---|---|---|
| eager-8192 | 14.458 | 14.452 | 15.620 | - | 15.621 | 15.428 | 15.621 | 0.411 | 0 MB | - |
| exact-8192 | 14.458 | 14.989 | 15.654 | - | 15.654 | 15.428 | 15.654 | 0.512 | 0 MB | True |
| cap-8192 | 14.458 | 14.989 | 15.654 | - | 15.654 | 15.428 | 15.654 | 0.512 | 0 MB | True |
| graph-8192 | 14.458 | 14.989 | 15.654 | **0.034** | **15.722** | 15.293 | **15.722** | **0.119** | 0 MB | True |
| cap-512-1024 | 14.458 | 14.519 | 14.810 | - | 14.811 | 14.781 | 14.811 | 1.374 | 0 MB | True |
| graph-512-1024 | 14.458 | 14.519 | 14.810 | 0.034 | 14.879 | 14.823 | 14.879 | 1.170 | 0 MB | True |

- static cache 的 KV 预分配使 cache 初始化阶段就从 14.452 涨到 14.989 GB
  （capacity 8192 的 537 MB KV），之后的 prefill/decode 峰值只比 eager 高 34 MB。
- **graph-private pool 只多 34 MB**（expandable segments 下 graph 与普通 pool
  共享同一批 segment），远小于之前担心的“几百 MB”。
- P0 必测生命周期（graph 常驻 → reset → 7680 prefill → 511 decode → reset →
  prefill → 再 decode 16 → reset → prefill）全部通过：无 OOM、无 stale state、
  两个生命周期之间 allocated/reserved 增量为 **0 MB**，KV 地址不变。
- **但 8K + graph 只剩 0.119 GB free（reserved 16.054 GB）**。这比 M2 的
  0.413 GB 更紧；Phase 3 加 drafter 之前必须先做容量预演（roadmap §7 的已知风险）。

## 7. SOL（实际字节）

`docs/sol.md` 由 `make m3-sol` 生成，权重按运行时唯一 storage 统计：

| 类别 | 字节 | 说明 |
|---|---|---|
| NVFP4 packed weight | 12.799 GB | E2M1，0.5 B/参数 |
| NVFP4 swizzled scale | 1.600 GB | 每 16 元素一个 UE4M3 |
| BF16 params | 0.052 GB | norm / conv1d / in_proj_a,b / A_log / dt_bias |
| **合计（GPU）** | **14.452 GB** | 401 个 NVFP4Linear；vision 未加载 |
| host embedding | 2.543 GB | 不进 GPU 每步流量 |

- B_mean：depth 512 = **14.812 GB/step** → SOL **64.8 tok/s**；
  depth 7680 = **15.282 GB/step** → SOL **62.8 tok/s**（扫 capacity 8192 时
  B_mean 15.298 → 62.8）。
- 实测 %SOL（模型速度）：eager 17.8–18.5%；graph-512-1024 48.5%；
  graph-512-8192 36.2%；graph-8192 35.9%。
- FP8 KV 理论值（未实测）：depth 512 14.787 GB → 64.9 tok/s；depth 7680
  15.021 GB → 63.9 tok/s。M3 只实测 BF16 KV。

## 8. 诊断：为什么 eager 只有 11.5 tok/s

86.4 ms/step 里，GPU kernel 累计只有 30.2 ms，其余约 56 ms 是 CPU 提交和间隙；
每步 7242 个 kernel，93.9% 小于 5 µs。也就是说 **eager 完全被 launch 开销支配**，
逻辑带宽只有 171 GB/s（17.8% SOL）。这与 M2 的 11.6 tok/s 诊断值一致，M3 的
统一计时入口只是把它固定成 baseline。

graph 之后 kernel 累计 29.7 ms 仍是 1.93× 于 SOL（15.4 ms 权重 + KV）。其中
NVFP4 GEMM 20.7 ms 对应 14.399 GB 的 packed weight + scale，即 **696 GB/s，
72.5% 峰值带宽**；剩余 9.0 ms 是 elementwise / copy / quantize / norm 等。

## 9. Phase 2 的优化顺序（按实测时间占比）

1. **NVFP4 GEMV**：20.7 ms/step（graph-512 的 69.5%、graph-8192 的 49.9%）。
   当前 72.5% 峰值带宽；换 swap-AB + 原生 `mxf4nvf4`、去掉每层的
   `1/input_scale` 与 `alpha` 逐次计算（实测 `reciprocal` 401 次/step 共 0.35 ms），
   目标 80–85% 带宽即可省 2–4 ms/step。
2. **消掉 8K 的 GQA 展开**：`repeat_kv` 物化 K/V 9.4 ms/step + attention matmul
   4.3 ms/step。自写 attention 直接对 4 个 KV head 做 GEMV 应该能拿回大部分。
3. **norm + activation quant 融合**：`quantize_with_block_size` 401 次（0.52 ms）
   和 209 次 RMSNorm reduce（0.32 ms）都在等权重之前串行执行。
4. **GDN decode**：`direct_copy`/`elementwise` 里 192+706 次小 kernel 的大头来自
   48 层 GDN 的 conv/recurrent/out_proj 组合。
5. ShapeStableLinear 的 M=128 padding：96 次/step 共 0.40 ms，占 1.7%，优先级低。
6. 最后才考虑 megakernel / 权重预取：现在时间线里 **没有** 明显的层间空隙，
   graph 已经把提交开销消掉；先做 kernel 效率。

## 10. 已知限制

- P1（五个数据集各 20 条真实请求）和 P2（temp 1.0 / top-p 0.95 / top-k 20 /
  min-p 0）**未跑**，属于后续报告项，不阻塞 M3。
- `eager-original-io`（M2 的 pageable H2D）未单独测；四模式共用 pinned staging。
- 只测了 BF16 KV；FP8 只有理论值。
- K 分割和 `mm_fp4` 的 tactic 由 FlashInfer 选择，本文没有对比 `b12x`/`cudnn`
  backend；同一进程内同 shape 的路径是 bitwise 可复现的。
- nsys 2023.4 看不到 sm_120（报告里没有 CUDA kernel data），时间线改用
  `/opt/nvidia/nsight-systems-cli/2026.2.1`。
- graph 与 static-cap 的 free-running greedy trace 在长序列上会和 eager 分叉
  （见 §4 的 diagnostic）；这是数值灵敏度，不是实现差异，16 步的逐 step bitwise
  检查才是 gate。

## 11. 复现命令与结果目录

```bash
source scripts/m2_env.sh
make m3-fixtures      # 固定深度 prompt + trace + smoke（约 3 分钟）
make m3-check         # 211 个 gate，5 个场景（约 12 分钟）
make m3-bench         # 11 个场景 × 5 次 × 两套计时（约 95 分钟）
make m3-memory        # 分阶段显存 + 完整生命周期（约 10 分钟）
make m3-profile       # nsys 时间线（约 10 分钟）
make m3-sol           # 实际字节账 + docs/sol.md
make m3-attn-effect   # attention 长度效应 microbench
```

| 产物 | 路径 |
|---|---|
| fixtures + manifest | `bench/workloads/m3-p0/` |
| 正确性验收 | `results/m3/20261021T180905-m3-check/` |
| 性能矩阵 | `results/m3/20261021T163844-m3-bench/` |
| 分阶段显存 | `results/m3/20261021T181311-m3-memory/` |
| nsys 时间线 | `results/m3/20261021T182309-m3-profile/` |
| SOL 字节账 | `results/m3/<run>-sol/sol.json`、`docs/sol.md` |
| attention 长度效应 | `results/m3/20261021T162645/attn_length_effect.json` |
