# Phase 2 第 3 轮任务 L：GDN gated RMSNorm + out_proj activation quant 融合

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §4 第 2 行、§5、§6、§7；
  [decisions 2026-10-21「M2 chunked prefill 数值口径」](../decisions.md)
- session：`phase2-gatednorm`；worktree `.tmp/worktrees/r3-gatednorm`，
  branch `exp/phase2-gatednorm`
- 起始基线：集成 commit `89ff69fb6d2f59820f5964ce41d11448a9ecd1de`
- 候选 commit：`783b36a3b23e963ef1382d964aaa73120d61545d`
- job：`job-000043-19c1f1a0`（`succeeded`，223.9 s，`correctness=passed`）
- checkout：`models/Qwen3.8-27B-NVFP4-RTX5090`，
  revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- 对照 baseline：`job-000041-d2baf18e`（commit `89ff69f`，**511 步 × 5 次**正式协议）
- 结果：`results/experiments/job-000043-19c1f1a0/output/{op/op.json,check/check.json,bench/bench.json}`
- 上游依据：任务 H 的 norm+quant 融合（journal `2026-10-23-phase2-normquant.md`，在
  `explorer-normquant` worktree；本分支未合入该文件）§7.2 的 per-consumer 契约、R2、
  `_pending_quantized`、`_gemm` dispatch；第 2 轮 profile `job-000042-0c4fdf2d`

## 1. 结论摘要

48 个 GDN 层的 `Qwen3_5RMSNormGated.forward`（decode 时约 12 个 tiny kernel）**加**
out_proj 的 `quantize_with_block_size` 被合成 **48 次 launch**，语义 bitwise 不变。

| 指标 | baseline `job-000041` | 候选 `job-000043` | 变化 |
|---|---|---|---|
| `graph-8192` step p50 | 21.824 ms | **21.142 ms** | **−0.683 ms / −3.1%** |
| `graph-8192` 系统 tok/s | 45.77 | **47.23** | +3.2%（→ 75.3% SOL） |
| `graph-512-1024` step p50 | 20.829 ms | **20.131 ms** | **−0.698 ms / −3.4%** |
| `graph-512-1024` 系统 tok/s | 47.95 | **49.56** | +3.4% |
| peak allocated（8K / 512） | 15.293 / 14.822 GB | 15.294 / 14.822 GB | 不变 |

- op **63/63**：`real-layers` 288 个样本、**1,769,472 个 E2M1 code** 在 `out_bf16` /
  `a_fp4` / `a_sf` 上**全部 0 字节差**；`mean-tree` 60/60 逐位相同，而「线程内树合并」
  变体 25/60 不同（gate 有分辨力）。
- check **139/139**（`numeric-512` 115 + `graph-1024` 24）：`eager→static-exact`、
  `padded-ref→static-cap`、reset 的 bitwise gate 全部保持；接线计数
  `gated_norm.fused = 12288/960`、`unconsumed = 0`、唯一的回退原因是 `shape`（prefill）。
- **收益比预期大 3 倍**：派发时按 profile 逐 kernel 折算的净收益是 −0.2~−0.25 ms，
  实测 −0.683 ms。原因是那条折算漏掉了「48 次 `1/input_scale` 标量 kernel」与
  「12+1 个 tiny kernel 的整条依赖链」，见 §6。

## 2. 设计与数据流

```text
改造前（每层，decode）：
  core_attn_out[48,128] --norm.forward--> 12 个 tiny kernel（cast/pow/mean/add/rsqrt/
  mul/cast/bf16-mul/cast/silu/mul/cast）--> out[48,128] bf16
  out.reshape(1,6144) --out_proj--> 1/input_scale 标量 kernel + nvfp4_quantize + mm_fp4

改造后（每层）：
  x[48,128] bf16  +  z(gate)[48,128] bf16  +  weight[128] bf16  +  gs=fl32(1/input_scale)
    -> 1 个 Triton kernel（6 个 chunk × 8 行，一个 program）
         -> out_bf16[48,128]（= 模块 forward 的返回值）
         -> (a_fp4[1,3072], a_sf[128,384])  R2 per-consumer 量化
    -> out_proj 的 `_pending_quantized` 单次槽位 -> NVFP4Linear._gemm(..., m=1)
```

kernel（`runner/ops/gdn/gated_norm.py::_gated_norm_quant_kernel`）每个 chunk（8 行 =
64 个 16-block = 1024 个元素）依次做：

1. **每行 128 维的 squared-mean**，顺序逐位复刻 `at::native::mean_kernel_cuda`（见 §3.3）；
2. `variance = mul_rn(sum, factor)` 再 `add.rn(., eps)`、`rstd = rsqrt.approx.f32`；
3. `xn = x * rstd`（mul.rn）→ `bf16` → `* weight`（bf16 opmath）→ `bf16`（见 §3.1/§3.2）；
4. `y = normed_f32 * silu(gate_f32)`，`silu = div_rn(g, 1 + __nv_expf(-g))`；
5. `y → bf16` 写 `out_bf16`，再按 16-element block 求 amax，走任务 E 冻结的 **R2**：
   `sf_pre = gs * (amax * c6)`、`cvt.rn.satfinite.e4m3x2.f32`、`out_scale =
   rcp.approx.ftz(sff * rcp.approx.ftz(gs))`、`code = cvt.rn.satfinite.e2m1x2.f32(y*out_scale)`。

量化部分**原样复用** `runner/ops/nvfp4/norm_quant.py` 的 PTX 助手与 `_emit_consumer`
（连 128x4 swizzle 落点 `(b//4)*512 + (b%4)`、低 nibble 是偶数下标都一样），chunk 取
64 个 block 就是为了让 `_emit_consumer` 的 reshape/packing 一行不改 —— 两份 R2 实现
漂移是这类融合最容易出的静默错误。6144/16 = 384 个 block，是 4 的倍数，`a_sf` 只有
第 0 行有数据、padding 由一次性 `torch.zeros` 保持不变。

接线（`runner/model/gated_norm_patch.py`）：patch **类** `Qwen3_5RMSNormGated.forward`。
install 时遍历模型里的 `Qwen3_5GatedDeltaNet`，建立 `layer.norm -> layer.out_proj` 的
映射（以及 out_proj 的 `gs`），所以 norm 的 patched forward 能直接把量化结果放进
对应 out_proj 的 `_pending_quantized`。只覆盖 `(48,128)` 的 decode 形状；prefill
（`T*48` 行）回退原 forward 并计 `fallbacks.shape`。

**融合快路径必须走 `NVFP4Linear._gemm(..., m=1)`**（任务 F 的 per-shape decode backend
选型，`(5120,6144) -> cute-dsl`）；`_pending_quantized` 的消费点就在 `forward` 里，
所以走槽位天然共用同一条 dispatch。op 的 `wiring` case 直接断言
`decode_backend_stats()["selected"]["5120x6144"] == "cute-dsl"`，防止有人以后在
槽位分支里直接调 `_mm`（任务 H §7.2 的教训）。

## 3. 三个 bitwise 刀锋点（相对任务 H 的 `Qwen3_5RMSNorm`）

### 3.1 weight **直接乘**，不是 zero-centered 的 `1 + w`

`Qwen3_5RMSNormGated.forward` 是 `hidden_states = self.weight * hidden_states.to(input_dtype)`；
checkpoint（以及 loader 写回的）`norm.weight` 是 **BF16 [128]**，值围绕 1。任务 H 的
`Qwen3_5RMSNorm` 用的是 `1 + weight.float()`（f32）。照抄 H 的 convention 会把输出整体
放大到约 2 倍；这不是 ulp 级差异，而是**量级级**错误（op 的 `weight-plus-one` 负对照
nrmse = 2.229）。

### 3.2 **两次独立的 BF16 舍入**

```text
xn      = x32 * rstd                 # f32
xb      = xn.to(bf16)                # 舍入 1（cvt.rn.bf16.f32）
normed  = weight * xb                # bf16 opmath：f32 里乘、再舍入回 bf16（舍入 2）
out     = normed * silu(gate.f32)    # bf16 -> f32 提升，f32 乘法
out     = out.to(bf16)               # 舍入 3
```

`Qwen3_5RMSNorm` 只有最后一次 cast（`x32*(1+w)` 全在 f32 里算完再 cast 一次），
gated norm 有**三处**独立的 BF16 舍入，其中第 1、2 处之间的顺序（先 cast 再乘）不能
交换：若按 `weight_f32 * xn_f32` 再 cast 一次，量化输入的 bf16 值就不再相同，
block amax 与 code 都可能翻转。任务描述里把这条写成「先 cast 再乘」，与源码一致。

实现上 `xb.to(f32) * weight` 之后立刻 `.to(bf16)`，PTX 里就是
`mul.rn.f32` + `cvt.rn.bf16.f32`（本 kernel 的 PTX 里 `cvt.rn.bf16.f32` 共 96 个 =
每线程 2 处 × 48 元素），没有任何「把乘法和 cast 合成一次」的空间。

### 3.3 reduction 维是 **128（per v-head）**，launch 配置与五级 halving

`x32.pow(2).mean(-1, keepdim=True)` 的输入是 **[48,128] contiguous FP32**（不是 H 的
`[1,1,5120]`）。按 `ATen/native/cuda/Reduce.cuh::setReduceConfig` 逐条推导：

```text
num_outputs = 48, inputs_per_output = 128, 最快维 stride = 4B
reduction_on_fastest_striding_dimension = true
dim0 = 128 >= 128 且 num_reduce_dims == 1  ->  vectorize_input = true, dim0 = 128/4 = 32
set_block_dimension<float>(32, 48):
    dim0_pow2 = 32, dim1_pow2 = last_pow2(48) = 32
    block_width = min(32, 32) = 32; block_height = min(32, 512/32) = 16
    block_width = min(32, 512/16) = 32          -> block = (32, 16), 512 线程, grid.x = 3
input_mult = {1,0,0}, step_input = 32           # 每个 lane 一个 float4，只迭代一次
values_per_thread = div_up(128,32) = 4 < warp_split_threshold(256) -> 不分 warp
output_mult = {0,1}, step_output = 16           # 每个 warp 一行，3 个 CTA 覆盖 48 行
无 global reduce
```

所以每 lane 的 4 个累加器是 `sq[4l], sq[4l+1], sq[4l+2], sq[4l+3]`，
`ReduceOp::input_vectorized_thread_reduce_impl` 最后**按 0,1,2,3 顺序**合并：

```text
acc[l] = ((sq[4l] + sq[4l+1]) + sq[4l+2]) + sq[4l+3]
sum    = block_x_reduce 的 warp shfl_down：offset 16, 8, 4, 2, 1（五级「一半加一半」）
var    = mul_rn(sum, factor) + eps           # factor = fl32(48/6144) = 2^-7（精确）
```

Triton 里就是连续的 `tl.sum(tl.reshape(v, (2, n)), axis=0)`（n = 16,8,4,2,1）。
**不能**用一次 `tl.sum`：Triton 的 warp 归约是 xor 1,2,4,...,16（相邻先加），括号化
不同、FP32 结果差 ulp。平方用 inline PTX `mul.rn.f32` 钉住（防 fma 收缩），
`variance = mul.rn` 再 `add.rn` 也显式两步。

silu 也是刀锋：PyTorch `silu_kernel`（fp32 opmath）是 `x / (1 + exp(-x))`，
`exp` 走 `::expf -> __nv_expf`、除法是 `div.rn.f32`。Triton 的 `a / b` 会落到
`div.full.f32`，所以用 `libdevice.div_rn` + `libdevice.exp`（任务 I 的 conv kernel
已用同一对函数在约 2M 个 bf16 输出上验证过）。

### 3.4 结构对账（CPU 离线编译，不建 CUDA context）

`triton.compile + ASTSource(target=sm_120)` 读 PTX：

| 项 | 数量 | 说明 |
|---|---|---|
| `fma.rn.f32` | 48 | 与 48 个 `ex2.approx.ftz.f32` 一一对应，等于每线程元素数 → 只来自 `__nv_expf` 内部 |
| `mul.rn.f32` | 96 | 平方（inline asm）与 silu 前的乘法 |
| `add.rn.f32` | 48 | variance 的 `add.rn(., eps)`（inline asm） |
| `rsqrt.approx.f32` | 48 | 每行一次 |
| `rcp.approx.ftz.f32` | 49 | 每个 block 一次 out_scale + 1 个 hoist 的 gs 倒数 |
| `cvt.rn.satfinite.e4m3x2/e2m1x2.f32` | 48 / 48 | R2 的 SF 与 code |
| `cvt.rn.bf16.f32` | 96 | §3.2 的两处舍入 |
| `div.full.f32` | **0** | 除法是 `div.rn.ftz.f32`（`libdevice.div_rn`）；ftz 只影响 denormal |

## 4. op 的三层证据（`scripts/phase2_fuse_check.py`，5 个 case / 63 条 gate）

**第一层：算术序列**（`mean-tree`）——把「对 launch 配置的读法」和「Triton 实现」分开。
用 torch op 逐级复刻上面的 lane/tree，在 5 层 × 6 种输入（`gauss-0.5/2/8`、
`saturation`、`zeros`、真实 embedding 行）共 60 个样本上与 `torch.mean(-1)` **逐位相同
（0 mismatch）**；同一份代码把「线程内改成树合并 `(a+b)+(c+d)`」当变体，25/60 不同 ——
既证明 launch 读法正确，也证明这条 gate 对括号化有分辨力。

**第二层：48 个真实层逐字节**（`real-layers`）——48 个 GDN 层各自的
`linear_attn.norm.weight` 与 `out_proj.input_scale`（`gs` 范围 20.06 ~ 480.54），
每层 6 种输入，比较 `out_bf16`、`a_fp4`、`a_sf`（含 127 行 padding 必须为 0）：
**288/288 样本、1,769,472 个 E2M1 code、0 字节差**，worst 三个张量都是 0。
其中 240 个样本 `amax > 0`，48 个 `zeros` 覆盖 `amax == 0` 的 0 分支，
`saturation` 覆盖 `sf_pre` 冲 e4m3 satfinite。

**第三层：分辨力 + 接线**（`negative-controls` / `wiring` / `micro`）——
三种错实现显著超差，正确候选在同一份输入上 nrmse = 0：

| 错实现 | nrmse | 字节差 |
|---|---|---|
| `weight-plus-one`（用 H 的 `1+w`） | 2.229 | 11326 |
| `no-gate`（直接输出 normed） | 1.048 | 10907 |
| `reduce-6144`（variance 在整条 6144 上求平均） | 0.611 | 8005 |

`wiring` 用真实 `Qwen3_5GatedDeltaNet`（meta harness）+ 真实 CUDA norm/out_proj 权重走
`layer.norm(x, z) -> layer.out_proj(...)`：`out_bf16` 与 **out_proj 的 GEMV 输出**都与
未打 patch 的模块路径**逐字节相同**，`fused=1 / unconsumed=0 / fallbacks={}`，
且 `selected["5120x6144"] == "cute-dsl"`（没绕过 decode 选型）。

`micro`（同一计时边界，eager + graph replay 两个口径，`num_warps` 扫 2/4/8）：

| 形态 | 两步路径 | 融合 | 倍率 |
|---|---|---|---|
| eager（模块 norm + 1 次 quantize） | 0.074848 ms | **0.016768 ms** | 4.46× |
| graph replay | 0.016228 ms | **0.004704 ms** | 3.45× |
| graph 折算 48 层 | 0.779 ms | 0.226 ms | 省 **0.553 ms** |

`num_warps` 2/4/8（eager）= 0.018656 / 0.017120 / 0.015008 ms，接入固定 **4**（未细调）。

**模型里真的命中**（check / bench 的 `load.norm_quant.gated_norm.counters`）：

| 步骤 | `fused` | `unconsumed` | `fallbacks` |
|---|---|---|---|
| check `numeric-512` | 12288 | 0 | `{shape: 4800}` |
| check `graph-1024` | 960 | 0 | `{shape: 1152}` |
| bench `graph-8192`（32 步） | 192 | 0 | `{shape: 17280}` |
| bench `graph-512-1024` | 192 | 0 | `{shape: 1152}` |

`fused` 全是 48 的整数倍、`unconsumed = 0`、回退只有 prefill 的 `shape` —— 接线没有
静默漏掉任何一层（第 2 轮任务 H 的 job-000038 就是接线 bug 只在 eager 第 0 层才暴露）。

## 5. check / bench

check（`m3_check.py --only numeric-512,graph-1024 --steps 16`，139/139）：
`numeric-512` 的 `eager→static-exact`、`padded-ref→static-cap`、边界长度 1/127/128/
129/300 的逐位一致全部 PASS，`graph-1024` 的 static-cap→graph 逐位一致也 PASS。
四种执行方式的 decode 都走同一条融合路径（`fused` 计数在每种方式里都增长），
所以这些 bitwise gate 仍然可比。

bench（`--only graph-8192,graph-512-1024 --steps 32 --repeats 1`，1 次重复）：

| 场景 | step p50 / p95 | 模型 tok/s | 系统 tok/s | prefill s |
|---|---|---|---|---|
| `graph-8192` | 21.142 / 21.154 ms | 47.30 | **47.23** | 6.685 |
| `graph-512-1024` | 20.131 / 20.140 ms | 49.67 | **49.56** | 0.421 |

**协议差异必须标注**：候选是 32 步 × 1 次，对照的 `job-000041` 是 511 步 × 5 次的
正式协议（21.824 / 20.829 ms）。step p50 与 p95 只差 0.012/0.009 ms，跨协议比较的
方向性可信，但正式结论仍应在同协议（511×5）下复测 —— 这一步留给 Coordinator 的
组合集成 job。

显存：bench 的 `capture_peak_allocated` 与 baseline 完全一致（8K 15.294 vs 15.293 GB，
512-1024 都是 14.822 GB），prefill 峰值处的 `free_gb` 是 8K **0.342 GB**、512-1024
**1.202 GB**（baseline 的 memory 步骤是 0.340 / 1.191 GB）。新增 scratch 只有
**64512 B**（BF16 输出 12288 + `a_fp4` 3072 + `a_sf` 49152），在 capture 之前分配、
地址固定。**本轮没有跑完整的 `m3_memory.py` 生命周期**（load→prefill→capture→
reset→prefill→decode），所以「无 stale state / 无持续增长」这一条仍未验证。

## 6. micro → e2e：0.43 估算 vs 0.683 实测

派发时按 `job-000042`（graph-7680，8 步）的 per-kernel 均值折算：48 层的 gated norm
自身（pow 29.5 + mean 64.4 + rsqrt 31 + add 30.7 + mul 34.6 + bf16-mul 30.5 + silu 44.5
+ cast ~85 µs）加 out_proj 的 48 次 quantize（~76 µs）≈ **0.43 ms/步**，再按「48 个融合
kernel × 3.5–4 µs」扣 0.19 ms，净收益估 **−0.2 ~ −0.25 ms**。实测 **−0.683 ms**
（14.2 µs/层）。差 3 倍，账可以拆开：

| 项 | ms/步 | 依据 |
|---|---|---|
| graph micro 折算（48 × 11.524 µs） | 0.553 | `graph_two_step 16.228 - graph_fused 4.704 µs/call` |
| 48 次 `1.0 / input_scale` 标量 kernel | 0.076 | profile 里 `reciprocal_kernel_cuda` 129 次/步 × 1.58 µs，out_proj 占 48 次；**micro 的 baseline 用预算好的 gs 直接调 `nvfp4_quantize`，不含这笔** |
| 48 个 graph node 消失后的依赖间隙 | 0.015 | profile：kernel 26.49 ms vs 墙钟 28.35 ms，平均 0.31 µs/间隙 |
| 未归因（L2 / 依赖链变短） | 0.039 | 残差 |
| **合计** | **0.683** | 与实测一致 |

两条教训：

1. **profile 的 per-kernel 均值会把「同一 kernel 名下的不同来源」摊薄**。`job-000042`
   里同一个名字同时覆盖好几条路径：`bfloat16_copy_kernel_cuda` 147 次/步、
   `AUnaryFunctor<Mul>` 131 次/步、`elementwise_kernel<128,2,BinaryFunctor<Mul>>` 114 次/步、
   `vectorized_elementwise<4,BinaryFunctor<Mul>>` 49 次/步、`unrolled_elementwise/direct_copy`
   258 次/步 —— 每个名字里既有 81 个 norm（gated 48 + q/k_norm 32 + final 1）的，也有
   attention/MLP/conv 的。用「实例数 × 均值」再按比例分给 gated norm，会把 48 层里
   那 12 个 kernel 的真实成本摊薄到别处，所以 0.43 ms 是**下界**。要更准只能按
   layer/op 打 NVTX 区间再统计（本轮没做）。
2. **micro 的两步 baseline 必须与真实 `forward` 同口径**。真实 `NVFP4Linear.forward`
   除了 `nvfp4_quantize` 还每次算一次 `1/input_scale`（1 个 1-element kernel）；
   micro 把 gs 预计算好，就漏掉了 48 × 1.58 µs。以后 micro 的 baseline 应当直接调
   `module(x)`，而不是「等价的手写两步」。

## 7. 两个 env flag 的关系（集成时要注意）

`gated_norm_patch` 挂在 `norm_quant_patch.install_from_env` 的末尾（loader 只认识那一个
入口），但两者**互相独立**：

| flag | 关掉的东西 | 影响的层 |
|---|---|---|
| `NORM_QUANT_FUSED=0` | 任务 H 的 decoder-layer norm+quant 融合 | 128 个 decoder norm / 272 个消费者 |
| `GATED_NORM_FUSED=0` | 任务 L 的 gated norm + out_proj quant 融合 | 48 个 GDN 层 / 48 个 out_proj |

- 两者 patch 的层次不同（`Qwen3_5DecoderLayer.forward` vs
  `Qwen3_5RMSNormGated.forward`），任何一条 install 失败只回退自己；
  `NORM_QUANT_FUSED=0` **不会**关掉 gated norm。做 A/B 时必须两个都置 0，
  否则「关掉 H 的融合」里仍然含着 L 的收益（−0.68 ms）。
- 产物里两边的计数分别是 `load.norm_quant.counters`（H）与
  `load.norm_quant.gated_norm.counters`（L）。

## 8. 限制与回退

- **只走 decode**：`hidden_states.shape == (48,128)`。prefill / chunked prefill 回退原
  forward，`fallbacks.shape` 就是它的计数（`graph-8192` 里 17280 = 48 层 × 360 个
  prefill chunk）。prefill 的 gated norm 没有优化。
- **shape 硬编码**：`48 × 128 = 6144`、chunk 6 × 8 行、384 个 block。config 里
  `linear_num_value_heads=48`、`linear_value_head_dim=128` 一变，install 直接抛错
  （`install_from_env` 只记录不抛，模型回退原路径）。
- **单消费者**：kernel 只产 1 份 `(a_fp4, a_sf)`（out_proj）。如果以后有别的层也消费
  gated norm 的输出，需要改成 H 那样的多消费者契约。
- **silu 只支持 `activation == "silu"`**；install 时校验，别的 activation 拒绝安装。
- **ftz**：`libdevice.div_rn` 在 Triton 里落成 `div.rn.ftz.f32`，PyTorch 是
  `div.rn.f32`。只在 bf16 denormal 的 gate 值上才可能有差异（真实分布里不出现），
  这是唯一一处「语义上可能不同、但实际不可达」的边界。
- **`num_warps` 未细调**：扫到 8 更快（eager 0.015008 vs 0.017120 ms），但 graph 口径的
  per-call 只有 4.7 µs，先固定 4；若要再抠，按 graph 复测一次再改默认值。
- **8K 显存生命周期未重测**（见 §5）：只看了 bench 的 peak 与 prefill 时的 free，
  `m3_memory.py` 的分阶段生命周期检查留给组合集成 job。
- 回退是**整层**的：`try_run` 里任何异常都回原 forward 并计 `kernel-error`，
  不会把 decode 弄崩。

## 9. 复现

```bash
# 队列（GPU 主机，单 job；commit 783b36a）
.venv/bin/python scripts/experiment.py submit --request .tmp/req-gatednorm.json
.venv/bin/python scripts/experiment.py result job-000043-19c1f1a0
# 产物
results/experiments/job-000043-19c1f1a0/output/op/op.json      # 63 gate
results/experiments/job-000043-19c1f1a0/output/check/check.json # 139 gate + gated_norm 计数
results/experiments/job-000043-19c1f1a0/output/bench/bench.json # 2 场景 x 32 步
```

请求的三步：op（`phase2_fuse_check.py`，约 100 s，含首次编译）→ check
（`m3_check.py --only numeric-512,graph-1024 --steps 16`）→ bench
（`bench/e2e/decode.py --only graph-8192,graph-512-1024 --steps 32 --repeats 1`）。

CPU-only 的接线自测：`python -m pytest tests/model/test_gated_norm_patch.py -q`
（5 个用例，**不建 CUDA context**；`gated_norm_patch` 因此把
`runner.ops.nvfp4.linear` 的 import 放在 `install()` 里 —— flashinfer 的 import 会
探测 CUDA capability）。

## 10. 下一步建议

1. **同协议复测**：511 步 × 5 次确认 −0.68 ms（本 job 是 32×1）。按两个 depth 一致的
   0.68/0.70 ms，与「gated norm 是每层固定开销、与 context 长度无关」的预期吻合。
2. **`num_warps=8`**：graph 口径复测一次，若稳定更快就改默认值（量级可能只有
   −0.02 ~ −0.05 ms，属于细调）。
3. **gated norm 已经压到 4.7 µs/次（graph）**，48 次 ≈ 0.23 ms；这条边界继续压的绝对
   空间 <0.2 ms。剩余大头（GEMV 16.5 ms、decode attention ~1.1 ms、剩余 elementwise）
   应按第 2 轮结论继续在 attention / elementwise 上找。
4. **micro 的 baseline 口径**（§6 教训 2）应当固化成约定：融合类的 microbench 基线
   直接调模块 `forward`，不要手写「等价两步」，否则会系统性低估收益。
5. `mean-tree` 这类「launch 配置读法」的 case 值得复制到下一个 reduction 类融合
   （例如 attention softmax 的 max/sum），它是唯一能在不看结果的情况下判定
   「括号化读对了没有」的 gate。
