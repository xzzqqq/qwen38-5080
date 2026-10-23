# Phase 2 任务 G：ShapeStableLinear 的 M=1 decode 路径

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §3 任务（profile 驱动）、§5 验证要求、§6 反馈阶梯；
  [decisions 2026-10-21「M2 chunked prefill 数值口径与 shape-stable a/b」](../decisions.md)
- session：`phase2-stable-linear`；worktree `.tmp/worktrees/explorer-stable-linear`，
  branch `exp/phase2-stable-linear`
- 起始基线：集成 commit `9e35873a01c0769fbed302a9f0b32a222e3ffcd1`
- 候选 commit：`ff0c87bee0005235cbd8f2a1d63567f85255db26`
  （`bbcbcb8ace9245baed0029a5e490b4be6452fe91` 实现 + gate 规格修正；
  kernel 本身在 `bbcbcb8` 后未再改动）
- 成功 job：`job-000033-1bfa42c1`（219.4 s，correctness `passed`）；
  失败 job：`job-000031-1356864c`（4.98 s，`gate_failed`，只挂在 gate 阈值上，见 §4）
- checkpoint：`models/Qwen3.8-27B-NVFP4-RTX5090`，revision
  `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`
- baseline（同协议）：`job-000029-16bb35a9`（commit `9e35873`，32 步 × 1 次）
- 结果路径：`results/experiments/job-000033-1bfa42c1/{output/op/op.json,output/check/,output/bench/bench.json}`

## 1. 结论摘要

profile 驱动的一个纯浪费被消掉：decode 的 96 个 `in_proj_a/b` 每步把 **1 行有效
数据 pad 到 128 行**再跑 wmma GEMM。候选在 `flat rows == 1` 时改走不 padding 的
Triton BF16 matvec，M>1（prefill）路径一行未改。

| 指标 | baseline `job-000029` | 候选 `job-000033` | 变化 |
|---|---|---|---|
| `graph-8192` step p50 | 27.781 ms | **27.052 ms** | **−0.729 ms / −2.6%** |
| `graph-8192` 系统 tok/s | 35.95 | **36.92** | **+2.7%** |
| `graph-512-1024` step p50 | 26.779 ms | **26.079 ms** | **−0.700 ms / −2.6%** |
| `graph-512-1024` 系统 tok/s | 37.29 | **38.30** | **+2.7%** |
| peak allocated（8K / 512） | — | 15.288 / 14.818 GB | 与 baseline 同档 |
| free 最低（8K / 512） | 0.329 GB | 0.329 / 1.191 GB | 不变 |

- 两个 depth 省的时间一致（0.729 / 0.700 ms），与「a/b 是每层固定开销、与
  context 长度无关」一致。
- op gate **37/37**、check **139/139**（`numeric-512` 115 + `graph-1024` 24）。
  `eager→static-exact`、`padded-ref→static-cap`、`padded-fill3 vs fill0`、reset 的
  **全部 bitwise gate 仍通过**，即「decode 的四个执行方式都走同一条新路径」这一
  假设成立，没有把任何 bitwise gate 降级。
- microbench（真实权重、graph replay）：**7.873 → 1.617 µs/次（4.87×）**，
  96 次/步预计省 **0.601 ms**；实测 e2e 省 **0.729 ms**（见 §5 的解释）。

## 2. 设计与数据流

```text
M > 1（prefill，未改）:
  x [B,T,K] -> reshape [M,K] -> 每 block_m=128 行 pad(zero) -> F.linear -> 截断
           -> reshape [B,T,N]

M == 1（decode，新增）:
  x [B,T,K] -> reshape [1,K] -> _bf16_matvec_kernel -> [1,N] -> reshape [B,T,N]
```

kernel（`runner/ops/bf16/stable_linear.py::_bf16_matvec_kernel`）：

- **一个 program 算一个输出行**（grid = N = 48），`w_row = w_ptr + n*K`；
- K 方向按 `BLOCK_K=512` 分块，每块 `acc(BLOCK_K, f32) += x[块] * w[块]`，
  最后一次 `tl.sum` 树归约；bias 用 `HAS_BIAS: tl.constexpr` 编译期选择；
- BF16 载入、FP32 累加；**没有原子操作、没有 split-K、reduction 顺序固定**，
  所以同一输入逐位可复现，CUDA graph capture/replay 安全（生产路径是 graph）；
- 预期瓶颈：权重 480 KB/次、grid 只有 48 个 CTA → latency-bound，不是带宽 bound。

参数扫描（`CONFIGS`，同一份真实权重 + graph replay）：`block_k=512,warps=4`
**1.617 µs**、`1024,4` 1.860 µs、`2048,4` 1.840 µs → 默认取 **512/4**
（`DECODE_BLOCK_K` / `DECODE_NUM_WARPS`）。

对照（同一计时边界，每调用）：`torch.matmul(x, w.t())` **2.969 µs**、
`F.linear(M=1)` **2.966 µs**、原 padded 路径 **7.873 µs**（其中 M=128 GEMM 5.10 µs、
pad 的 `fill`+`torch.cat` 2.77 µs）。选 Triton 的实测理由：比 cuBLAS 的两种 M=1
形态快 1.8×，且不依赖 cuBLAS 对 M=1 的启发式选择、reduction 顺序固定。

## 3. 为什么 prefill 的 M=128 padding 不能动、decode 可以换

decisions 2026-10-21 的口径是：把 96 个 `in_proj_a/b` 固定成 `block_m=128` 分块、
尾块补零，**目的是让同一行 token 在单段 prefill 与分段 prefill 里走同一个 M 的
GEMM**，从而让 chunked prefill 的数值 reference 可以只用边界对齐的 full 前缀
（`full 256 vs 128+128`）做 bitwise 对拍。这条约束只要求「**同一 M 的调用逐位
一致**」，不要求「M=1 与 M=128 的输出逐位一致」。

因此：

- **M=1 的单 token 前向不参与那个对齐**：它只出现在 decode 和 1-token prefill，
  两条路径都只有这一个 token，没有「同一行 token 换 M」的对照；
- 换 reduction 形状带来的差异只落在 decode 的 a/b 上，量级是 bf16 输出的
  1 ulp（§4 实测），而且**不改变任何 prefill 语义**；
- 反过来说，如果为了 decode 去改 M>1 的分块（例如 decode 也走 `block_m=1` 的
  分支而让 M=129 的最后一块变成 1 行），才会破坏 chunked prefill 的口径 ——
  本次没有这样做。

**M>1 逐位不变的证据**：op gate `padded-path-bitwise-unchanged` 9/9 —— 96 个真实
module × M ∈ {2,17,127,128,129,300,512}（`[1,M,5120]`）与 `[2,150,5120]`，候选输出
与逐行照抄旧实现的 `PaddedReference` **全部 `torch.equal`**。`git diff` 里 M>1 的
循环只多了 `_STATS["padded_calls"] += 1` 一行纯计数，不参与计算。

## 4. gate 口径修正（本轮唯一一处阈值变更，按时间顺序）

**先冻结、后看结果的部分**（`bbcbcb8`，job-000031 之前写死）：数值判据沿用
decisions 里 GDN 的绝对容差 `nrmse ≤ 1e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 5e-3`，
candidate 与「逐行照抄 `9e35873` 的 `PaddedReference`」对拍，`M>1` 用 bitwise。

**job-000031 的结果（4.98 s，`gate_failed`）**：padded bitwise 9/9、decode 接线
8/8（`decode_calls=96`、M=128 时 0、无回退）都通过，只挂在数值 gate 上
（module 级 4 条 + microbench 配置级 3 条，以及因此级联的
`coverage:parameter-sweep`）：

| 指标（worst of 48） | 实测 | 原阈值 | 判定 |
|---|---|---|---|
| `nrmse` | 3.77e-3 ~ 4.29e-3 | ≤ 1e-3 | FAIL |
| `max_rel` | 6.58e-3 ~ 7.58e-3 | ≤ 5e-3 | FAIL |
| `cosine` | 0.9999912 ~ 0.9999945 | ≥ 0.99999 | PASS（余量 1.7e-6） |
| **padded reference 自身到 fp64 的 nrmse** | **2.93e-3 ~ 3.44e-3** | — | — |
| 候选到 fp64 的 nrmse | **2.13e-3 ~ 2.21e-3** | — | 比 reference 更准 |

**判定：规格错，不是候选错。** `nrmse ≤ 1e-3` 比 **reference 自己到 fp64 的距离**
（3.3e-3）还紧 —— 单是 bf16 输出舍入就超过这条线，任何实现都过不了；这与本轮任务 A
发现「冻结容差连 baseline 自己都过不了」是同一类错误。`max_abs` = 0.015625 = 2^-6，
正是 absmax 元素（~2）的 1 ulp；`max_rel` 的分母是 absmax，所以它必然在 1 ulp 量级，
5e-3 的绝对阈值在这里不可判定。

**修正（`ff0c87b`，重投 job-000033）**，判据改成两条可判定的相对判据：

- `nrmse ≤ 5e-3`（抓 layout/scale/reduction 错到完全不同的量级）；
- `cand_vs_fp64_nrmse ≤ ref_vs_fp64_nrmse × 1.05`（**候选到 fp64 的距离不得比
  padded 路径差 5% 以上** —— 这才是「没有变差」的判据）；
- `cosine`、`max_rel` 降为 **diagnostic**（值仍写进产物，附「1 ulp 级刀锋值」说明）；
- `padded-path-bitwise-unchanged`、`decode-patch-coverage`、finite、`coverage:*`
  全部保持原样（没有把任何 bitwise gate 降级）。

job-000033 实测 ratio：a/0.5 **0.672**、b/0.5 **0.620**、a/0.05 **0.707**、
b/0.05 **0.750**（cand 0.00213~0.00221 vs ref 0.00293~0.00344）。这个阈值调整
发生在看到 job-000031 的产物之后、由 Coordinator 明确授权，且**只改了 gate 与
默认参数**，kernel 未动；理由与前后数字都写进了 `op.json` 的 case config / gate
detail（`why` 字段）与脚本 docstring。

## 5. 三段数字

### op（`bench/micro/stable_linear_decode.py`，4.6 s，37/37）

真实 checkpoint 的全部 96 个 `in_proj_a/b`（`[48, 5120]`，manifest 与派发消息里的
「a=N16」不符，以 manifest 为准）：

| 指标（96 module，graph replay） | baseline（padded） | 候选（matvec） | 变化 |
|---|---|---|---|
| warm 每调用 | 7.873 µs | **1.617 µs** | 4.87×，省 6.26 µs |
| cold（96×2 份副本，> 64 MiB L2） | 9.142 µs | **2.515 µs** | 3.63×，省 6.63 µs |
| 96 次/步估计（warm / cold） | 0.756 / 0.878 ms | 0.155 / 0.241 ms | 省 0.601 / 0.636 ms |

L2 容量 64 MiB（`L2_cache_size=67108864`）；warm 组 16 个 module = 7.9 MB，
必然 L2-resident，所以另做了 cold 组。**decisive 的是 e2e**，见下。

### check（`m3_check.py --only numeric-512,graph-1024 --steps 16`，150.9 s，139/139）

- `numeric-512` 115/115：`eager→static-exact`（logits/hidden/kv.keys/kv.values/
  conv/recurrent 6 项）、`padded-ref→static-cap`（6 项）、`padded-fill3 vs fill0`
  （6 项）、3 组 reset（各 6 项）**全部 PASS**；23 条 diagnostic 差异是 M3 已记录的
  free-running / capacity 数值分叉，不是 gate。
- `graph-1024` 24/24（1 条 diagnostic：逐步 argmax 与冻结 trace，M3 已知）。
- 结论：把 decode 的 a/b 从「M=128 GEMM 的 reduction」换成「M=1 matvec 的
  reduction」**没有破坏** eager→static-exact / padded-ref→static-cap 的 bitwise ——
  因为四种执行方式的 decode 都走同一条新路径，比较两侧用的是同一个 kernel。

### bench（`graph-8192,graph-512-1024 --steps 32 --repeats 1`，63.5 s）

| 场景 | step p50 | 模型 tok/s | 系统 tok/s | prefill s | peak alloc GB | free 最低 GB |
|---|---|---|---|---|---|---|
| `graph-8192` | **27.052**（−0.729） | 36.962 | **36.915**（+2.7%） | 6.562 | 15.288 | 0.329 |
| `graph-512-1024` | **26.079**（−0.700） | 38.344 | **38.296**（+2.7%） | 0.437 | 14.818 | 1.191 |

显存不变（与 job-000029 的 0.329 / 1.191 一致）。

### microbench → e2e 的差

microbench 预测省 0.601 ms（warm）/ 0.636 ms（cold），实测 **0.700~0.729 ms**，
换算成每调用是 **7.3~7.6 µs**（vs 6.26~6.63 µs）。差的约 0.10 ms/步没有单独归因，
最可能的两项都与「microbench 里 16 个 module 复用同一块 1.3 MB pad buffer」有关：
真实模型里 96 次调用各有自己的 pad buffer（graph pool 里不同地址），会被 14 GB
的 NVFP4 权重流冲出 L2，所以 `fill`+`torch.cat` 的实际成本高于 microbench 的
2.77 µs；另外每步少掉 192 个 tiny kernel，也就少掉它们的启动/间隙
（profile 里 6013 kernel/步、26.49 ms kernel 时间 vs 28.35 ms 墙钟，平均 0.31 µs/间隙）。

### profile 归因修正（对后续轮次重要）

集成后 profile（`job-000028` 的 `graph-7680/kern_sum.csv`）的 bf16 GEMM 是三类、
**每步合计 128 次 / 1.529 ms**：`32x32_128x2_tn` **96 次/步**（4.14 µs）= a/b；
`32x32_128x2_nn` 16 次/步（50.7 µs）+ `32x32_64x1_tn` 16 次/步（20 µs）=
16 个 full-attention 层的 QK/PV bmm，**不属于本任务**。所以 a/b 的真实成本是
约 0.73 ms/步（GEMM 0.48 + pad 0.25），不是 1.529 ms；本任务拿掉的是这 0.73 ms。

## 6. 限制与回退

- 开关：`SHAPE_STABLE_DECODE=0` 关闭 decode 分支（进程内第一次检查时解析），
  便于集成时做 A/B。
- 安全回退（都计数在 `decode_stats()` 的 `fallback_reasons` 里）：非 CUDA、
  权重与输入不同 device、非 bf16、`flat.shape[1] != K`、`N > 1024`、非 contiguous、
  bias dtype/shape 不符、缺 triton、kernel 首次 launch 抛错（警告一次后永久回退）。
  回退目标就是原来的 padded 路径，输出仍是正确值。
- 本分支只对 `flat rows == 1` 生效；1-token prefill 也会走它（语义相同、且
  full vs chunked 的边界对齐不受影响，见 §3）。
- `op` 的 `padded-path-bitwise-unchanged` 用的是 3D 形态（模型只发 3D）；2D M>1
  没有单列，但 M>1 代码逐位未改，`tests/model/test_stable_linear.py` 有 2D↔3D
  逐位用例（CUDA 用例，本次未在本机跑）。
- microbench 的绝对值受 L2 影响（warm 组必然 resident），报告里已单列 cold 组；
  真正的性能结论以 e2e 为准。

## 7. 复现

```bash
# op gate + microbench（4.6 s，不加载整模型，只用真实 a/b 权重）
.venv/bin/python bench/micro/stable_linear_decode.py --out results/experiments/<job>/output/op

# 数值/执行方式 gate（必须跑 numeric-512：它含 eager→static-exact 的 bitwise）
.venv/bin/python scripts/m3_check.py --only numeric-512,graph-1024 --steps 16 \
    --out results/experiments/<job>/output/check

# 端到端（对照 job-000029 的同协议数字）
.venv/bin/python bench/e2e/decode.py --only graph-8192,graph-512-1024 \
    --steps 32 --repeats 1 --out results/experiments/<job>/output/bench
```

## 8. 下一轮这一块还剩什么

- a/b 自身已经只剩 **0.155 ms/步**（96 × 1.617 µs，全是 DRAM/L2 读 480 KB + launch），
  继续优化单项的空间 <0.1 ms，不值得再投。
- 若要再压，只能改**调用结构**而不是 kernel：`in_proj_a` 与 `in_proj_b` 吃的是同一个
  `hidden_states`，把两者合成一个 grid=96（或一次 kernel 出 96 个输出）可以省掉
  一半 launch 与一份 x 读取 —— 约 0.05~0.08 ms/步，需要同时改模型接线与
  ShapeStableLinear 的接口，建议只在集成后仍有 0.1 ms 级缺口时再做。
- 同一份 profile 里更大的 bf16 GEMM 是 full-attention 的 QK/PV bmm
  （16 + 16 次/步、约 1.13 ms/步），那才是下一步该看的；本任务不动它。
