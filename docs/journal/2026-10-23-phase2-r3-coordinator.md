# Phase 2 第 3 轮 Coordinator 日志

- 日期：2026-10-23
- 起点：第 2 轮集成 commit **`89ff69fb6d2f59820f5964ce41d11448a9ecd1de`**
  （5 个 graph 场景 511×5：20.83 / 21.84 / 21.82 ms，系统 47.95 / 45.74 / 45.77 tok/s，
  72.9–74.0% SOL；163/163 gate；8K free 0.340 GB）
- 目标：**FP8 KV 的 80% SOL = 19.57 ms**（B_mean ≈ 15.02 GB → SOL 63.9 tok/s）。
  当前 21.82 ms，**差 2.25 ms（10.3%）**。
- 上一轮记录：[r2 coordinator](2026-10-23-phase2-r2-coordinator.md)

## 本轮依据（第 2 轮后的时间结构）

| 项 | 估算 | 依据 |
|---|---|---|
| NVFP4 GEMV（cute-dsl） | 16.55 ms（76%） | job-000035/037 |
| decode attention（bmm+softmax+PV） | ≈1.1–1.4 ms | profile QK 50.7 µs + PV 20 µs × 16 层 |
| fused norm+quant | 0.41 ms | job-000040 |
| GDN（recurrent 0.26 + conv 0.05） | 0.31 ms | job-000017/034 |
| ShapeStableLinear a/b | 0.16 ms | job-000033 |
| 其余 elementwise/reshape/copy | ≈2 ms | profile（4000+ 个 <5 µs kernel） |
| graph 间隙 | ≈1.3 ms | profile wall − kernel |

计划（第 2 轮 journal 已列）：FP8 KV（验收前提）→ 融合 decode attention kernel →
gated RMSNorm → 剩余 elementwise 融合。

## 时间线

| 时间 | task/session | 动作 | 理由与结果 ID |
|---|---|---|---|
| 14:40 | infra | `2988b24` 注册 `phase2_fp8kv_check.py` / `phase2_fuse_check.py`，确认队列空闲后重启 serve（pid 1414341） | 第 3 轮两个新入口；重启前显式检查 `running/queued` 为空与 GPU 无计算进程 |
| 14:41 | coordinator | 提交集成后 profile `job-000042-0c4fdf2d` | 第 2 轮后时间分布变了（GEMV 换 backend、quant/norm 融合、conv 融合），第 4 项（elementwise）需要新数据才能定目标 |
| 14:42 | J/K/L | 派发 3 个 Explorer | worktree 从 `89ff69f` 切出；F（backend）、G、I、H 的成果已在基线上 |

## 派发

| 任务 | session | worktree / branch | 脚本入口 | 目标 |
|---|---|---|---|---|
| J FP8 KV | `phase2-fp8kv` | `.tmp/worktrees/r3-fp8` / `exp/phase2-fp8kv` | `scripts/phase2_fp8kv_check.py` | 验收配置；KV 0.54→0.27 GB/step，−0.2~0.3 ms + 显存 |
| K 融合 decode attention | `phase2-attnfuse` | `.tmp/worktrees/r3-attnfuse` / `exp/phase2-attnfuse` | `scripts/phase2_attn_check.py`（扩展） | −0.6 ~ −0.8 ms |
| L gated RMSNorm + out_proj quant | `phase2-gatednorm` | `.tmp/worktrees/r3-gatednorm` / `exp/phase2-gatednorm` | `scripts/phase2_fuse_check.py` | −0.1 ~ −0.2 ms |
| M 剩余 elementwise 融合 | 待 profile | — | — | 待定（预计 −0.5 ~ −1.0 ms） |

## Coordinator 冻结的 FP8 方案（J/K 共用，J 可用 op 证据挑战）

- dtype：`torch.float8_e4m3fn`，K/V 都存 FP8；
- scale：**per (layer, kv_head, token) fp32 标量**，K/V 各一套，布局 `[capacity, kv_heads]`；
  `s = amax(|x_head|)/448`（amax 在该 token 的 256 个 dim 上取），
  `q = clamp(x/s, -448, 448).to(fp8_e4m3fn)`，反量化 `x = q.float() * s`；
- 有效长度外写 0（scale 也写 0），mask 语义不变；
- 参考与 gate 分层：① 量化/反量化 bitwise；② FP8-KV attention vs「同一份量化后 KV 的 fp64
  attention」在冻结容差内；③ FP8 vs BF16 只作 diagnostic（plan §5 不要求两种格式
  greedy trace 相同）。
- 已知需要一起改的 gate：`scripts/m3_check.py` 的 `cache_gate` 现在硬编码
  `kv_dtypes == ["torch.bfloat16"]`，必须改成按配置声明的格式判定（BF16 仍按 bf16），
  并在 journal 里写清「格式由 plan §7 指定，不是为接纳候选而放宽」。

## 本轮边界

- 每个实现任务一个方向、最多 3 组粗粒度参数、一次正确性修复后重测。
- J/K/L 都必须带硬 bitwise 或独立 reference；J 额外要求 8K 显存检查。
- K 的 kernel 接口要预留 FP8 scale 参数（J 可能同期落地），合并时由 Coordinator 统一。
- 组合 gate + 三个 graph 场景正式测量 + 8K 显存由 Coordinator 做；FP8 落地后要按
  **FP8 的 B_mean 重算 %SOL**，并与 BF16 分开报告。

## 集成后 profile（job-000042-0c4fdf2d，graph-7680 8 步）

`21.20 ms` kernel 时间 / **3212 kernel 每步**（第 2 轮前是 6013）：

| ms/step | 次数 | kernel |
|---|---|---|
| **16.570** | 401 | `kernel_cutlass_kernel_flashinfergemmkernelssm12x_cuteindependent_small`（F 的 cute-dsl） |
| **1.128** | 32 | `cutlass::Kernel2`（16 层 QK + 16 层 PV bmm） |
| 0.991 | 1347 | `vectorized_elementwise` |
| 0.509 | 128 | `_norm_quant_kernel`（H 的融合） |
| 0.494 | 354 | `unrolled_elementwise`（`StoreWithCast`，dtype cast） |
| 0.488 | 409 | `elementwise_kernel` |
| 0.264 | 48 | `_gdn_decode_kernel` |
| **0.204** | **129** | `quantize_with_block_size` ← 未融合的消费者 |
| 0.193 | 96 | `_bf16_matvec_kernel`（G） |
| 0.135 | 82 | `at::native::`（截断） |
| 0.114 | 82 | `reduce_kernel`（剩余 norm：gated norm 等） |
| 0.050 | 32 | `index_elementwise` |
| 0.048 | 48 | `_gdn_conv_decode_kernel`（I） |

elementwise 2.15 ms 的细分（按 functor 聚合）：

| ms/step | 次数 | 归因 |
|---|---|---|
| 0.581 | 501 | dtype cast（`StoreWithCast` + `bfloat16_copy`） |
| 0.438 | 502 | `MulFunctor<float>`（sigmoid·attn、silu·mul、scale） |
| 0.197 | 274 | 残差 add |
| 0.211 | 243 | norm 的 `pow` + `MeanOps` + `rsqrt`（剩余 81 个 norm） |
| 0.210 | 176 | `silu` / `sigmoid` / `softplus` |
| 0.168 | 133 | `direct_copy` |
| **0.112** | **129** | `reciprocal`（每个未融合消费者一次 `1/input_scale`） |
| 0.050 | 32 | `index_elementwise` |

**401 − 272 = 129 个未融合消费者**：`down_proj` 64、`out_proj` 48、`o_proj` 16、
`lm_head` 1。它们对应的 `quantize`(0.204) + `reciprocal`(0.112) + 相关 cast/mul ≈ **0.4–0.5 ms**。
其中 `down_proj` 的输入是 `silu(gate)*up`（plan §4 的融合项）、`o_proj` 的输入是
`attn_out*sigmoid(gate)`、`lm_head` 的输入是 final norm —— 都可以在**生产者侧**融合。

## 派发（第 4 项）

| 任务 | session | worktree / branch | 脚本入口 | 目标 |
|---|---|---|---|---|
| M 剩余 129 个消费者的 quant 融合（down_proj 的 silu·mul、o_proj、lm_head） | `phase2-fuse` | `.tmp/worktrees/r3-fuse` / `exp/phase2-fuse` | `scripts/phase2_normquant_check.py`（扩展） | −0.4 ~ −0.5 ms |
