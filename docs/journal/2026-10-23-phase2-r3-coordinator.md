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

## 任务 L 采纳（job-000043-19c1f1a0，commit `783b36a`）

| 指标 | baseline job-000041 | L | 变化 |
|---|---|---|---|
| graph-8192 p50 | 21.824 ms | **21.142 ms** | **−0.683 ms** |
| graph-8192 系统 | 45.77 tok/s | **47.23** | +3.2% |
| graph-512-1024 p50 | 20.829 ms | **20.131 ms** | −0.698 ms |
| graph-512-1024 系统 | 47.95 tok/s | **49.56** | +3.4% |

- op **63/63**：`real-layers` 288 样本 / 1,769,472 个 e2m1 code 在 out_bf16 / a_fp4 / a_sf
  上 0 字节差；`mean-tree` 0 mismatch 而树合并变体 25 mismatch（分辨力）；三个负对照
  nrmse 2.23 / 1.05 / 显著超差。
- check **139/139**（`numeric-512` + `graph-1024`，bitwise 保持）。
- 收益是 Explorer 预估（−0.2~−0.25）的约 3 倍：graph 口径 48 层 −0.683 ms
  （14.2 µs/层），说明被省掉的是整条链路（pow/mean/rsqrt/add/mul/cast/silu +
  48 次 quantize + 48 次 `1/input_scale`），不只是它逐 kernel 折算的那几项。

## 任务 K 采纳（job-000044-bf3a545f，commit `76268da`）

| 指标 | baseline job-000041 | K | 变化 |
|---|---|---|---|
| graph-8192 p50 | 21.824 ms | **21.195 ms** | −0.629 ms |
| graph-8192 系统 | 45.77 tok/s | **47.10** | +2.9% |
| graph-512-1024 p50 | 20.829 ms | **20.648 ms** | −0.181 ms |

- op **224/224**、check **163/163**（`numeric-512` + `graph-1024` + `graph-8192` 的 bitwise 全保）。
- micro（每层 per-call，8K）：fused **0.0396 ms** vs grouped_bmm 0.1796 vs 原始 repeat_kv 0.8533
  → **比第 1 轮的紧凑 bmm 又快 4.5×**，有效带宽约 847 GB/s（33.55 MB / 0.0396 ms）。
- `fused-vs-bmm` 69/69：到 fp64 的距离比 grouped_bmm 更小（ratio 0.53）。
  `fused-fp8` 17/17：已用模拟 scale 跑通 FP8 分支（两个 scale 布局都覆盖），
  等 J 的落地后由 Coordinator 接上真实 scale。
- 设计要点：一个 program = 一个 (kv_head, split)、一次覆盖共享该 head 的 6 个 q head
  （按 q head 切会把 K/V 读 6 遍）；log2 域 online softmax；每层 2 个 kernel（原 7 个）。

## 任务 M 第一次提交（job-000047-47c60783）：o_proj 一条结构性错误

- `downproj-silu-mul` 与 `lmhead-final-norm`：`out_bf16 / a_fp4 / a_sf` **全 0 字节差**，
  负对照分辨力成立（code diff 0.886）。
- `oproj-gate` 失败：out_bf16 11076 字节差、dequant nrmse **1.58**、`producer-wiring`
  的 `attention: false`（fused −1.3516 vs ref −0.9336）。是 `attn_output * sigmoid(gate)`
  的结构错（`gate` 必须来自 `q_proj` 输出按最后一维切半 + `reshape(*input_shape,-1)`，
  `attn_output` 也要先 reshape；且 sigmoid 与乘法是**两次独立 BF16 舍入**）。
- 已允许一次修复，并要求顺带 rebase/merge 到集成 tip `66bac5c`（含 L 与 K），
  这样结果可与 L/K 叠加。

## 任务 J 第一次提交（job-000045-fbb25434）：路径选错，方向被数据否定

- `quant-roundtrip` 11/11（nrmse 2.6%、cosine 0.99966，与 CPU 校准一致）；
  `attn-fp8-vs-ref` 数值全过（nrmse 0.0059、cosine 0.99998）但一条 `coverage:` gate 计数写错；
  `attn-fp8-vs-bf16`（diagnostic）nrmse 3.7%、cosine 0.99931。
- **micro：fp8 0.2635 ms/call vs bf16 0.1795（speedup_min 0.506）**，dequant 单独
  0.0835 ms、写出带宽 71 GB/s —— 物化 bf16 前缀是净回退（×16 层约 +1.3 ms/step）。
  这与 J 自己的预判一致，也是我派发时标出的耦合点。
- 处理：要求 rebase 到 `66bac5c`，把 decode 直接喂 **FP8 + scale 给 K 的融合 kernel**
  （kernel 内折叠 scale，零 scratch），`dequant_kv` 只留给 prefill/reference 与回退；
  修 coverage 计数；**补 `smoke-graph`**（plan §5 的三条真实 smoke 是 gate，FP8 有 3.7%
  attention 误差，near-tie 可能翻转）。

## 任务 M 判定：**不接入（组合里是 +0.26 ms 回退）**

- M 自己的证据全部成立：op **277/277**（`downproj-silu-mul` / `oproj-gate` / `lmhead-final-norm`
  三条的 `out_bf16 / a_fp4 / a_sf` 全 **0 字节差**，负对照分辨力成立）、check 139/139、
  `producer-wiring` 6/6（`mlp/attention/lm_head/norm_bf16/logits` 全 true）。
- 但同协议 32×1 的组合测量（M 的分支已 merge `66bac5c`）：

| commit | graph-8192 p50 | graph-512-1024 p50 |
|---|---|---|
| `job-000041`（无 L/K/M） | 21.824 ms | 20.829 ms |
| **`job-000050`（L+K，`43b0b73`）** | **20.493 ms** | **19.951 ms** |
| `job-000048`（L+K+**M**，`132e95b`） | **20.756 ms** | 20.221 ms |

L+K 恰好等于可加预测（21.824 − 0.683 − 0.629 = 20.512 → 实测 20.493），
而加上 M 反而 **+0.263 ms**。所以 **M 不接入集成分支**；分支与 journal 保留作证据。

- 结论：**「隔离 microbench 省、组合后回退」是真实可能**（M 的 `fuse-micro` 在 eager 下
  down_proj 0.0249→0.0129、o_proj 0.0232→0.0129 ms，但 graph 口径整步变慢）。
  可能原因（未定位，留给下一轮）：复制的 `Qwen3_5MLP/Attention.forward` 让 HF 的
  `output_capturing`/`logits_to_keep` 包装多一层、或 producer kernel 对
  `[1,1,17408]`/`[1,1,6144]` 的量化比 FlashInfer 的 320-block 形态慢、
  或 `_pending_quantized` 槽位与 L/H 的槽位在 graph 内多一次分支。
  **这正是我坚持在每个候选之后做同协议组合测量的原因。**
