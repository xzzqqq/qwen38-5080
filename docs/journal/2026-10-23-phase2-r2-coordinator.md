# Phase 2 第 2 轮 Coordinator 日志

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §3–§4、[workflow](../optimization-workflow.md)
- 起点：第 1 轮集成 commit **`9e35873a01c0769fbed302a9f0b32a222e3ffcd1`**
  （紧凑 GQA decode attention + GDN recurrent decode 融合；3 个 graph 场景 511×5 =
  26.80 / 27.84 / 27.83 ms，系统 37.27 / 35.89 / 35.89 tok/s，57.2–57.5% SOL）
- 第 1 轮记录：[r1 coordinator](2026-10-23-phase2-r1-coordinator.md)

## 本轮依据（集成后 profile，job-000028，graph-7680 8 步）

每步 6013 个 kernel / 26.49 ms kernel / 28.35 ms 墙钟：

| ms/step | 次数 | kernel |
|---|---|---|
| 19.299 | 401 | FlashInfer `mm_fp4`（747 GB/s） |
| 2.275 | 3156 | `vectorized_elementwise` |
| 1.529 | 128 | wmma bf16 GEMM（ShapeStableLinear，M=128 padding） |
| 1.001 | 610 | `unrolled_elementwise`（direct_copy） |
| 0.730 | 633 | `elementwise_kernel` |
| 0.527 | 401 | `quantize_with_block_size` |
| 0.406 | 274 | `at::native::`（同类 elementwise） |
| 0.314 | 210 | `reduce_kernel`（RMSNorm） |
| 0.263 | 48 | `_gdn_decode_kernel`（上轮融合） |
| 0.086 | 96 | `cublasLt::splitKreduce` |

结论：GEMV 已经等于 L2-free 聚合口径下的 747 GB/s，**换 GEMV kernel 不是杠杆**；
7.19 ms 全在小 kernel（92% <5 µs）。本轮打「kernel 数量与浪费」。

## 时间线

| 时间 | task/session | 动作 | 理由与结果 ID |
|---|---|---|---|
| 13:18 | infra | `66782a9` 注册 4 个第 2 轮 op 入口，重启 serve | 队列确认空闲（无 running/queued）后重启；吸取上轮误杀 job-000019 的教训 |
| 13:20 | coordinator | 提交短协议 baseline `job-000029-16bb35a9` | 3 个 graph 场景 × 32 步 × 1 次，commit `9e35873`，作为本轮筛选对照 |
| 13:22 | F/G/H/I | 派发 4 个 Explorer | worktree 从 `9e35873` 切出，各自独立 branch |

## 派发

| 任务 | session | worktree / branch | 脚本入口 | 目标 |
|---|---|---|---|---|
| F per-shape mm_fp4 backend/tactic | `phase2-tactics` | `.tmp/worktrees/explorer-tactics` / `exp/phase2-tactics` | `bench/micro/nvfp4_tactics.py` | −1.0 ~ −2.5 ms |
| G ShapeStableLinear M=1 decode | `phase2-stable-linear` | `.tmp/worktrees/explorer-stable-linear` / `exp/phase2-stable-linear` | `bench/micro/stable_linear_decode.py` | −1.1 ~ −1.3 ms |
| H norm + activation quant 融合（R2 bitwise） | `phase2-normquant` | `.tmp/worktrees/explorer-normquant` / `exp/phase2-normquant` | `scripts/phase2_normquant_check.py` | −0.5 ~ −1.2 ms |
| I GDN conv decode 融合 | `phase2-gdn-conv` | `.tmp/worktrees/explorer-gdn-conv` / `exp/phase2-gdn-conv` | `scripts/phase2_gdn_conv_check.py` | −0.5 ~ −1.0 ms |

本轮边界：每个任务一个实现方向、最多 3 组粗粒度参数、一次正确性修复后重测；
G/H/I 必须带 `numeric-512` 或独立 gate 证明 bitwise 语义未变；组合集成与正式测量
由 Coordinator 做。

## 短协议 baseline（`job-000029-16bb35a9`）

commit `9e35873`，32 步 × 1 次，结果
`results/experiments/job-000029-16bb35a9/output/bench/bench.json`：

| 场景 | step p50 | 模型 tok/s | 系统 tok/s | free GB |
|---|---|---|---|---|
| graph-512-1024 | 26.779 | 37.33 | 37.29 | 1.191 |
| graph-512-8192 | 27.793 | 35.99 | 35.93 | 0.730 |
| graph-8192 | 27.781 | 35.99 | 35.95 | 0.329 |

与 511×5 正式值（26.80 / 27.84 / 27.83 ms）差 <0.2%，四条线都用它做筛选对照。

## 事件：外部 GPU pytest 触发 halt（13:31）

- 13:31 serve 取 job 前检测到外部 CUDA 进程 `pytest tests/ops tests/model`（pid 1389382，
  304 MiB），等 10 s 未释放后按设计 halt 并退出（`job-000030` 保持 queued，不丢）。
- 处理：等该进程自然退出（90 s）、确认 `nvidia-smi --query-compute-apps` 为空后
  `serve --clear-halt` 恢复；给 4 个 Explorer 重申「不允许任何带 CUDA context 的
  pytest / `make m2-test`」。
- 说明：GPU 守卫按设计工作（fail closed），代价是队列暂停约 2 分钟。这属于进程级
  误用，不是工具缺陷；但**多 session 下必须把「CPU 单测」和「GPU 用例」分开**，
  后续派发消息里要写死这一点。

## 任务 I 第一次提交（job-000030-c4bd4215）失败，已允许一次修复

- 失败项 5 个：`numeric:in-place-path:out`（nrmse 0.691）、`numeric:in-place-path:conv_state`
  （nrmse 0.700）、以及三个负对照 gate（`flipped-kernel-weight` / `misaligned-conv-state`
  / `zero-conv-state` 的 nrmse 全为 0.0，说明负对照没有分辨力）。
- 已成立：`patch-conv-wiring-real-shapes`（fused 96 / reference 0 / fallbacks {}，
  nrmse 9.2e-8）；显式 out 槽路径数值正确；microbench 48 层 conv
  **0.318 → 0.047 ms（省 0.271 ms/step）**。
- 判断：in-place 写回 `x` 的错误很可能来自模型里 `mixed_qkv` 是 `transpose(1,2)` 的
  非连续 view；建议直接放弃 in-place、默认用 96 个 20 KB 的 out 槽（总计 960 KB）。
  已请 Explorer 修一版重投（第二次也是最后一次）。

## 口径修正（任务 G 指出）：`cutlass::Kernel2` 1.529 ms 不是单一来源

我在第 2 轮依据里把 `cutlass::Kernel2`（128 次/setp、1.529 ms）整体当成了
ShapeStableLinear 的 a/b GEMM。Explorer 用 `graph-7680/kern_sum.csv` 拆开：
- a/b（`32x32_128x2_tn`）**96 次 × 4.14 µs = 0.397 ms**，外加紧随其后的
  `fill` + `torch.cat` pad kernel 约 0.25 ms → 合计约 **0.65 ms**；
- 另外 32 次是 16 层 full attention 的 QK（`128x2_nn` 50.7 µs）与 PV
  （`64x1_tn` 20 µs）bmm，约 1.13 ms，**不属于任务 G**。

所以 G 的目标从 −1.1~−1.3 ms 修正为 **−0.4 ~ −0.6 ms**；同时这说明 attention 在
融合短 kernel 之后仍占约 1.1 ms（bmm + softmax + transpose/copy），是下一轮的一个
可选项。教训：profile 的 kernel 名被模板参数截断后会把多个 GEMM 合成一类，
**按名字聚合必须同时看 count 与逐实例均值**，不能只看总和。

## 任务 G 第一次提交（job-000031-1356864c）：gate 规格错，允许一次修正

- 失败项：`numeric:a/b:M=1:scale*`（nrmse 0.0038 / cosine 0.9999942 / max_rel 0.0076）
  与 microbench 的 `numeric:config-*`。
- **判定：候选更准，gate 阈值本身错**。同一份 op.json 里
  `ref_vs_fp64_nrmse = 0.0033/0.0030`、`cand_vs_fp64_nrmse = 0.0022/0.0021` —— M=1
  matvec 到 fp64 的距离比 padded 路径更小；`cand vs padded` 的 max_abs = 0.015625
  = 2^-6 正好是幅度 2 处的 1 个 bf16 ulp。而脚本里设的 `nrmse ≤1e-3`
  **比 reference 自身到 fp64 的 3.3e-3 还紧**，任何实现都过不了。
- 已成立：`padded-path-bitwise-unchanged` 8/8（M>1 逐位不变）、
  `decode-patch-coverage` 96/0/0；microbench 7.87 → **1.62 µs/call（4.87×）**，
  96 次/步预计省 **0.60 ms**。
- 处理：允许一次「gate 规格修正」重投 —— 判据改为
  `nrmse ≤5e-3` **且** `cand_vs_fp64 ≤ 1.05 × ref_vs_fp64`，其余硬 gate
  （M>1 bitwise、decode 接线、finite、coverage）不变。**这不是把 bitwise gate
  降级**：prefill 逐位不变、四个执行方式的 decode 走同一路径，`numeric-512` 的
  两组 bitwise gate 仍是硬判据。

## 任务 G 的 gate 口径修正（记录，供审查）

第一次提交的 op gate 用 `nrmse ≤1e-3 / cosine ≥0.99999 / max_rel ≤5e-3` 判
「M=1 matvec vs M=128 padded GEMM」，实测 nrmse 3.77e-3–4.29e-3、cosine
0.9999912、max_rel 7.6e-3。修正后（commit `ff0c87b`）：
- **硬 gate**：`nrmse ≤5e-3` **且** `cand_vs_fp64_nrmse ≤ 1.05 × ref_vs_fp64_nrmse`
  （实测 ratio 0.62–0.75，候选更准）；
- `cosine` / `max_rel` 降为 diagnostic（值仍写进产物）——理由是它们与 nrmse 同源，
  且 max_rel 7.6e-3 正好等于 absmax 元素的 1 个 bf16 ulp、cosine 只差 1.7e-6；
- `padded-path-bitwise-unchanged`（9/9）、`decode-patch-coverage`（96/0/0）、finite、
  `coverage:*` 全部保留为硬 gate。

判定：这不是把 bitwise gate 降级 —— prefill（M>1）逐位不变，decode 的四个执行方式
走同一条新路径，`numeric-512` 的 `eager→static-exact` / `padded-ref→static-cap`
仍是硬 bitwise。若 check 挂，直接判负。
