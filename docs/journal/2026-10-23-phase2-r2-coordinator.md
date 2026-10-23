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
