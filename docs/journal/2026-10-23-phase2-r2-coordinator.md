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

## 任务 F 第一次提交（job-000032-08b17afa）：cutlass 基线缺失，但 backend 级数据有效

- 失败项：`coverage: backend cutlass 可用性或不可用原因已记录`（`produced_configs = 0`）
  与 `wiring: cutlass 默认 tactic 与显式 tactic 逐位一致`（`bitwise_equal_tactics = []`）。
  即：私有 tactic  traversal 把 cutlass 默认路径也打断了，**唯一缺失的是 baseline**，
  所以 `proposed_table` 全填 cute-dsl 是没有基准的假象。
- 有效的部分（每 case 工作集 1.0–1.4 GB，L2-free，graph replay）：

| case | cutlass（C 的历史值） | b12x | **cute-dsl** | cudnn |
|---|---|---|---|---|
| in_proj_qkv | 793 | 850.6 | **870.2** | 803.4 |
| in_proj_z | — | 804.5 | **836.4** | 729.3 |
| q_proj | — | 796.0 | **843.3** | 655.4 |

- 所有 backend 的 numeric gate 全过（nrmse ≈1.64e-3），且 **backend 之间逐位一致**
  （`nrmse = 0.0 / cosine = 1.0`）——换 backend 不改变数值语义，接入风险低。
- 若整步也按这个比例，潜在收益是 **−1.3 ~ −2.6 ms**，比 F 自己保守估计的
  −0.3 ~ −1.0 好；但要等修好 cutlass 基线、拿到同 job 内的加权对照才能确认。
- 处理：允许一次修复重投，要求 (a) cutlass 基线必须走官方 `mm_fp4(backend="cutlass")`
  入口，(b) 私有 tactic 路径放 try/except，确认不可达就删掉子实验、收敛为 backend 级。

## 任务 G 采纳（job-000033-1bfa42c1，commit `ff0c87b`）

| 指标 | baseline job-000029 | G | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 27.781 ms | **27.052 ms** | −0.729 ms（−2.6%） |
| graph-8192 系统 | 35.99 tok/s | **36.92** | +2.6% |
| graph-512-1024 step p50 | 26.779 ms | **26.079 ms** | −0.700 ms（−2.6%） |
| graph-512-1024 系统 | 37.29 tok/s | **38.30** | +2.7% |

- check **139/139** 全过（`numeric-512` + `graph-1024`）：`eager→static-exact` 与
  `padded-ref→static-cap` 的 bitwise 没被破坏，说明「四个执行方式的 decode 走同一条
  M=1 路径」这个假设成立。
- op 37/37：`cand_vs_fp64 / ref_vs_fp64 = 0.62–0.75`（候选更准）、
  `padded-path-bitwise-unchanged` 9/9、`decode-patch-coverage` 96/0/0、
  microbench warm 7.87 → 1.617 µs（4.87×）。
- 两个 depth 收益一致（0.70 / 0.73 ms），符合「与 depth 无关的固定开销」预期；
  比 microbench 外推的 0.60 ms 还多，因为 pad 的两个 kernel（fill + cat，2.78 µs/次）
  在真实模型里也被一并省掉了。

## 任务 I 采纳（job-000034-486a8274，commit `4a16d0e`）

| 指标 | baseline job-000029 | I | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 27.781 ms | **27.514 ms** | −0.267 ms |
| graph-8192 系统 | 35.99 tok/s | **36.30** | +0.9% |
| graph-512-1024 step p50 | 26.779 ms | **26.550 ms** | −0.229 ms |
| graph-512-1024 系统 | 37.29 tok/s | **37.62** | +0.9% |

- op **524/524**：worst nrmse 1.29e-9（实质逐位）、`fused=96 / reference=0 / fallbacks=0`、
  三个负对照 nrmse 1.45 / 1.25 / …（v1 的「负对照恒等」已修好）。
- microbench 0.271 ms vs e2e 0.267/0.229 ms，局部收益完整迁移。
- v1 失败的两个根因都是**判据 bug**：in-place 的 reference 用了调用后的 state 快照、
  负对照拿「reference 喂同样的错输入」当参照；都不是 kernel 数学错。最终按建议放弃
  in-place，改为 48 个固定 out 槽（0.94 MB）。

## 任务 F backend sweep（job-000035-461c30c4，commit `959b667`）：cute-dsl 全胜

198/198 gate 全过，11 个 case 覆盖全部 401 个 NVFP4 Linear（14.399 GB），L2-free +
graph replay 计时，cutlass 基线走官方入口：

| case | cutlass GB/s | cute-dsl GB/s | cutlass ms | cute-dsl ms |
|---|---|---|---|---|
| in_proj_qkv | 799.5 | 871.2 | 1.7707 | 1.6248 |
| in_proj_z | 732.5 | 836.4 | 1.1596 | 1.0155 |
| q_proj | 648.1 | 843.3 | 0.8736 | 0.6714 |
| k_proj / v_proj | 164.6 | 504.8 | 0.2867×2 | 0.0935×2 |
| o_proj | 681.6 | 828.6 | 0.4154 | 0.3417 |
| out_proj | 684.0 | 832.2 | 1.2418 | 1.0206 |
| gate_proj | 827.3 | 882.4 | 3.8786 | 3.6362 |
| up_proj | 822.5 | 882.5 | 3.9011 | 3.6359 |
| down_proj | 742.0 | 883.7 | 4.3245 | 3.6311 |
| lm_head | 888.9 | 911.0 | 0.8045 | 0.7851 |
| **合计** | **760** | **870** | **18.943** | **16.549** |

- **cute-dsl 11/11 全胜，加权 −2.39 ms/step（−12.6%）**；`auto == b12x`（诊断），
  cudnn 全面更慢。k/v_proj 从 164.6 涨到 504.8 GB/s（CTA 数从 8 变多的效果）。
- 三个 backend 的输出**逐位一致**（`backend-agreement` nrmse 0.0），所以换 backend
  不改变数值语义 —— 这让接入的 bitwise 风险很低。
- 已派发第二个 job：只在 M==1 查静态 `(N,K)->backend` 表（全填 cute-dsl、未命中回退
  cutlass），prefill 保持 cutlass，然后跑 `numeric-512,graph-1024` + 两个 graph 场景。

## 任务 H 第一次提交（job-000036-e448042d）：差 1 个字节

- 失败项只有 2 类，且都是 1 byte：
  1. `reference-agreement: F.rms_norm 与 Qwen3_5RMSNorm 逐位相同`（24 样本
     `worst_diff_bytes = 1`）；
  2. `distribution-edge: bitwise[layers.0.input_layernorm/saturation]`
     （`out_bf16_diff_bytes = 1`）。
- 判定与修正：
  - **第 1 条是 gate 规格错**：模型执行的是 `Qwen3_5RMSNorm.forward`，`F.rms_norm` 是
    另一个实现，1 ulp 差异正常，应降为 diagnostic；决定性 gate 是「kernel 产出的
    `out_bf16 + 每消费者 (a_fp4, a_sf)` 与**模块路径**逐字节相同」。
  - **第 2 条是真差异**，最可能来自 `Qwen3_5RMSNorm.forward` 的两个细节：
    `output * (1.0 + self.weight.float())`（权重是 0 附近的偏移，必须在 fp32 里先加 1.0
    再乘，不能预计算成 bf16）与「全程 fp32、最后一次性 `type_as(x)`」
    （不能像 `Qwen3_5RMSNormGated` 那样先 cast 再乘 weight）。
- 已请 H 修 + 重投一次（本轮最后一次）。

## 任务 F 采纳（job-000037-150586b0，commit `3c3ac27`）

| 指标 | baseline job-000029 | F 接入后 | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 27.781 ms | **25.084 ms** | **−2.697 ms（−9.7%）** |
| graph-8192 系统 | 35.99 tok/s | **39.81** | +10.6% |
| graph-512-1024 step p50 | 26.779 ms | **24.090 ms** | −2.689 ms（−10.0%） |
| graph-512-1024 系统 | 37.29 tok/s | **41.42** | +11.1% |

- **check 139/139**：三组 bitwise（`eager→static-exact`、`padded-ref→static-cap`、
  `static-cap→graph`）全部保持，说明「backend 间输出逐位一致」在端到端成立。
- 接入面：`DECODE_BACKEND_TABLE`（8 个 (N,K) → cute-dsl），只在 `M==1` 查表，
  prefill 仍 cutlass，`mm_fp4` 其它参数未动；失败回退 + 永久禁用 + 每进程一次 warning。
- JIT 时机可判定：`nvfp4_decode.first_call_ms` 每 shape 一条（lm_head 19.95 ms、其余
  ~1 ms），发生在 `GraphDecodeStep.warmup()` 的 3 步 eager decode 里，steady state 无编译。
- e2e −2.70 ms 比 op 级加权 −2.39 ms 多 0.31 ms（k/v_proj 从 164→505 GB/s 在整图内
  还被额外放大），两个 depth 收益一致，说明是纯权重带宽收益。

## 任务 H 第二次提交（job-000038-ad133bc8）：op 全过、check 在 eager 路径崩

- **op 步骤 260/260 全过**：`real-layers` 128 个真实 norm / 1,392,640 个 code 逐字节相同、
  `scales-extremes` 72/72、`distribution-edge` 48/48、`mean-tree` / `rsqrt-approx`
  （8228 个样本）全 0 差异；micro 2 消费者 0.0702 → **0.0155 ms/次**。
  即「用 E 的 R2 契约把 norm+quant 融合成 bitwise 相同的 kernel」这个难点已经成立。
- check 的 `numeric-512` 在 **eager（DynamicCache）路径**崩：
  `norm_quant_patch.py:236 → norm_quant_fused: a_fp4 / a_sf length must match gs_list`。
  是消费者计划的长度与应用数量不一致，不是 kernel 算术问题（op 的 static/graph 都过了）。
  说明上一轮只验证了 static/graph 两种执行方式。
- 处理：授权**最后一次**修复 —— 要求 (a) 长度不一致时安全回退 + 计数而不是抛异常，
  (b) **四种执行方式都要跑通**，(c) 重投同一个 3 步 job。
- 记账：这一条线目前**未采纳**（op 有强证据，但端到端未验证）；若最后这次仍失败，
  留给第 3 轮，原因是「融合 kernel 在 eager/DynamicCache 路径的消费者计划不完整」。

## 第 2 轮集成测量（job-000039-3eb94d6c）

- 集成 commit **`a855a56`** = `9e35873`（A+D）+ G（`af034f5`）+ I（`14f06de`）+ F（`1035abd`）。
  唯一一处需要人工解决的冲突是 `runner/loader/nvfp4_model.py` 的 info 字段（A 与 D 各加
  一个 key，已同时保留）。
- 步骤：`check --only numeric-512,graph-1024,graph-8192 --steps 16` +
  `bench --only 三个 graph 场景 --steps 511 --repeats 5`，timeout 3000。
- 预期（三项相加）：8K 27.78 − 0.73（G）− 0.27（I）− 2.70（F）= **约 24.1 ms**，
  系统约 41.5 tok/s，%SOL 约 66%；512-1024 约 23.1 ms。

## 第 2 轮集成正式结果（job-000039-3eb94d6c，commit `a855a56`）

512 步 × 5 次，correctness=**passed**，**163/163 gate 全过**
（numeric-512 115、graph-1024 24、graph-8192 24）：

| 场景 | M3 baseline | 第 1 轮集成 | **第 2 轮集成** | 系统 tok/s | %SOL |
|---|---|---|---|---|---|
| graph-512-1024 | 31.84 ms | 26.80 ms | **23.117 ms** | 31.38 → **43.21** | 48.5% → **66.7%** |
| graph-512-8192 | 44.04 ms | 27.84 ms | **24.098 ms** | 22.70 → **41.45** | 35.0% → **66.1%** |
| graph-8192 | 44.38 ms | 27.83 ms | **24.097 ms** | 22.52 → **41.45** | 35.9% → **66.1%** |

- 可加性：G(−0.73) + I(−0.27) + F(−2.70) = −3.70 ms，实测 −3.73 / −3.74 / −3.68 ms，
  在 0.05 ms 内吻合，说明三项互不干扰。
- 5 次重复的离散度很小（8K model tok/s std ≈ 0.03，见 bench summary）。
- 显存：8K peak allocated 15.294 GB（第 1 轮 15.293），free **0.335 GB**（第 1 轮 0.329）。
- 相对 M3 baseline：44.38 → 24.10 ms（−45.7%），系统 22.52 → 41.45 tok/s（+84%）。
