# Phase 2 第 1 轮 Coordinator 日志

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) §3、[optimization workflow](../optimization-workflow.md)
- 本轮集成基线：`1ac7460`（`feat: add an op experiment kind for phase 2 kernel gates`，
  在 `ed52250` 之上只加 `op` kind 的队列/产物基础设施与文档）
- 冻结基准：checkpoint `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`；历史性能
  baseline `bff10d3`（`results/m3/20261022T003024-m3-bench/`）

## 时间线

| 时间 | task/session | 动作 | 理由与结果 ID |
|---|---|---|---|
| 12:30 | coordinator | 启动唯一 `serve` | 队列空闲、GPU 无外部进程；`status` 显示上个 session 已退出 |
| 12:35 | infra | 新增 `op` kind（`1ac7460`） | Phase 2 的 5 个任务各自需要 ≤5 分钟的算子 gate 入口；复用 m3_check 的 fixtures/report 结构会强制每个 op 都加载整份模型。`op` 契约要求每 case 有 `coverage:` gate + 数值 metric，diagnostic-only 不算通过 |
| 12:40 | infra | `pytest tests/experiment -q` → 79 passed | 新增 11 个用例覆盖正例、gate 失败、dirty、缺 format、缺 coverage、场景/steps 不一致、submit 接受/拒绝 op 脚本 |
| 12:42 | coordinator | 提交短协议 baseline job `job-000012-6fa10f8c` | plan §2 要求短测 baseline 用同一短协议另测；`graph-512-1024,graph-512-8192,graph-8192` × 32 步 × 1 次，commit `1ac7460` |
| 12:45 | 5×Explorer | 派发 A/B/C/D/E | 见下表；worktree 从 `1ac7460` 切出，各自独立 branch |

## 派发记录

| 任务 | session_id | worktree / branch | 脚本入口 | 本轮范围 |
|---|---|---|---|---|
| A BF16 decode attention | `phase2-attn` | `.tmp/worktrees/explorer-attn` / `exp/phase2-attn` | `scripts/phase2_attn_check.py` | 不物化 `repeat_kv` 的紧凑 GQA decode attention + 端到端筛选 |
| B CUDA-core NVFP4 GEMV | `phase2-gemv-cuda` | `.tmp/worktrees/explorer-gemv-cuda` / `exp/phase2-gemv-cuda` | `bench/micro/nvfp4_gemv.py` | SIMT GEMV 对 4 个代表 shape 的 op gate + microbench |
| C tensor-core NVFP4 GEMV | `phase2-gemv-tc` | `.tmp/worktrees/explorer-gemv-tc` / `exp/phase2-gemv-tc` | `bench/micro/nvfp4_gemv.py` | swap-AB + 原生 FP4 MMA 的同一组 shape |
| D GDN decode | `phase2-gdn` | `.tmp/worktrees/explorer-gdn` / `exp/phase2-gdn` | `scripts/phase2_gdn_check.py` | recurrent decode 片段融合（FP32 state 原地） |
| E activation quant 归因 | `phase2-quant` | `.tmp/worktrees/explorer-quant` / `exp/phase2-quant` | `scripts/phase2_quant_attrib.py` | 差异规则归因 + 冻结融合 kernel 所需的 quant reference |

## 短协议 baseline（`job-000012-6fa10f8c`）

commit `1ac7460`，checkpoint `5b7a687`，32 步 × 1 次，结果
`results/experiments/job-000012-6fa10f8c/output/bench/bench.json`：

| 场景 | step p50 ms | 模型 tok/s | 系统 tok/s | free GB |
|---|---|---|---|---|
| graph-512-1024 | 31.85 | 31.40 | 31.36 | 1.170 |
| graph-512-8192 | 44.03 | 22.71 | 22.69 | 0.520 |
| graph-8192 | 44.33 | 22.55 | 22.54 | 0.119 |

与 511 步 × 5 次的历史值（31.84 / 44.04 / 44.38 ms）一致，差异 <0.2%，
所以短协议可以作为筛选对照；但正式结论仍以 511×5 为准。

## 事故与修复：stale serve 拒绝 `op` kind（job-000013/14/15）

- 现象：GDN / attention / tensor-core GEMV 三个 job 在 0.05–0.09 s 内 failed，
  `steps_done=0`，错误 `请求重放校验失败: steps[0].kind 必须是
  ('check','bench','memory','profile') 之一`，`category=invalid_request`。
- 根因：`serve`（pid 1238791）于 12:21:17 启动，`op` kind 的提交 `1ac7460`
  是 12:23:11；执行器在**启动时**导入 `runner/experiment/schema.py`，执行期
  重放校验用的是进程内的旧 `KINDS`。submit 客户端读磁盘上的注册表，所以
  「submit 成功」不代表执行器认识这个 kind。
- 处理：重启 serve（pid 1264935，加载 `1ac7460` + `edfcc6b`），让三个 Explorer
  用同一 commit 重投。写入 [experiments.md §11](../experiments.md) 的已知限制。
- 另一个跨 session 的坑：job worktree 的 `.venv` / `models` 是符号链接，
  `.gitignore` 的 `.venv/`、`models/` 只匹配目录，裸 `git status --porcelain`
  把它们当 untracked → 所有 `op` 产物 `dirty=true` → `validate_op` 拒绝。
  修在 `edfcc6b`：writer 改用 `runner.report.git_state`（只统计源码改动），
  validator 额外容忍这两个名字的 untracked；补了 2 个用例，`pytest
  tests/experiment` → 81 passed。代价是 3 个 job 的提交槽被浪费。

## 中间结果（截至 12:40）

### A：紧凑 GQA decode attention（commit `914b001`，job-000016 op 步骤）

op 步骤的 `op.json` 已写出并通过真实 `validate_op`（128/128 gate、7 个 case、
dirty=false），但脚本在写 stdout 汇总时 `KeyError: 'metrics'` 退出 1，导致
check/bench 两步没跑。这条路**不重跑 op**：证据已经在产物里，等 Explorer 修打印
路径后重投同样三步。

| 长度 | baseline ms/call | candidate ms/call | 加速 | KV GB/s（候选） | ×16 层省 |
|---|---|---|---|---|---|
| 513 | 0.0527 | 0.0431 | 1.22× | 48.8 | 0.15 ms |
| 8191 | 0.8556 | 0.1775 | 4.82× | 189.0 | **10.85 ms** |

- 数值：vs baseline nrmse 3.1e-4（8191）、vs fp64 的距离与 baseline 之比 1.00；
  5 个长度 × 4 种 mask/view 组合都过，128 gate 全绿。
- 容差口径调整（已接受）：原建议的绝对判据 nrmse ≤5e-3 / cosine ≥0.99999 在
  L=1025 连**冻结的 eager baseline 自己**都过不了（实测 5.48e-3 / 0.9999855），
  因此绝对 sanity 冻在 1e-2 / 0.9999 / max_rel 2e-2，主判据仍是「候选到 fp64
  ≤ 2× baseline」。调整发生在看候选结果之前、并写进产物；记录保留。
- 预期 8K：44.33 → 约 33.5 ms（~29.9 tok/s，+33%）；短 context 基本持平。

**端到端确认（job-000020，commit `59bb8a5a`，correctness=passed，175.5 s）**：

| 指标 | baseline job-000012 | 候选 | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 44.33 ms | **31.63 ms** | −12.70 ms（−28.7%） |
| graph-8192 系统吞吐 | 22.54 tok/s | **31.58 tok/s** | **+40.1%** |
| 峰值 allocated | 15.287 GB | 15.287 GB | 同 |
| free | 0.119 GB | **0.308 GB** | +0.19 GB |

- op 128/128、check `graph-1024` **24/24 gate 全过**：三条同 shape bitwise gate
  （eager→static-exact、padded-ref→static-cap、static-cap→graph）没有被破坏。
  唯一失败项是 diagnostic「graph 逐步 argmax 与冻结 trace 一致」，属 M3 已记录的
  free-running 数值分叉，不是 gate。
- 8K 单步已基本等于 512 档（31.85 ms），attention 与 depth 脱钩；显存余量
  0.119 → 0.308 GB。
- 剩余结构：31.63 ms 里 NVFP4 GEMM 仍约 20.7 ms（65%），所以下一轮的杠杆是 GEMV。

### D：GDN recurrent decode 融合（commit `a609f4c`，job-000017）

| 项 | baseline | candidate | 结论 |
|---|---|---|---|
| 48 层/step，graph replay | 3.922 ms | 0.676 ms | **省 3.246 ms（5.80×）** |
| 48 层/step，eager | 6.223 ms | 0.791 ms | 7.86×（含 host launch，只作诊断） |
| 参数 | bv32-w4 0.791 / bv64-w8 0.820 / bv16-w2 0.945 | | bv32-w4 |

- 202/202 gate 通过；vs 未修改的 `torch_recurrent_gated_delta_rule`（2 步、96 层次）
  worst_nrmse 9.2e-5、cosine 0.99999982、max_rel 1.28e-3；state 仍 FP32、原地、
  data_ptr 与字节不变。
- job 级 `correctness` 被记成 `not_checked`：执行器只把 `check` 当 correctness
  步骤，`op` 的证据只在 step 级 validation 里。已在 `cc1c9aa` 修掉
  （`compute_correctness` / `merge_metrics` 纳入 op，含用例，82 passed）。
- 预期 e2e：省 ~3.2 ms/step，且与 depth 无关（8K 44.33 → 41.1 ms；512
  31.85 → 28.6 ms）。已派发端到端接入任务（要求不放宽任何 bitwise gate）。

## 本轮边界

- 每个实现任务只做一个实现方向、最多 3 组粗粒度参数、一次正确性修复后重测；E 限一次归因与验证。
- B/C 只做 op 级证据，不接主线；A 因为收益来自“全局 decode attention 路径”，
  本轮就做端到端接入（同一 job 里 op gate → `m3_check graph-1024 16` → `graph-8192 32×1`）。
- B/C 的胜者与 A/D 的接入验证留到本轮汇总后，作为下一批任务；组合集成由 Coordinator 做。

## 操作失误：重启 serve 杀掉刚启动的 job（job-000019）

- 12:38:09.4 `job-000019`（attention 重投）开始执行；12:38:10.3 我为加载
  `cc1c9aa`（op 聚合修复）重启 serve，执行器按设计 SIGTERM 进程组，job 记
  `interrupted`（step 0 exit_code -15，0.9 s，无有效测量）。
- 原因：重启前的检查只看了 `status.jobs[-2:]`，没有显式检查
  `any(state == "running")`。queue_depth=0 不代表没有 job 在跑。
- 处理：请 Explorer 用同一 commit 重投；把「重启前必须确认没有 running job」
  写进本节，后续重启按此执行。
- 代价：本轮第 3 个被浪费的提交槽（前两个是 stale serve）。

## 条件：重启 serve 的检查清单

1. `status` 里没有任何 `state in {running, queued}`；
2. `nvidia-smi --query-compute-apps` 为空；
3. 记下旧 pid，确认它已退出后再启动新进程。

### C：tensor-core NVFP4 GEMV（job-000018 第一版 / job-000021 重测）→ 方向判负

第一版（`50205e7`，job-000018）报了 3.27× 加速，但小 shape 的「带宽」超过 960 GB/s
物理峰值：in_proj_qkv 1425、gate_proj 2423、k_proj 158 GB/s。原因是工作集
（29.5 / 50.2 / 2.6 MB）整块进了 L2（GB203 ≈ 64 MB），迭代之间重复读同一份权重；
另外 `mm_fp4` 的 60–68 µs 与 shape 无关，是 host 开销被 event 计时算进去。
只有 lm_head（715 MB）是干净的：0.795 vs 0.804 ms，1.01×。

重测（`20d75d1`，job-000021）改成「每 case 扫 copies 份不同地址副本 + graph replay
边界 + 两侧吃同一份 FlashInfer packed A」，加权聚合（cover 37% 的 step 字节）：

| shape | instances | 候选 GB/s | mm_fp4 GB/s |
|---|---|---|---|
| in_proj_qkv | 48 | 767 | 793 |
| gate_proj | 64 | 625 | 813 |
| lm_head | 1 | 887 | 890 |
| k_proj | 16 | 147 | 165 |
| **加权** | | **664** | **789** |

- `aggregate_speedup = 0.84`：**比 cutlass 慢 16%**，est_step_ms 21.7 vs 18.3。
  唯一打平的是 lm_head（都到 0.93 峰值带宽）。结论：这个方向不接入，不细调参数。
- **记录更正（由 Explorer 复核 artifact 后指出）**：v2 的数值 gate 是 **4/4 全失败**，
  而 `mm_fp4`（M2 冻结的正确路径）与同一个 reference 也不一致（nrmse 0.89/0.32/0.89/0.92）；
  反变换出的 scale 与 reference quantizer 逐字节相同（`scale_diff=0/320`），
  `gemm_tc.py` 在两个 commit 之间零改动。所以是**测量脚本 v2 的 reference/接线回归**，
  不是 kernel、也不是 `_unswizzle_a_scale`。候选的正确性证据仍以 v1（job-000018，
  nrmse 0.0016–0.0017、36/36 gate）加 CPU interpreter 为准；v2 的 numeric gate 无效。
  这不改变方向判定（0.84× 的 timing 与数值无关），但记录必须是「性能更慢」而不是
  「gate 失败 + 更慢」。
- 独立交叉证据（对任务 E 有用）：同一 seed 下 `in_proj_qkv`/`k_proj` 的
  FlashInfer packed A 与 PyTorch reference 逐字节相等，`gate_proj`/`lm_head`
  只有 `input_scale` 不同 —— 即 M2 记录的 E2M1 midpoint / FP8 scale 差异是真实的，
  不是比较方式问题。
- 重要口径修正（**外推，不是实测**）：在这 4 个 shape 上 mm_fp4 单 kernel 聚合是
  789 GB/s；按同档外推到 14.399 GB 得到 18.26 ms，而 profile 的整步值是 20.7 ms。
  差的 ~2.4 ms 来自 activation quant、reciprocal、逐层 launch/层间干扰和未覆盖的
  shape（up/down_proj 等，占 63% 字节）。所以「GEMV 能省 4.7 ms」不能当上界，
  下一轮要先用 profile 把这 2.4 ms 的归属测出来，再决定投 kernel 还是投融合。

### B：CUDA-core SIMT NVFP4 GEMV（job-000022）→ 数值 gate 无效（reference 错），性能打平

- 失败项：6 个 case 的 `numeric: nrmse/cosine/max_rel` 以及部分 `k_split` 档。
- **判定：reference 错，不是 kernel 错**。`op.json` 里每个 case 的 `gemv_nrmse` 与
  `mm_fp4_nrmse` 逐位相同（0.9632/0.9632、0.7127/0.7127、0.9587/0.9587…）。两个独立
  实现偏差完全相同，只能是 dequant 参考的口径错了。已要求修 reference 后重测一次，
  自检标准是 `mm_fp4_nrmse` 回到 ~1.6e-3（M2 记录值）。
- 性能（同口径真实实例聚合，覆盖 60% 的 step 字节）：gemv 12.88 ms / 8.64 GB =
  **670 GB/s**，mm_fp4 13.00 ms = **665 GB/s**，比值 **1.01×**。分 shape：
  in_proj_qkv 1.07×、gate_proj **0.83×**、lm_head 0.95×、down_proj 0.97×，
  k_proj 的 2.65× 因工作集仅 1.4× L2 不计。
- 与 C 合起来的结论：**M=1 的 GEMV kernel 不是本轮的杠杆**。cutlass 在干净工作集上
  已经到 665–890 GB/s（随 shape 混合变化），换 SIMT / tensor-core 都没有聚合收益。
  下一轮的 GEMV 方向必须改问「能不能把 norm+quant 融进去」，而不是「换一个更快的
  GEMV」；这条依赖 E 的规则结论。
- 保留价值：SIMT kernel 能原地吃 128x4 swizzled scale（不 extra 1.6 GB），修好
  reference 后的正确性证据是后续融合的基础。

### E：activation quant 差异定位（job-000024，commit `6a5a1a2`）→ 规则确定，可冻结

- 51/51 gate 通过，8 个 case（3 synthetic + 3 real-M2 replay + 2 heldout），
  correctness=passed，113 s。
- **候选规则 R2 = FlashInfer kernel 的 PTX 算术序列**在 **977,408 个 block /
  15,638,528 个 code 上逐位复现**（`rule_mismatch_codes=0`、`residual_ratio=0`，
  fit 与 heldout 都是 0）。
- 对照 R1（用正确舍入的 1/x 代替 `rcp.approx.ftz`）仍有 84–130 个 code 差异
  （1.1e-4 ~ 3.9e-4）→ **必须照抄 kernel 的近似倒数序列**，数学上更精确的参考反而错。
- R0（现有 M2 reference）的差异分布以 `global-divisor-mul-recip` 为主
  （real-m2-k5120 902 块 / k17408 4237 块），其次 `fp8-scale-rounding`、
  `e2m1-midpoint-tie`；`m2_replay_match_ratio=1.0`，重放复现了记录的失败计数。
- 结论：**「差异来自什么规则」已回答；融合 kernel 所需的 quant reference 可以冻结为
  R2**。下一轮的 norm+quant 融合有了可依赖的语义，不需要再靠容差。


### D：GDN 端到端接入（job-000023 接线失败 / job-000026 修复后成功）

第一次接入（`a80029d`）op 199/199 通过，但 bench **完全没有变化**
（44.329 / 31.840 ms）。产物里的 `load.gdn_decode.counters` 给出根因：
`fused=0, reference=960, fallbacks={head-count: 960}` —— patch 的命中条件按
**repeat 之前**的 head 数写（16×3=48），真实调用点是 repeat **之后**的 48 head
（48×3≠48），所以 960/960 全部回退。

修复（`e6bd6c7`）后：

| 场景 | baseline | 修复前 | 修复后 | 变化 |
|---|---|---|---|---|
| graph-8192 step p50 | 44.334 ms | 44.329 | **40.490 ms** | −3.84 ms / −8.7% |
| graph-8192 系统 | 22.54 tok/s | 22.54 | **24.68** | +9.5% |
| graph-512-1024 step p50 | 31.845 ms | 31.840 | **28.208 ms** | −3.63 ms / −11.4% |
| graph-512-1024 系统 | 31.36 tok/s | 31.37 | **35.40** | +12.9% |

- op 615/615：`patch:fused-path-executed` 断言 `fused_calls=96 / reference_calls=0 /
  fallback_kinds=0 / out_bytes=589824`，接线可判定；check 24/24。
- 两个 depth 都省 3.6–3.8 ms，与「GDN recurrent 开销与 depth 无关」一致；比
  microbench 的 3.239 ms 还多一点，说明局部收益**可以**迁移。

### B：SIMT GEMV（job-000022 / job-000025）→ 数值正确、性能平局、不接入

- v1（job-000022）：reference 把 UE4M3 scale 当 uint8 `.float()` 用，ref 被逐 block
  放大 3–9×，nrmse 饱和到 ~1；候选与 `mm_fp4` 对同一错误 ref 偏差**逐位相同**
  （0.9632/0.9632 等），已据此判定是 reference 错。
- v2（job-000025，`b4d6a15`）：reference 修对后，6 个 case 的 `gemv_nrmse`
  全部 ≈1.67e-3、cosine ≈0.9999986，且 **`cand_vs_mm_fp4_nrmse = 0.0`** ——
  候选与 M2 冻结的正确路径**逐位一致**。
- 唯一失败 gate 是 v2 新加的 `reference: 两条 A 反量化路径一致（nrmse ≤1e-2）`
  （gate_proj 1.38e-2）。这条比的是**两个不同量化器**（FlashInfer R2 vs M2 R0），
  而 E 的 job-000024 已把差异证明到 bitwise（gate_proj 的 divisor=827.1，byte_diff
  10/2560，`a_code_mismatch=13` 正是它）→ **gate 规格错，不是候选错**。本轮
  不再重测（B 已用掉「一次正确性修复后重测」）。
- 性能（同口径、真实实例聚合、60% 字节）：gemv 12.88 ms / 8.64 GB = 670 GB/s，
  mm_fp4 13.00 ms = 665 GB/s，**1.01×**；gate_proj 0.83×、lm_head 0.95×、
  down_proj 0.97×。**不接入**，但 kernel 能原地吃 128x4 swizzled scale、且与
  cutlass 逐位一致，可作为下一轮 norm+quant 融合的基础。

## 集成与确认（job-000027）

- 集成分支 `exp/phase2-integration`：`9e35873`（A `59bb8a5` + D `e6bd6c7`，
  `runner/loader/nvfp4_model.py` 的 info 字段做了唯一一处冲突合并，两个 install
  都保留）。
- `job-000027-cf8ecb65`（commit `9e35873`，timeout 3000）：
  1. `check`：`m3_check --only numeric-512,graph-1024,graph-8192 --steps 16`
     —— 补上 A 单独验证时缺的 `eager→static-exact` 与 `padded-ref→static-cap`
     bitwise（只在 `numeric-*` 里），外加 8K 场景；
  2. `bench`：三个 graph 场景 × 511 步 × 5 次（正式口径）。
- 之后单独提交 `m3_memory --only graph-8192 --steps 511` 做 8K 显存检查。
