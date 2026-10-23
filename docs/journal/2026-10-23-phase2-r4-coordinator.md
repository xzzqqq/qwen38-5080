# Phase 2 第 4 轮 Coordinator 日志

- 日期：2026-10-23
- 起点：第 3 轮集成 commit **`43b0b73a5bab608a926e203cb54a2f5f891de164`**
  （L+K；511×5 = 19.97 / 20.50 / 20.51 ms，系统 50.02 / 48.71 / 48.70 tok/s，
  %SOL 77.2 / 77.6 / 77.6；163/163 gate；8K free 0.423 GB）
- 缺口：BF16 口径 80% = 19.92 ms（差 **0.59 ms**）；FP8 验收口径 80% = 19.57 ms（差 **0.94 ms**）

## 本轮派发（复用第 3 轮的空闲 session，保留上下文）

| 任务 | session | worktree / branch | 目标 | 依据 |
|---|---|---|---|---|
| N M 的「组合回退」根因 | `phase2-fuse` | `.tmp/worktrees/r3-fuse` / `exp/phase2-fuse` | 让 M 变现（预期 −0.42） | L+K 20.493 vs L+K+M 20.756（+0.263） |
| O FP8 收尾到 e2e | `phase2-fp8kv` | `.tmp/worktrees/r3-fp8` / `exp/phase2-fp8kv` | 拿到 e2e 数字（判据换成相对判据，只跑 op+bench） | op 级已证明；三次 job 停在 check 脚本 |
| P GEMV 尾部 | `phase2-tactics` | `.tmp/worktrees/r4-gemv` / `exp/phase2-gemvtail` | −0.86 ms（合并 k/v、官方入口内的结构改动） | GEMV 16.57 ms / 870 GB/s |

本轮边界：每个任务一个方向、最多 3 组粗粒度参数、一次正确性修复后重测；
N 与 P 直接决定能否在 BF16 口径达到 19.92 ms；O 决定 FP8 验收格式能否关闭。
轮末由 Coordinator 做组合 gate + 三场景 511×5 + 8K 显存。

## 第 4 轮结果：**无候选被采纳**（本轮无有效提升）

| 任务 | job | 结果 | 判定 |
|---|---|---|---|
| O FP8 收尾 | `job-000053-be869f0b` | failed（2.4 s，op） | 相对判据的 **ratio 算错**（`ratio = 3.2e9`，candidate_nrmse 0.0035，分母 ~0）；**FP8 第四次停在 check 脚本**，仍无 e2e |
| N M 根因 + tile | `job-000054-57b76ad0` | **succeeded**，op 279/279、check 139/139 | bench **20.742 ms**（L+K 20.493，**+0.25**）；N 自己的双峰分析说 mean 相等（20.511 vs 20.495），**即 M 在 p50 口径仍不成立，且没有可测收益** → 不采纳 |
| P GEMV 尾部 | `job-000055-2aa2e4e3` | op **222/222 通过**（合并 k+v / gate+up / qkv+z 逐位一致），但 `numeric-512` 的 **eager 路径崩**（coverage 缺全部 eager 项） | 无 bench 数据 → **未验证** |

## 事实与判断

1. **本轮无有效提升**（第 3 轮是 K/L 的 −1.315 ms）。按 workflow §5，这是**连续无提升的第 1 轮**；
   若第 5 轮仍无提升，必须暂停等待人工介入。
2. 三个任务都撞在**同一个系统性问题上**：新接线在 `numeric-512` 的 eager（DynamicCache）
   路径上崩，而 op 级（static/graph）全绿。本轮的 P 与第 3 轮的 J/H 都是同一形态
   （J 三次、H 一次、M 一次）。**教训**：任何 patch 型接线在提交前必须本地跑四种执行方式的
   CPU dry-run（H 的 stub 方案已证明有效），否则 op 全绿也会在 check 短路，白烧 job 窗口。
3. FP8 的四次失败全部在**测量脚本的判据**（coverage 计数 → 解包 → 刀锋容差 → ratio 分母），
   候选本身没有被否定；但也没有任何 e2e 证据。下一轮应把判据简化到「只跑 op + bench，
   判据只用 K 已冻结的相对判据 + 逐位 roundtrip」，其余全部 diagnostic。
4. P 的合并思路有 op 级证据（222/222，含逐位），**值得在下一轮先修 eager 路径再测**；
   它覆盖的 k+v/gate+up/qkv+z 正好是 870 GB/s 以下的那些 shape。
5. M 的结论保留：**GPU 成本中性、p50 被瞬态放大**；tile 参数化（down 17→68 CTA）
   没有把它变成正收益，所以不再投入。

## 当前最佳（未变）

- 集成 commit **`43b0b73a5bab608a926e203cb54a2f5f891de164`**（branch `exp/phase2-integration`）
- 8K：**20.509 ms / 48.76 tok/s / 77.6% SOL（BF16 KV）**，163/163 gate，free 0.423 GB
- 距 80%：BF16 口径差 **0.59 ms**；FP8 验收口径差 **0.94 ms**，且 FP8 无 e2e 证据。
