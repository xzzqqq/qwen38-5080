# Phase 2 第 5 轮 Coordinator 日志（收口轮）

- 日期：2026-10-23
- 起点：第 4 轮集成 commit `918d5d0`（正式 511×5：19.787 / 20.323 / 20.324 ms，
  系统 50.47 / 49.14 / 49.14 tok/s，%SOL 77.9 / 75.8 / 78.3，163/163 gate，8K free 0.438 GB）

## 派发与结果

| 任务 | session | job | 结果 | 判定 |
|---|---|---|---|---|
| Q FP8 收尾（判据简化 + 只跑 op/bench） | `phase2-fp8kv` | `job-000059-832633dc` | succeeded，op 36/36；graph-8192 **20.509 ms / 48.69 tok/s**、graph-512-1024 20.199 / 49.40；8K free **0.686 GB** | **不接主线**：换 0.25 GB 显存、付 0.18 ms 速度 → FP8 是容量/验收格式选项，不是性能优化 |
| R GEMV 已到顶的复核 + 融合 norm kernel grid=1→5 + `1/input_scale` 预计算 | `phase2-tactics` | `job-000060-139b253e` | succeeded，op 482/482、check 139/139；graph-8192 **19.817 ms / 50.37 tok/s**、graph-512-1024 **19.288 / 51.77** | **采纳**：−0.507 / −0.499 ms |

- GEMV 方向由数据关闭：合并后 kv 506→695、gate/up 884→**897**（峰值 93.4%）、qkv/z 857→886 GB/s，
  `merged_saves 0.240 ms`；官方 `mm_fp4` 无 tile 参数（tile 只能经私有 module），不再投入。
- R 的收益来源：`_norm_quant_kernel` 原来 grid=1、单 program 串行处理 5 个 chunk
  （3.98 µs × 128 次/步 = 0.509 ms）；改成每 chunk 一个 program（各自冗余重算同一段 FP32
  RMSNorm 树 → rstd 逐位相同、落点互不重叠），加上 `1/input_scale` 预计算去掉 81 次/步的
  1-element reciprocal。这正是「kernel 数已降到 ~3200、但单 kernel 延迟仍高」的那类问题。
- R 坚持多加一步 `phase2_normquant_check.py`（128 个真实 norm 对独立 reference 的逐字节
  gate）作为 kernel 改动的独立正确性门 —— `numeric-512` 只做同实现内对拍，改错但确定性时
  163 gate 会照样全绿。这是本轮的流程收获。

## 32×1 口径的最终数字（正式 511×5 由 `phase2-final` job 复测，见下节）

| 场景 | 第 4 轮 | **第 5 轮（32×1）** | 系统 tok/s | %SOL |
|---|---|---|---|---|
| graph-512-1024 | 19.787 ms | **19.288 ms** | 51.77 | ~79.9% |
| graph-8192 | 20.324 ms | **19.817 ms** | **50.37** | **80.3%** |

## 收口状态

- 集成 commit：**`7c95b0eaef07462c01fc1eded7ce3e936109b5f2`**（branch `exp/phase2-integration`）
  = 第 4 轮 `918d5d0` + R（norm kernel grid=5 + `global_divisor` 预计算）。
- 正式口径复测：`phase2-final` job（check 3 场景 + 三场景 511×5 + 8K 显存），结果见
  后续追加。
- **未完成项（明确记录，不算通过）**：
  1. **FP8 主线未达到 80%**：FP8 实测比 BF16 慢 0.18 ms（转换受限），按 FP8 SOL（63.8）
     当前约 78.4%；plan §7.2 要求 FP8 达到 ≥80% 实际逻辑 SOL → **Phase 2 未关闭**。
  2. `m3_sol` 字节账未重算（%SOL 仍用 M3 冻结的 B_mean 口径），完整 `m3-check`（5 场景 227 gate）
     与 P1/P2 真实 workload 未跑。
  3. FP8 的 smoke-graph 与 8K 显存生命周期仍未测（Q 只跑了 op+bench）。
- 下一轮入口：FP8 口径的 0.4 ms 缺口（~1000 个模型级 tiny elementwise、graph 间隙），
  以及接受/拒绝「FP8 用 0.18 ms 换 0.25 GB」这一取舍需要人工决策。
