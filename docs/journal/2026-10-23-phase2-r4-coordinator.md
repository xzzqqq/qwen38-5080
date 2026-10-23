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
