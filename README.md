# qwen38-5080

在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 的 bs=1 decode 速度推到接近 SOL。

| 文档 | 内容 |
|---|---|
| [`docs/project.md`](docs/project.md) | 背景、范围、repo 结构和运行环境 |
| [`docs/roadmap.md`](docs/roadmap.md) | 目标、benchmark 规格、SOL 数字、里程碑 |
| [`docs/plans/m1.md`](docs/plans/m1.md) | M1 执行清单 |
| [`docs/plans/m2.md`](docs/plans/m2.md) | M2 执行清单 |
| [`docs/plans/m3.md`](docs/plans/m3.md) | M3 执行清单（decode baseline） |
| [`docs/sol.md`](docs/sol.md) | 按实际字节的 SOL 模型与每步公式 |
| [`docs/bench.md`](docs/bench.md) | decode benchmark 用法：场景、参数、产物、失败恢复 |
| [`docs/model.md`](docs/model.md) | 模型结构、checkpoint、NVFP4 约定、manifest 和显存账 |
| [`docs/journal/`](docs/journal/) | 实验记录 |

当前进度：**M1**、**M2**、**M3** 完成（M3：2026-10-21）。

- 四执行方式（eager / static-exact / static-cap / CUDA Graph）在同一 checkpoint、
  dtype 和 workload 下建立 baseline：graph **31.41 tok/s**（depth 512 / cap 1024，
  48.5% SOL）、**22.53 tok/s**（8K / cap 8192，35.9% SOL）；eager 与 static cache
  都是 11.5–11.9 tok/s，瓶颈是每步 7242–7290 个 kernel 的提交开销。
- 8K + graph 峰值 15.722 GB、free 0.119 GB；graph pool 只多 34 MB。
- 227 个正确性 gate 全通过（四模式 + 独立 padded reference + reset/边界 + smoke），
  6/6 run dir 通过产物契约的 `--strict` 校验。

详见 [M3 journal](docs/journal/2026-10-21-m3-decode-baseline.md)、
[`docs/roadmap.md`](docs/roadmap.md) §6.1 和 [`docs/sol.md`](docs/sol.md)。
下一步 Phase 2：按时间线顺序优化 NVFP4 GEMV、8K 的 GQA 展开和算子融合。

M1 用的 llama.cpp 不纳入版本管理，按固定 commit 拉取和编译：

```bash
scripts/setup_llama_cpp.sh
```

本机环境：RTX 5080 16303 MiB / 驱动 580.105.08 / CUDA 13.0.88（micromamba `cuda13` env）。
