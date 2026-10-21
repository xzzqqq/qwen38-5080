# qwen38-5080

在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 的 bs=1 decode 速度推到接近 SOL。

| 文档 | 内容 |
|---|---|
| [`docs/project.md`](docs/project.md) | 背景、范围、repo 结构和运行环境 |
| [`docs/roadmap.md`](docs/roadmap.md) | 目标、benchmark 规格、SOL 数字、里程碑 |
| [`docs/plans/m1.md`](docs/plans/m1.md) | M1 执行清单 |
| [`docs/plans/m2.md`](docs/plans/m2.md) | M2 执行清单 |
| [`docs/model.md`](docs/model.md) | 模型结构、checkpoint、NVFP4 约定、manifest 和显存账 |
| [`docs/journal/`](docs/journal/) | 实验记录 |

当前进度：**M1 实机参照完成**；**M2 完成（2026-10-21）**。401 个量化 Linear 已装载；三条 prompt 在 `max_new_tokens=2048` 下正常 EOS；runner 数值验收（3 prompt ×16 步）通过；512/7680 容量检查通过，8K peak 15.62 GB allocated / 15.86 reserved，free 0.413 GB。下一步 M3：static cache、CUDA Graph 和性能 baseline。详见 [`docs/roadmap.md`](docs/roadmap.md) §6.1 与 [M2 关闭记录](docs/journal/2026-10-21-m2-close.md)。

M1 用的 llama.cpp 不纳入版本管理，按固定 commit 拉取和编译：

```bash
scripts/setup_llama_cpp.sh
```

本机环境：RTX 5080 16303 MiB / 驱动 580.105.08 / CUDA 13.0.88（micromamba `cuda13` env）。
