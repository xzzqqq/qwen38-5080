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

当前进度：**M1 实机参照完成**；**M2 进行中**，1a 预检、1b checkpoint/manifest、§2 NVFP4Linear 已完成。checkpoint 固定在 revision `5b7a687…`，两个 shard CRC32 已校验；文本 GPU 权重 14.452 GB，8K + BF16 KV 估算余量约 0.68 GB。`NVFP4Linear` 对拍 nrmse ≈ 0.00166、cosine ≈ 0.9999986。下一步拼 eager runner。详见 [`docs/roadmap.md`](docs/roadmap.md) §6.1。

M1 用的 llama.cpp 不纳入版本管理，按固定 commit 拉取和编译：

```bash
scripts/setup_llama_cpp.sh
```

本机环境：RTX 5080 16303 MiB / 驱动 580.105.08 / CUDA 13.0.88（micromamba `cuda13` env）。
