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

当前进度：**M1 实机参照完成**；**M2 进行中**，1a 预检、1b checkpoint/manifest、§2 NVFP4Linear、§3 eager runner 已完成。401 个量化 Linear 已装载；三条真实 prompt 正常结束；8K（7680+512）实测 peak 15.62 GB allocated / 15.86 reserved，free 0.413 GB。开放项：chunked prefill 与单段 logits 尚未一致（cos 0.929）。详见 [`docs/roadmap.md`](docs/roadmap.md) §6.1。

M1 用的 llama.cpp 不纳入版本管理，按固定 commit 拉取和编译：

```bash
scripts/setup_llama_cpp.sh
```

本机环境：RTX 5080 16303 MiB / 驱动 580.105.08 / CUDA 13.0.88（micromamba `cuda13` env）。
