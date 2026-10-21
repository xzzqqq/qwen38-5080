# AGENTS.md

## 项目

目标：在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 的 bs=1 decode tok/s 推到接近 SOL。

| 文档 | 内容 |
|---|---|
| `docs/project.md` | 背景和组织方式 |
| `docs/roadmap.md` | 目标、benchmark 规格、各里程碑的退出标准 |
| `docs/decisions.md` | 已经做过的决定和理由 |
| `docs/journal/` | 实验记录 |
| `docs/model.md` | 模型结构、checkpoint 布局、NVFP4 约定和已知坑 |

## 命令

GPU 命令都必须在 full access 下运行：bwrap 沙箱看不到 `/dev/nvidia*`。
运行 `llama-bench` / `llama-server` 时不需要设 `LD_LIBRARY_PATH`（rpath 已指向 cuda13 env）。

| 命令 | 作用 | 耗时 |
|---|---|---|
| `scripts/setup_llama_cpp.sh` | 按固定 commit 拉取并编译 llama.cpp | 10–20 分钟 |
| `scripts/run_ar_bench.sh` | M1 AR decode benchmark（depth 512 / 8192，各 5 次） | 约 2 分钟 |
| `scripts/run_server_check.sh` | math / code / chat 三条真实 prompt | 约 30 秒 |
| `make m3-quick` | M3 快档：graph + capture + capacity guard + 显存 + 两套计时冒烟 | 约 5 分钟 |

环境检查：`nvidia-smi --query-gpu=...`、`nvidia-smi -q`、`cuda13/bin/nvcc --version`，
结果存 `results/`。

## 约定

- 渐进式验证：改一点先跑 `make m3-quick`（或对应 op 的单测 / 单场景 `--only`），
  把 loop 做小、fail fast；里程碑关闭前才跑全量（check / memory / bench / profile / sol）。
  产物类问题（缺文件、缺字段、退出码）只需 1 个场景就能发现，不要用全量去等。
- 每做完一次实验，在 `docs/journal/` 记一篇：做了什么、结果、结论。没成功的也要记。
- 给出性能结论时，附上复现命令和 commit。
- 在 kernel 源文件开头简要说明设计，包括数据流和预期瓶颈。
- 汇报时简要说明涉及到的 GPU 机制。
- 文档、注释、汇报用中文；标识符、命令和技术术语保持英文。
- commit message 一律用英文（`type: subject` 格式），不要用中文。

## 结论与验证

- 数值/性能结论必须附可复现命令、commit、checkpoint revision 和结果路径。
- 报告差异前先说明比较对象是否可比；不可比路径的差异不能作为回归或正确性结论。
- 明确区分 gate 和 diagnostic；diagnostic failure 不能写成“通过”。
- 正确性 baseline 未冻结前，不做 kernel 级性能优化。
- 失败实验也要记录，结论和已知限制不能只留在对话里。

## 需要先问的事

- 用 sudo，或改动驱动、CUDA、系统设置
- push、改 main、改写 commit 历史
