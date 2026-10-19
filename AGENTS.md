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

搭好脚手架后再填。需要列出环境检查、编译、测试、benchmark、profiling 的命令，以及各自大概的耗时。

## 约定

- 每做完一次实验，在 `docs/journal/` 记一篇：做了什么、结果、结论。没成功的也要记。
- 给出性能结论时，附上复现命令和 commit。
- 在 kernel 源文件开头简要说明设计，包括数据流和预期瓶颈。
- 汇报时简要说明涉及到的 GPU 机制。
- 文档、注释、汇报用中文；标识符、命令和技术术语保持英文。

## 需要先问的事

- 用 sudo，或改动驱动、CUDA、系统设置
- push、改 main、改写 commit 历史
