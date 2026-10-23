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
| `docs/bench.md` | decode benchmark 使用说明：场景、参数、产物、失败恢复 |
| `docs/experiments.md` | GPU 实验队列：submit/status/result/wait/cancel/serve、请求格式、产物契约、失败恢复 |

## 命令

GPU 命令都必须在 full access 下运行：bwrap 沙箱看不到 `/dev/nvidia*`。
运行 `llama-bench` / `llama-server` 时不需要设 `LD_LIBRARY_PATH`（rpath 已指向 cuda13 env）。

| 命令 | 作用 | 耗时 |
|---|---|---|
| `scripts/setup_llama_cpp.sh` | 按固定 commit 拉取并编译 llama.cpp | 10–20 分钟 |
| `scripts/run_ar_bench.sh` | M1 AR decode benchmark（depth 512 / 8192，各 5 次） | 约 2 分钟 |
| `scripts/run_server_check.sh` | math / code / chat 三条真实 prompt | 约 30 秒 |
| `make m3-quick` | M3 快档：graph + capture + capacity guard + 显存 + 两套计时冒烟 | 约 40 秒 |
| `make verify` | 按改动文件推荐最小验证档位（`make verify-run` 执行） | < 5 秒 |
| `make check-results` | 校验 `results/` 下 run dir 的产物契约 | < 1 秒 |
| `make m3-bench` | M3 全量性能矩阵（11 场景 × 5 次 × 511 步） | 约 89 分钟 |
| `scripts/m3_close.sh` | 完整关闭：check + memory + bench + profile + sol（`SKIP=` 可裁剪，`--resume` 续跑） | 约 135 分钟 |
| `.venv/bin/python scripts/experiment.py serve` | GPU 实验队列的唯一执行器（full access；常驻） | 常驻 |
| `.venv/bin/python scripts/experiment.py submit --request PATH` | 提交实验 job（不等待 GPU） | < 1 秒 |
| `.venv/bin/python scripts/experiment.py status/result/wait/cancel` | 查询、等待、取消 job（查询输出 JSON） | 查询 < 1 秒；`wait` 阻塞 |
| `make experiment-test` | 实验工具 CPU 测试（队列/worktree/超时/产物校验） | 约 43 秒 |

`make m3-bench` / `scripts/m3_close.sh` 只用于里程碑关闭和本机单 session 校准；
会被引用、比较或跨 session 共享的 GPU 实验走 `scripts/experiment.py` 队列。

环境检查：`nvidia-smi --query-gpu=...`、`nvidia-smi -q`、`cuda13/bin/nvcc --version`，
结果存 `results/`。

## 约定

- 需要 GPU 的实验（check / bench / memory / profile，且结果会被引用、比较或跨
  session 共享）一律经 `scripts/experiment.py` 队列串行执行：`submit` 入队，GPU
  主机上的 `serve` 消费，结果用 `result` / `wait` 读取。只有本机单 session 的开发
  冒烟才直接用 `make verify` / `make m3-quick` 的最小档位；不要和其他 session 的
  job 并发占卡。请求格式、产物契约和 halt 恢复见 `docs/experiments.md`，Coordinator
  的派发流程见 `docs/optimization-workflow.md`。
- 渐进式验证：改一点先跑 `make verify` 给出的档位（或对应 op 的单测 / 单场景
  `--only`），把 loop 做小、fail fast；里程碑关闭前才跑全量
  （`make verify-close`：check / memory / bench / profile / sol）。
  产物类问题（缺文件、缺字段、退出码）只需 1 个场景就能发现，不要用全量去等。
- 每个 plan 必须有“反馈阶梯”表（档位 / 命令 / 耗时 / 证明什么，模板见
  `docs/plans/_template.md`）；DoD 要求新增路径至少有一个 ≤5 分钟的档位能覆盖它。
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
