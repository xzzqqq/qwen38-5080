# Qwen3.8-27B on RTX 5080：项目背景与组织形式

- 日期：2026-10-19
- 里程碑和进度见 `docs/roadmap.md`

## 1. 目标

在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 单请求（bs=1）的 decode 速度推到接近 SOL。

- **范围**：bs=1 的 decode tok/s；context ≤ 8K；格式固定（见 roadmap §3）。
- **不做**：
  - batch>1 的吞吐；
  - 更长的 context；
  - prefill / TTFT 优化（只记录数据）；
  - 质量目标（用"格式固定 + 实现正确"代替）；
  - 支持其他模型或 GPU。

## 2. 关键事实

| 项 | 内容 |
|---|---|
| 模型 | 共 64 层：48 层 Gated DeltaNet（线性 attention，带递归 state），16 层 gated attention（GQA 24/4，head_dim 256）。hidden 5120，FFN 17408，vocab 248320 |
| GPU | 带宽 960 GB/s；显存 16 GiB（= 17.18 GB），**空闲可用 16.61 GB**（15841 MiB，驱动保留 463 MiB）；有原生 FP4 tensor core（`mxf4nvf4`）；BF16 / FP8 在 FP32 累加时只有半速 |
| 瓶颈 | bs=1 decode 每个 token 读约 15 GB、算 51 GFLOP，算术强度比机器平衡点低约 264 倍。因此 速度 ≈ 带宽 ÷ 每步要读的字节数 |
| SOL | AR：64 tok/s（FP8 KV）/ 63（BF16 KV）；DFlash2 spec decode：236–314 tok/s |
| 现有推理栈（外推） | AR：46–50 tok/s；DFlash2：125–170 tok/s |
| **M3 baseline（实测）** | graph：**31.41 tok/s**（depth 512 / cap 1024，48.5% SOL）、**22.53 tok/s**（8K / cap 8192，35.9% SOL）；eager 与 static cache 都是 11.3–12.0 tok/s（launch-bound）。见 `docs/journal/2026-10-21-m3-decode-baseline.md` |
| AR 实机参照（M1，外部） | IQ4_XS：55.90 / 54.19 tok/s（depth 512 / 8192），逻辑带宽约 80%；GPU 权重 13.35 GB，比 NVFP4 少约 1.06 GB，不能直接外推 |
| 显存 | baseline 8K 实测 15.62 GB allocated / 15.86 reserved，free 0.413 GB；**M3 加 graph 后 prefill peak 15.722 GB、reserved 16.054 GB、free 0.119 GB**（graph pool 只多 34 MB）。带上 DFlash2 估算 16.39–16.74 GB，容量更紧，是否够用待 Phase 3 验证。spec decode 回滚时不能保存整份 state 快照，否则要多占 1.21 GB |

数字的推导见 `docs/roadmap.md` §4。

## 3. 组织形式

### 3.1 Repo 结构

```
qwen38-5080/
├── AGENTS.md          # agent 规则：项目入口、命令、约定、需要先问的事
├── CLAUDE.md          # 只有一行：@AGENTS.md
├── Makefile           # 所有入口命令
├── requirements.txt   # 锁定版本的依赖
├── docs/
│   ├── project.md     # 本文件
│   ├── roadmap.md     # 目标、约束、SOL 数字、关键决定、里程碑和 Status
│   ├── sol.md         # SOL 模型、每步字节数公式
│   ├── model.md       # 模型结构、checkpoint 的 tensor 布局、NVFP4 约定、已知坑
│   ├── decisions.md   # 已做的决定：日期、决定、理由、被否决的选项
│   ├── plans/         # 各里程碑的执行清单
│   ├── journal/       # 实验记录：YYYY-MM-DD-<topic>.md
│   └── history/       # 早期的分析和 review
├── runner/
│   ├── loader/        # 权重装载、NVFP4 repack
│   ├── model/         # 模型骨架
│   ├── ops/<op>/      # 参考实现、若干优化实现、字节模型
│   ├── kernels/       # CUDA 源码
│   └── spec/          # MTP、DFlash2、回滚
├── tests/
│   ├── ops/           # 每个 op 和参考实现对拍
│   ├── e2e/           # 固定 prompt 的 logits / token 对拍
│   └── golden/        # 参考输出
├── bench/
│   ├── workloads/     # 固定的 prompt 集
│   ├── micro/         # 单个 op：耗时、实际带宽、%SOL
│   └── e2e/           # decode tok/s、τ、每步时间拆分
├── results/           # benchmark 记录
└── scripts/           # 环境报告、nsys/ncu 的封装和摘要
```

### 3.2 机制

| 机制 | 做法 |
|---|---|
| op 注册表 | 每个 op 一个目录，放参考实现、若干优化实现、字节模型、单测和 microbench。运行时可以选择用哪个实现，做 A/B 对比只需要换参数 |
| 两档 harness | 快档（< 2 分钟）：跑改动涉及的 op 的单测和 microbench，外加少量 prompt 的 logits 对拍。慢档（10–30 分钟）：跑完整 workload |
| 结果记录 | 每次 benchmark 往 `results/` 追加一条记录，包含 commit、环境和指标 |
| repo 内记忆 | 跨 session 需要的上下文都放在 repo 里：roadmap 的 Status、decisions、journal（失败的实验也记）、results |

### 3.3 工作方式

- 任务从 roadmap 里取，以对应的退出标准作为验收标准。
- 实验结论写进 journal，决定写进 decisions，进度更新到 roadmap 的 Status。
- AGENTS.md 只放项目入口、命令、约定和需要先问的事。新规则等出现具体问题时再加。

### 3.4 运行环境

- agent 运行在装有 5080 的 Linux 机器上，通过 SSH + tmux 使用。
- 显示器接核显，或者不接，避免桌面占用显存。
- 工具链：
  - 驱动 R570 及以上；
  - CUDA 12.8 及以上，编译目标 `sm_120a`；
  - Nsight Systems / Nsight Compute；
  - compute-sanitizer。
- AGENTS.md 是唯一的规则文件。Claude Code 通过 `CLAUDE.md`（内容为 `@AGENTS.md`）导入它。

## 4. 文档地图

| 原始文件 | 现在的位置 | 内容 |
|---|---|---|
| `qwen38_27b_rtx5080_project.md` | `docs/project.md` | 本文件 |
| `qwen38_27b_rtx5080_roadmap.md` | `docs/roadmap.md` | 目标、约束、SOL 数字、关键决定、里程碑 |
| `qwen38_27b_rtx5080_m1_plan.md` | `docs/plans/m1.md` | M1 的执行清单 |
| `qwen38_27b_rtx5080_agents_draft.md` | `AGENTS.md` | agent 规则（草稿） |
| — | `docs/sol.md` | 已写：按运行时唯一 storage 的实际字节的 SOL 模型与每步公式（`make m3-sol` 生成） |
| — | `docs/model.md` | 已写：模型结构、checkpoint 的 tensor 布局、NVFP4 约定 |
| — | `docs/decisions.md` | 已写：M1–M3 的决定、理由和被否决的选项 |
| — | `docs/validation.md` | 数值/容量验收经验与参考 |

`docs/model.md` 需要写进去的已知坑：

- FP4 的 nibble 顺序。
- scale 要从 checkpoint 的线性布局转换成 kernel 需要的布局。
- `alpha = input_scale × weight_scale_2`。
- prefill 只需要最后一个位置的 logits。否则 2K 的 prompt 光 logits 就要占 1 GB。
- persistent kernel 一旦死锁，会卡住整张 GPU。
- `ncu` 报 `ERR_NVGPUCTRPERM`，说明缺少性能计数器的权限。
