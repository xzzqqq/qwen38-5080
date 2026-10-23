# decode benchmark 使用说明（`bench/e2e/decode.py`）

- 日期：2026-10-22；对应：[M3 plan](plans/m3.md) §2–§3、[SOL](sol.md)
- 适用：M3 之后的 AR decode baseline；Phase 2 换 kernel 后仍然用同一套协议对比。

## 1. 它测什么、不测什么

测：bs=1、固定深度 prompt + 固定 greedy trace 下的 **decode** 速度，分
**模型执行速度**和**含 sampling 的系统速度**两套；同时记录逐步延迟分布和峰值显存。

不测：prefill/TTFT 优化（只记录 TTFT）、batch>1、生成质量、FP8 KV（M3 只实测 BF16）。
数值正确性不在这个脚本里，用 `scripts/m3_check.py`。

## 2. 计时协议（plan §2.2–§2.3）

```text
[ tokenize ] [ prefill chunk=128 ] [ 首 token 采样 ]      ← TTFT，不算 decode
                                  └─ 511 次 decode forward / 511 个输出 token  ← 计时区间
```

| 指标 | 计时范围 | 方法 |
|---|---|---|
| 模型 decode tok/s | 固定输入 buffer 上的单步：decoder + cache/state 更新 + lm_head | 逐步 CUDA events；D2D copy 在 start event 之前；不逐步 sync |
| 系统 decode tok/s | host gather → pinned → H2D → 模型 → argmax → D2H | 墙钟，含空隙与同步 |
| request tok/s | 512 个输出 token ÷（prefill + decode 墙钟） | 另报 TTFT |

- 首 token 来自 prefill logits，归入 TTFT；**最后一个输出 token 不再 forward**，
  所以最终 cache 长度 = depth + 511（512→1023，7680→8191）。
- 每场景先**完整 warmup 一次**（prefill + 511 步模型计时 + prefill + 511 步系统计时），
  再测 `--repeats` 次；每次原地 reset 并重新 prefill。
- 模型速度的 events 包住提交区间，**包含设备空隙**，不等于 kernel 时间之和；
  kernel vs 空隙看 nsys（`make m3-profile`）。
- 场景之间在独立子进程里跑，避免 graph pool / JIT / allocator 历史互相干扰。

## 3. 场景矩阵

| 场景 | execution | depth | capacity | 为什么存在 |
|---|---|---|---|---|
| `eager-512` / `eager-8192` | eager | 512 / 7680 | dynamic | M2 的起点（DynamicCache） |
| `exact-512-1024` | static-exact | 512 | 1024 | 只换存储，不动 attention shape |
| `exact-512-8192` | static-exact | 512 | 8192 | 大 capacity 的显存代价 |
| `exact-8192` | static-exact | 7680 | 8192 | 8K 下的 exact |
| `cap-512-1024` / `cap-8192` | static-cap | 512 | 1024 / 8192 | attention 扫满 capacity 的代价 |
| `cap-512-8192` | static-cap | 512 | 8192 | 短 context 但大 capacity |
| `graph-*` | graph | 同上 | 同上 | 整步 CUDA Graph，去掉提交开销 |

`--list` 会打印每个场景 **每档测量的估算时间**和合计（估算按 2026-10-21 实测的
每步耗时：非 graph 85–86 ms、graph 1024 档 32 ms、8192 档 44 ms）。换模型或换
kernel 后这几个常数要更新（`EST_*`），它们只影响 `--list` 的估算。

## 4. 四档用法（与 AGENTS.md 的反馈阶梯对应）

| 档位 | 命令 | 时间 | 用途 |
|---|---|---|---|
| 单场景冒烟 | `--only graph-512-1024 --repeats 1 --steps 32` | ~15 s | 开发迭代；**`--steps<511` 的结果不能和正式数字比** |
| 定向重跑 | `--only graph-8192,cap-8192` | 3–20 min | 只改了部分路径 |
| 全量 | `make m3-bench`（`--repeats 5 --steps 511`） | ~89 min | 里程碑关闭 |
| 完整关闭 | `scripts/m3_close.sh` | ~135 min | check + memory + bench + profile + sol |

选档位不必记规则：`make verify` 会按 `git diff` 给出该跑哪档。

## 5. 参数

| 参数 | 默认 | 说明与坑 |
|---|---|---|
| `--list` | — | 列场景 + 估算时间后退出，不占 GPU |
| `--only` | `all` | 逗号分隔场景名；名字见 `--list` |
| `--repeats` | 5 | 测量次数（不含 1 次完整 warmup）；plan §2.4 的口径 |
| `--repeats-non-graph` | 同 `--repeats` | 只对非 graph 生效；后续里程碑想用 3 次时用它 |
| `--steps` | 511 | 计时步数。改成 256 等值会改变 mean context，**不可与正式数字比较** |
| `--chunk-size` | 128 | prefill 分块；改它需要 reference 同步改 |
| `--graph-warmup` | 3 | capture 前 side-stream warmup 次数，必须 ≥1 |
| `--fixtures` | `bench/workloads/m3-p0` | 启动时校验 manifest 里的 sha256 |
| `--out` | 自动 `results/m3/<ts>-m3-bench` | 指定后不会再自动换新目录（resume 时建议显式给出同一个） |
| `--resume` | off | 只有源码、运行配置、环境指纹三者都一致才复用；见 §7 |
| `--tag` | `m3-bench` | 自动目录的后缀 |
| `--worker` | — | 内部参数（driver 调子进程），不要手写 |

## 6. 产物与怎么读

```text
results/m3/<run-id>-m3-bench/
├── bench.json            # 全部场景的原始 samples + summary + table
├── summary.md            # 人读的汇总表
├── provenance.json       # commit / dirty / 环境 / source_manifest
├── source_manifest.json  # dirty tree 时逐文件 sha256
├── git_diff.patch        # dirty tree 时的可恢复 patch
└── <scenario>/result.json
```

`bench.json` 每个场景：`spec`、`steps`、`repeats_requested`、`samples[]`、`summary`。

- `samples[i].model.per_step_ms` 长度 = `steps`；`samples[i].tokens.generated` = steps+1。
- `sample.model.model_tok_s` = steps ÷ 逐步 events 之和；`wall_s` 是这一段墙钟。
- `sample.system.system_tok_s` = 闭环墙钟；`tokens.matches_frozen_trace` 只作
  diagnostic（cap/graph 与 eager 的 free-running trace 会因数值差异分叉）。
- `summary.model_tok_s` / `system_tok_s` / `request_tok_s` 是 run 间统计（median/min/max/std）；
  `summary.step_ms` 是 run 内 step 分布（p50/p95），两者不要混。
- `summary.peak_allocated_gb` 只覆盖计时区间；分阶段峰值看 `make m3-memory`。

跑完用 `make check-results` 校验契约（样本数 = repeats、`per_step_ms` 长度 = steps、
`complete=true`、provenance 齐全等）；它只需要秒级、纯 CPU。没有 `provenance.json`
的 legacy 目录（2026-10-22 之前）默认单列为 `[LEGACY]` 不算失败；关闭流程用
`scripts/check_results.py --all --strict`，那时 legacy 也判失败。

## 7. 失败与恢复

- **被中断**：同一 `--out`（或直接 `--resume` 用最近的 `*-m3-bench`）重跑。
  复用条件（三者都一致）：
  1. **源码**：commit + dirty 状态 + patch sha256；
  2. **运行配置**：`steps`、`repeats`（含 `--repeats-non-graph` 生效后的值）、
     `chunk_size`、`graph_warmup`、fixtures manifest 的内容 hash；
  3. **环境指纹**：python / torch / transformers / FlashInfer / CUDA / GPU / driver。
  任何一项变了就重跑对应场景，并打印差异（例如 `运行配置不同（steps: 8 -> 16）`）。
- **最近目录来自别的源码或矩阵配置**：`--resume` 不会覆盖它，会新建目录并打印原因。
- 每个 `result.json` 里的 `run_config` 和 `environment_fingerprint` 就是这两层指纹；
  旧结果没有这两个字段时不能复用（会被判为不可确认）。
- **单个场景 OOM**：先看是不是 `graph-8192`（8K + graph pool 只剩 ~0.12 GB），
  再看 `docs/plans/m3.md` §5.2 的排查顺序；不要擅自改 KV dtype 或把 embedding 搬上 GPU。
- **graph capture 之后长度不对**：看 `docs/decisions.md` 2026-10-21 那条（`past_key_values`
  必须显式传给 graph）。

## 8. 常见任务

```bash
# 我只想知道有哪些场景、各要多久
.venv/bin/python bench/e2e/decode.py --list

# 改了 decode_runner，先冒烟
.venv/bin/python bench/e2e/decode.py --only graph-512-1024 --repeats 1 --steps 32

# 只重跑 8K 的两个 graph 场景，3 次
.venv/bin/python bench/e2e/decode.py --only graph-512-8192,graph-8192 --repeats 3

# 昨天那次跑到一半挂了
.venv/bin/python bench/e2e/decode.py --resume

# M3 口径的全量
make m3-bench

# 完整关闭 + 产物校验
scripts/m3_close.sh
```

多 session 并发时不要直接跑上面这些命令：把 check/bench 写进
[实验队列](experiments.md) 的请求，由唯一的 `serve` 串行执行。队列会校验产物里的
`steps`、`repeats`、场景、fixtures 和 commit，避免把旧的 32 步 / 1 次结果当成新的
511 步 / 5 次完成：

```bash
.venv/bin/python scripts/experiment.py submit --request .tmp/experiments/req.json
.venv/bin/python scripts/experiment.py serve --drain   # GPU 主机上
.venv/bin/python scripts/experiment.py result <job-id>
```

## 9. 与 plan 的关系

- **协议固定**（不能为省时间改）：511 步、两趟计时、首 token 归 TTFT、
  完整 warmup、场景独立进程、5 次重复（M3 口径）。
- **可裁剪**（但要在记录里写明）：跑哪些场景、重复次数（`--repeats-non-graph`）、
  `--steps`（仅开发）、`--resume`。
- 任何裁剪都会让数字与正式表不可比，必须写进 journal 或 decisions，并在结果
  文件名/tag 上体现。
