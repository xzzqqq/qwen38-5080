# GPU 实验队列（`scripts/experiment.py`）

- 日期：2026-10-23
- 对应：[experiment-tools spec](experiment-tools-spec.md)、[optimization workflow](optimization-workflow.md)、
  [benchmark 协议](bench.md)、[实现记录](journal/2026-10-23-experiment-tools.md)

## 1. 它解决什么问题

单张 5080 只有 16 GiB，8K + graph 的峰值只剩约 0.12 GB。多个 Explorer 直接跑
GPU 命令会互相抢显存、抢带宽，而且结果没有统一的 commit / 配置记录。这个工具把
**一次提交的实验**做成一个 job：

- 只允许一个 `serve` 执行进程，队列 FIFO、串行使用 GPU；
- 每个 job 从提交的 commit 建 detached worktree 执行，后续编辑不影响结果；
- 每个步骤都必须退出码 0 **且**产物通过结构化校验，才跑下一步；
- 结果、日志、环境和通知情况都落盘，可重复查询，不触发重跑；
- 请求里写死 `steps / repeats / --only / fixtures`，校验器逐项比对产物，旧的
  32 步 / 1 次结果不可能被当成新的 511 步 / 5 次完成。

## 2. 目录、角色与状态

```text
results/experiments/                 # 默认在**主仓库**下，不进 git
├── queue.json                       # 队列与 job 记录（原子写 + flock）
├── queue.lock / serve.lock / gpu.lock
└── <job-id>/
    ├── request.json                 # 原始请求（提交后 0444）
    ├── execution.json               # 时间、命令、环境、进程、worktree、逐步骤记录
    ├── summary.json                 # 短结果（result / wait 返回它）
    ├── logs/                        # 每步 stdout/stderr、notify.log
    └── output/                      # `{output}` 展开处，现有脚本的原始产物
.tmp/experiments/worktrees/<job-id>/ # detached worktree（成功后退删，失败保留）
```

- **root**：默认取 `git rev-parse --git-common-dir` 的父目录，也就是主 worktree。
  从 Explorer 的 worktree 里提交，写到的仍是同一个主队列。
- **Explorer 的 worktree 必须基于包含本工具的 commit**，否则 worktree 里没有
  `scripts/experiment.py`；临时可以从主仓库调用
  `/path/to/main/.venv/bin/python /path/to/main/scripts/experiment.py ...`。
- **`serve` 必须在 GPU 主机的 full access 下运行**（bwrap 看不到 `/dev/nvidia*`）。
  查询命令（submit/status/result/wait/cancel）不需要 GPU。

| 角色 | 做什么 |
|---|---|
| Explorer | 实现候选 → 本地 CPU 检查 → 在分支 commit → 写请求 → `submit` → 结束 turn；结果到了再 `result` |
| Coordinator | 启动/复用 `serve`，用 `status` 核对，后台跑 `wait` 收摘要并转发，处理 halt |
| GPU 主机 | 唯一 `serve`，按 FIFO 串行执行，不决定下一次实验 |

## 3. 快速开始

```bash
# 1) GPU 主机：启动唯一执行进程（可放 tmux/systemd 后台）
.venv/bin/python scripts/experiment.py serve

# 2) Explorer：提交（不等待 GPU）
.venv/bin/python scripts/experiment.py submit --request .tmp/experiments/req.json

# 3) 查询 / 读结果
.venv/bin/python scripts/experiment.py status
.venv/bin/python scripts/experiment.py status --session explorer-attention
.venv/bin/python scripts/experiment.py result <job-id>

# 4) Coordinator：后台跑 wait（结束后由 harness 通知），或前台等
.venv/bin/python scripts/experiment.py wait <job-id> [...]

# 5) 取消
.venv/bin/python scripts/experiment.py cancel <job-id>
```

常用 make 入口：

| 命令 | 作用 |
|---|---|
| `make experiment-serve` | 启动唯一执行器 |
| `make experiment-status` | 查看队列 |
| `make experiment-test` | CPU 测试（`tests/experiment`，约 35 s） |

`wait` 默认一直等；`--timeout S` 只用于显式上限。队列停用、执行器不在或心跳超过
`--stale-after`（默认 60 s）时，`wait` 返回错误码 4 和当前状态，不会静默挂住。

## 4. 请求格式

```json
{
  "session_id": "explorer-attention",
  "commit": "34303182db95fe21be4993e2cc7805b4ff0246a8",
  "timeout_s": 300,
  "steps": [
    {
      "kind": "check",
      "argv": [".venv/bin/python", "scripts/m3_check.py", "--only", "graph-1024",
               "--steps", "16", "--out", "{output}/check"],
      "result": "{output}/check/check.json"
    },
    {
      "kind": "bench",
      "argv": [".venv/bin/python", "bench/e2e/decode.py", "--only", "graph-512-1024",
               "--steps", "32", "--repeats", "1", "--out", "{output}/bench"],
      "result": "{output}/bench/bench.json"
    }
  ]
}
```

| 字段 | 规则 |
|---|---|
| `session_id` | 提交者的原生 session 标识；1–200 字符、无控制字符；**每个 session 最多一个未完成 job** |
| `commit` | 仓库里存在的**完整 40 位** SHA；提交时解析并检查脚本在该 commit 里存在 |
| `timeout_s` | 正数，≤ 86400；从 job 开始执行（含 worktree/准备）计时，不含排队 |
| `steps` | 非空、有序；最多 32 个 |
| `kind` | `check / bench / memory / profile`，必须与脚本匹配 |
| `argv` | 数组，直接 `subprocess` 执行，不经 shell；`--out` 必须指向 `{output}` 下的目录。**argv 里指向本仓库任何 worktree 的绝对路径会被拒绝**，必须用仓库相对路径（否则会绕过 job worktree、读到主工作区的未提交修改） |
| `result` | 该步骤必须产出的结构化产物；必须以 `{output}/` 开头、`.json` 结尾，且在 `--out` 目录内 |

提交期就会拒绝（错误码 2）：未知字段、commit 不存在/不是完整 SHA、未注册脚本、
kind 与脚本不匹配、缺 `--out`、result 不在 output 内、`--only` 场景名不存在、
`--steps/--repeats/...` 非正数、重复 result、`--resume / --worker / --list`。

已注册脚本（新增脚本必须先补 `runner/experiment/artifacts.py` 的校验函数并注册）：

| 脚本 | kind | 产物契约 |
|---|---|---|
| `scripts/m3_check.py` | check | `check.json` + 每场景 `report.json`：gate 非空、coverage gate、全部通过、provenance commit 一致 |
| `bench/e2e/decode.py` | bench | `bench.json` + `provenance.json`：`complete=true`、样本数=repeats、逐步数据长度=steps、配置/环境指纹一致 |
| `scripts/m3_memory.py` | memory | `memory.json` + `provenance.json`：必需 stage、reset 确定性、无跨生命周期增长 |
| `scripts/m3_profile.sh` | profile | `timeline_summary.json` + `summary.md` + 每 case 的 nsys trace / gpu_trace.csv / meta.json |

提交后另有（错误码 3）：同一 session 已有未完成 job，返回 `existing_job_id`。

## 5. 执行语义

- **FIFO**：`serve` 每次原子地取队首 queued job 并标 running；同时最多一个 job
  使用 GPU。job 期间持有 `gpu.lock`。
- **worktree**：`git worktree add --detach <commit>`；`<worktree>/.venv` 和
  `<worktree>/models` 只是指向主仓库的符号链接，不复制权重；worktree 必须 clean。
- **commit 隔离**：argv 只允许仓库相对路径；提交时拒绝指向任一 worktree 的绝对
  路径，执行器还会二次检查绝对路径和 `..` 逃逸（相对路径按 worktree 归一化）。
  `request.json` 入队后按 sha256 + commit 再核对一次；产物 provenance 必须
  `dirty=false`，否则视为不是在提交的 clean worktree 里跑的。
- **环境**：复制 serve 的环境，再补 `CUDA_HOME / VENV / CPATH / LIBRARY_PATH /
  PATH / PYTHONPATH=<worktree>` 和 `FLASHINFER_WORKSPACE_BASE=<root>/.tmp/flashinfer`，
  并注入 `EXPERIMENT_JOB_ID / EXPERIMENT_SESSION_ID / EXPERIMENT_COMMIT /
  EXPERIMENT_OUTPUT_DIR / EXPERIMENT_STATE_DIR`。实际值写进 `execution.json`。
- **进程组**：每个 job 有一个 `sleep` keeper 做组长，所有步骤 join 同一个进程组；
  超时/取消先 SIGTERM、10 s 后 SIGKILL，确认组内非僵尸进程消失后才释放 GPU 锁。
  步骤正常结束后会清理同组残留后台进程并记录 `stray_pids_killed`。
- **超时**：`timeout_s` 是 job 级 wall time；到点杀当前步骤，状态 `timed_out`，
  后续步骤不执行，队列继续跑下一 job。
- **取消**：queued 立即变 `cancelled`（不执行、写 summary）；running 只置
  `cancel_requested=true`，实际退出前仍显示 `running`，执行器在 1 s 内终止进程组。
  执行器不在时，`cancel` 会按记录的 pid/pgid 直接杀孤儿进程组并确认退出。
- **GPU 守卫**：每次取 job 前用 `nvidia-smi` 查计算进程；**查询失败/不可用也直接
  停队列**（fail closed：无法确认 GPU 可用就不跑）。发现外部进程先等
  `GPU_DRAIN_S=10 s`（刚被杀掉的 CUDA context 释放 12 GB 有秒级延迟），持续占用
  才停队列。`--no-gpu-guard` 只用于 CPU fake 命令验证。
- **停队列（halt）**：外部 GPU 进程、`nvidia-smi` 不可用、`gpu.lock` 被占、进程组
  无法确认退出（**即使 job 本身已经 failed**）、执行器重启时发现 keeper 或进程组
  任一成员仍存活，都会写 `halted={reason, detail, at}` 并停止消费。确认设备空闲、
  进程已退出后，用 `serve --clear-halt` 恢复。
- **执行器重启**：核对记录的 pid + `/proc/<pid>/stat` starttime。进程仍存活 →
  停队列（不接管）；已退出且 summary 完整 → 采纳原状态并标 `recovered`；已退出但
  无完整结果 → `interrupted`。failed/timed_out/interrupted 都**不自动重跑**。
- **失败短路**：任一步骤退出码非 0、超时、产物缺失/配置不符/gate 失败，都停止后续
  步骤；错误记录首个失败步骤、原因、类别和日志路径。

## 6. 结果与校验

`summary.json`（`result` / `wait` 返回它）：

| 字段 | 内容 |
|---|---|
| `job_id / session_id / state` | 身份与执行状态 |
| `commit / commands` | 实际代码版本、展开后的执行命令 |
| `correctness` | `passed / failed / not_checked` |
| `metrics` | `check`（gate 计数、失败项、diagnostic）、`bench`（每场景 median/min/max、steps、repeats、计时口径）、`memory`（peaks/growth）、`profile`（每步 kernel 数/时长） |
| `queue_s / elapsed_s` | 排队与执行耗时（秒） |
| `error` | 首个失败步骤、原因、类别、日志路径；成功为 `null` |
| `output_dir / artifacts` | `output_dir` 是 job 目录；`artifacts.output` 是 `{output}`，另有 `artifacts.logs`、`logs_files`；`request_path / execution_path / summary_path` 是文件路径 |
| `checkpoint / environment` | checkpoint revision、执行器环境与 GPU 快照 |
| `notification` | 投递状态、命令、返回码、日志 |
| `steps` | 每步 argv、退出码、耗时、pid/pgid、产物 sha256、校验结果 |

正确性口径：

| 情况 | correctness |
|---|---|
| 请求里的**全部** check 步骤都执行完，且 gate + coverage 全部通过 | `passed` |
| 有 gate 失败（即使进程退出码非 0） | `failed` |
| 没有 check、有 check 未执行/未执行完、缺产物/配置不符/超时/取消 | `not_checked` |

覆盖判断用的是**请求的步骤集合**：只跑了第一个 check、后面的 check 根本没跑，
即使已跑的 check 全通过，也不能算 `passed`。

产物校验还会重算 fixtures manifest hash 并与记录的 `fixtures_sha256`/请求路径
比对；执行期 worktree 还在，重算失败直接判 failed。成功 job 的 worktree 删除后，
事后复查只能依赖记录值与跨场景一致性。

`diagnostic` 只记录、不参与判定，不因为它失败改判已通过的 gate。

产物契约与 `scripts/check_results.py` 一致：实测把 job 的 `output/check` 和
`output/bench` 直接交给 `check_results.py` 是 `[OK]`（见 journal）。

## 7. 完成通知

- 每个 job 终止后尝试一次通知，**按 job 去重**；失败不改变实验结果，也不重跑。
- 配置 `--notify-cmd` 或 `DSH_EXPERIMENT_NOTIFY_CMD` 时，用 `/bin/sh -c` 执行，
  注入：`JOB_ID / SESSION_ID / JOB_STATE / CORRECTNESS / OUTPUT_DIR /
  SUMMARY_PATH / SUMMARY_JSON / SUMMARY_TEXT`；stdout/stderr 写
  `logs/notify.log`，返回码与耗时记入 `notification`。
- 不配置时 `notification.state=pending`，由 Coordinator 用 `wait`/`result` 收取并
  转发给原 session；这正是 spec §6 的兜底路径，也让工具不依赖特定 harness。
- 在 DSH 里推荐：Coordinator 用**后台 job** 跑 `experiment.py wait <job...>`，
  完成通知由 harness 送回 Coordinator，再由它 message / follow-up 原 Explorer。
  不建议自动嵌套 `dsh headless --session-id`（会额外起 turn、可能重复消耗预算）。

## 8. 失败恢复 cheatsheet

```bash
# 看队列和 halt 原因
.venv/bin/python scripts/experiment.py status

# 结果/日志
.venv/bin/python scripts/experiment.py result <job-id>
ls results/experiments/<job-id>/logs/

# 队列被停：先确认没有外部 GPU 进程、孤儿进程已退出
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
ps -o pid,pgid,cmd -p <record.pid>
# 孤儿 job 用 cancel（没有 serve 时会直接杀记录下来的进程组）
.venv/bin/python scripts/experiment.py cancel <job-id>
# 再恢复
.venv/bin/python scripts/experiment.py serve --clear-halt
```

不要手动改 `queue.json`；它坏了（JSON 解析失败）时工具会报错而不是静默重置。

## 9. 反馈阶梯

| 档位 | 命令 | 耗时 | 证明什么 |
|---|---|---|---|
| 静态 | `python -m py_compile runner/experiment/*.py scripts/experiment.py` | 秒级 | 语法/import |
| CPU 队列测试 | `make experiment-test` | ~43 s | 请求校验、worktree/绝对路径隔离、FIFO/不重叠、失败短路、超时/取消杀进程组、重启核对、通知去重、产物校验正反例 |
| 单 job 冒烟（GPU） | `submit` + `serve --once` 一个 `graph-1024 --steps 16` check | 1–2 min（含 load） | 真实 worktree/env/产物/GPU 守卫 |
| check→bench（GPU） | 本文 §4 的请求 | ~2.5 min | spec §7 的 5080 验收路径 |
| 全量实验 | Explorer 自己按 `make verify` 选档 | 15–95 min | 具体候选性能 |

## 10. 已验证（2026-10-23）

- 实现提交：`cd16e2f`（`feat: add a queued GPU experiment runner`）；审查修复提交：
  `cd95142`（`fix: harden experiment runner review findings`）。
- CPU：`pytest tests/experiment -q` → **68 passed**，覆盖 spec §7 的九类场景
  （并发 FIFO/不重叠、改动 worktree、check 失败/缺产物、旧配置、超时/取消、
  重启、通知、重复查询、纯 bench/profile）。
- GPU（check/bench 在 commit `3430318`；profile/memory 在 `4891030`；checkpoint `5b7a687`）：
  - `job-000001-e6dcf12d`：`graph-1024 --steps 16` check + `graph-512-1024
    --steps 32 --repeats 1` bench；**24/24 gate 通过**，1 个 diagnostic 差异；
    bench **31.40 tok/s**（与 M3 baseline 31.41 一致）；correctness=passed；
  - `job-000002`：`timeout_s=6` 的 check → **timed_out**，进程组确认退出，
    `check_results.py` 校验 smoke 产物 `[OK]`；
  - `job-000004`：queued 取消，未执行、summary 保留；
  - 队列在超时后继续执行下一 job（`job-000006` succeeded），无 halt；
  - 最终版本代码复跑 `job-000007`：同样 24/24 gate、31.404 tok/s，`--notify-cmd`
    投递成功（并把更早的 `pending` 通知各补投一次，按 job 去重）。
  - `job-000008`（commit `4891030`）：`m3_profile.sh --out/--steps 8` 的 6 个 case
    全部通过产物校验。每步 kernel 数 eager 7242 / graph 7290；graph-512 墙钟
    **32.5 µs**、eager-512 112.6 µs，graph-7680 45.0 µs、eager-7680 111.3 µs，
    <5 µs kernel 占 93.6%；correctness=not_checked（纯 profile）。
  - `job-000009`（commit `4891030`）：`m3_memory.py --only graph-512-1024 --steps 32`
    的 10 个 stage 通过校验；load peak 14.458 GB、max allocated **14.879 GB** /
    reserved 15.005 GB、min free **1.170 GB**；allocated 与 cycle2 增长 0、KV ptr
    稳定、reset 后 logits 一致；correctness=not_checked。
  - 审查修复后复跑 `job-000011`（commit `ed52250`）：同样 24/24 gate、bench
    **31.401 tok/s**；真实外部 CUDA 进程（558 MiB）下 `serve` 等 10.08 s 后 halt，
    job 保持 queued（`job-000010` 随后取消）。
- 复现命令与结果路径见 [journal](journal/2026-10-23-experiment-tools.md) 和
  [审查修复记录](journal/2026-10-23-experiment-tools-review-fixes.md)。

## 11. 已知限制

- profile 产物约 295 MB（nsys trace 为主），只在 `results/experiments/` 下保留、
  不入库；复核时需要读本地路径，或把关键 `timeline_summary` 数字抄进 journal。
- 成功 job 的 worktree 会被删除，事后用校验器复查时无法重算 fixtures manifest
  hash；执行期（worktree 尚在）仍会重算并拦截错配。
- 默认不主动向 session 投递通知（`pending`），需要 Coordinator 转发；配置
  `--notify-cmd` 后是“尽力投递一次”，不做重试。
- GPU 守卫是“取 job 前检查 + 10 s 释放宽限”，检查与真正启动步骤之间仍有一个
  很小的竞争窗口；它防的是常态并发，不是恶意抢占。
- 所有 job 共享 GPU 主机的 `.tmp/flashinfer` JIT 缓存；被 kill 的首次运行可能
  留下半热缓存，下一次会变慢，但不影响结果。必要时人工清理该目录。
- `results/experiments/` 不入库；需要长期保留的实验应把 summary 摘要写进
  journal，或另行归档。
- 单 GPU、单 `serve`；不支持多卡或并行 job。
- `session_id` 只用于互斥和通知，不是认证机制。
