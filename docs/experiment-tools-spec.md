# GPU 实验助手实现 Spec

实现 `scripts/experiment.py`：接收本仓库的实验命令，在单张 RTX 5080 上按顺序执行，
保存结果并通知提交者。管理单位为 **job：一次提交的实验**。

## 1. 提交内容

`submit` 接收一个 JSON 文件，例如：

```json
{
  "session_id": "explorer-attention",
  "commit": "完整的待测 commit SHA",
  "timeout_s": 300,
  "steps": [
    {
      "kind": "check",
      "argv": [".venv/bin/python", "scripts/m3_check.py", "--only", "graph-1024", "--steps", "16", "--out", "{output}/check"],
      "result": "{output}/check/check.json"
    },
    {
      "kind": "bench",
      "argv": [".venv/bin/python", "bench/e2e/decode.py", "--only", "graph-512-1024", "--steps", "32", "--repeats", "1", "--out", "{output}/bench"],
      "result": "{output}/bench/bench.json"
    }
  ]
}
```

- `session_id`：提交者的原生 Agent/session 标识，用于查询和通知。
- `commit`：本仓库中存在的完整 commit SHA；Explorer 先在自己的分支提交待测源码。
- `timeout_s`：正数，限制整个 job 的执行 wall time，从开始执行计时，不含排队时间。
- `steps`：非空、有序命令列表；`kind` 为 `check / bench / memory / profile`。
- `argv`：参数数组，使用 subprocess 直接执行，不经 shell 拼接。
- `result`：该步骤必须产出的结构化结果；`{output}` 由助手替换为 job 独立产物目录的绝对路径。

提交时校验字段、commit 和命令对应的结果格式。新实验脚本先提供结构化结果的读取与
校验函数，再允许提交；不能靠 LLM 阅读 stdout 判断通过。脚本必须支持显式输出目录，
所有产物写入本 job 的 `{output}`。

## 2. 命令接口

统一入口：`.venv/bin/python scripts/experiment.py`。查询类命令输出 JSON，原始运行日志写文件。

| 命令 | 行为 |
|---|---|
| `submit --request PATH` | 持久化请求并入队，返回 `job_id`；不等待 GPU 执行 |
| `status [--session ID]` | 列出 job 的 ID、提交者、状态、排队顺序与起止时间 |
| `result JOB_ID` | 返回短结果与产物路径；未完成时返回当前状态 |
| `wait JOB_ID ...` | 等待任意一个指定 job 结束；已经结束的立即返回；队列停顿或执行器失效时返回错误及当前状态 |
| `cancel JOB_ID` | 取消排队或运行中的 job；返回取消是否已完成 |
| `serve` | 在 GPU 主机运行唯一执行进程，按 FIFO 消费队列 |

每个 session 最多一个未完成 job；再次提交时拒绝并返回已有 job ID。
已完成 job 可以重复查询，查询不触发重跑。新的显式提交创建新 job，不复用历史测量。

## 3. 代码与执行环境

1. 为 job 创建指向指定 commit 的 detached worktree，所有步骤均从该版本执行。
   原 Explorer worktree 后续编辑不影响该 job；请求内容入队后不可修改。
2. 使用 GPU 主机现有 `.venv`、CUDA 环境和 checkpoint；为执行目录提供其路径。
   记录实际环境和 checkpoint revision，不复制整份模型，不修改公共环境或权重。
3. GPU 主机按 `scripts/m2_env.sh` 配置环境，执行器在 full access 下运行。
   队列内所有 CUDA 检查、计时、profiling 和 JIT 共用同一张 GPU 的独占锁。
4. 不 push、不改 main、不重写历史。job 的执行目录与结果目录彼此隔离。

## 4. 队列、失败与取消

- 请求成功入队后按顺序执行，同时最多一个 job 使用 GPU；重复启动 `serve` 不能产生
  第二个执行者。提交、读取状态和取消期间保持队列记录一致。
- job 内的步骤顺序执行。每一步都必须退出码为 0、产物完整且校验通过，才运行下一步。
  correctness failure、缺必测项或缺产物均停止后续 benchmark。
- 每个 job 一个独立进程组。超时或取消时终止所属进程，确认退出后才释放设备锁并执行下一项。
  无法确认进程退出或 GPU 可用时停队列，记录原因并通知 Coordinator。
- queued job 取消后不再执行。running job 的取消请求可以立即受理，但在实际退出前仍
  标为 running，并返回 `cancel_requested=true`。重复取消不产生额外操作。
- 请求、状态、日志和结果保存在磁盘。执行器重启时先核对之前的进程：存活进程仍占用
  设备；退出状态不明则停队列；确认已退出但没有完整结果的 job 标为 interrupted。
- 不自动重跑 failed、timed_out 或 interrupted job；重试由 Agent 通过新的提交发起。
  执行器不决定下一次实验，也不自动展开参数搜索。

job 状态：`queued / running / succeeded / failed / timed_out / cancelled / interrupted`。
`succeeded` 只表示提交的步骤全部完成；正确性结论单独记录，性能收益由 Agent 根据可比结果判断。

## 5. 结果校验与摘要

复用 `runner/report.py` 和现有 check/bench 产物读取逻辑，补齐校验：

- 检查请求与实际场景、steps、repeats、输入和执行配置一致。不能把旧的32步/1次结果
  视为新的511步/5次请求已完成；执行时不使用现有 bench 的自动 `--resume`。
- check 必须覆盖请求的场景/必测项，gate 非空且全部通过；异常和空结果为失败。
  diagnostic 单列，不代替 gate，也不因 diagnostic 差异改判已通过的 gate。
- bench 必须完整结束，样本数和逐步数据符合请求，指标有效；保存原始 samples。
- memory/profile 按对应脚本检查必需产物和字段。校验失败视为 job failed，不仅看退出码。

`result` 和完成通知的短摘要包含：

| 字段 | 内容 |
|---|---|
| `job_id / session_id / state` | job 身份、提交者和执行状态 |
| `commit / commands` | 实际代码版本与展开后的执行命令 |
| `correctness` | `passed / failed / not_checked`；未执行或未完成有效检查时为 not_checked |
| `metrics` | 实际得到的速度、延迟、显存等指标，标明场景、计时口径和样本数 |
| `queue_s / elapsed_s` | 排队耗时、执行耗时，单位为秒 |
| `error` | 首个失败步骤和短原因；成功时为空 |
| `output_dir` | 原始产物与日志路径 |

任务含多个检查时，只有全部要求的检查完成且通过才能标 correctness=passed。
未运行数值检查的纯 profile/bench job 可以执行成功，但 correctness 仍为 not_checked。

产物目录：

```text
results/experiments/<job-id>/
  request.json             # 原始请求
  execution.json           # 状态、时间、命令、环境、进程信息、产物清单
  summary.json             # 短结果
  logs/                    # 各步骤 stdout/stderr、退出码
  output/                  # 现有脚本生成的原始结果
```

结果完整校验后发布 summary；中断和失败也保留记录，不覆盖其他 job 的产物。
记录必须支持复跑：commit、checkpoint revision、输入身份、完整参数、依赖/设备环境齐全。

## 6. 等待与完成通知

- 使用 coding harness 已有的后台执行/完成通知能力，将摘要送回请求中的原 session。
  session 运行中发送消息，空闲时用原生 follow-up/resume；不新建 Explorer。
- 不支持直接投递时，由 Coordinator 用 `wait` 收取完成摘要并转发。`wait` 的进程等待
  在程序内完成；空超时不要求 LLM 反复调用，使用 harness 的后台等待能力接续。
- 通知只携带短摘要和路径，完整日志按需读取。以 job ID 去重；通知失败不改变实验结果，
  原 session 仍可用 `result` 查询。已取消/关闭的 session 不自动重建。
- job 的执行结果与通知投递情况分别记录；不能因为消息发送失败重新跑实验。

Coordinator 的研究决策继续记录在 Workflow 指定的 journal，引用 job ID。
助手只记录执行事实，不替 Coordinator 生成研究结论。

## 7. 验收

先用 CPU fake command 验证队列、进程和结果处理，再在5080上执行一个小型 check→bench。

| 场景 | 必须满足 |
|---|---|
| 两个 session 同时提交 | FIFO 执行，GPU 使用不重叠，结果互不覆盖 |
| 提交后改动 Explorer worktree | 实际执行仍是提交的 commit |
| check 失败或缺产物 | job failed，bench 不执行，摘要可定位日志 |
| 旧结果配置与新请求不同 | 校验失败，不报告完成 |
| 超时/取消 | 所属进程退出后才执行下一 job；状态不明时停队列 |
| 执行器重启 | 能识别存活/中断 job，不重复启动占用 GPU 的执行 |
| 完成通知 | 返回原 session；通知失败仍能查询结果，不重跑实验 |
| 重复 status/result/cancel | 不增加实验执行次数，不破坏已有结果 |
| 纯 bench/profile | 不错误标记 correctness=passed |

CPU 验证和真实 GPU smoke 的命令、结果及限制写入 journal。
