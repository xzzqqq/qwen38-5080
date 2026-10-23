# GPU 实验队列：实现与验收

- 日期：2026-10-23
- 对应：[experiment-tools spec](../experiment-tools-spec.md)、
  [optimization workflow](../optimization-workflow.md)、[使用说明](../experiments.md)
- 代码 commit：check→bench 验收时 HEAD 为
  `34303182db95fe21be4993e2cc7805b4ff0246a8`；本工具提交为 `cd16e2f`；
  profile / memory 复核在 `4891030`；checkpoint revision
  `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`

## 1. 做了什么

按 spec 实现 `scripts/experiment.py`，管理单位是 job（一次提交的实验）：

| 文件 | 内容 |
|---|---|
| `scripts/experiment.py` | CLI：`submit / status / result / wait / cancel / serve`；默认 root 取 git common dir（多 worktree 共用一个主队列） |
| `runner/experiment/schema.py` | 请求 schema、argv/flag 解析、commit 校验、脚本注册表、`{output}` 路径约束、submit 期场景/steps/repeats 校验 |
| `runner/experiment/artifacts.py` | check / bench / memory / profile 的结构化产物校验（commit、场景、steps、repeats、fixtures、样本数、gate/coverage、stage、nsys trace） |
| `runner/experiment/store.py` | `queue.json` 原子写 + flock、单 `serve` 锁、GPU 锁、状态机、session 互斥、进程/心跳记录 |
| `runner/experiment/executor.py` | detached worktree、共享 `.venv/models` 符号链接、job 级进程组、超时/取消、失败短路、halt、重启核对、通知 |
| `tests/experiment/` | 51 个 CPU 测试（临时 git 仓库 + 同名 fake 实验脚本，走真实队列/校验路径） |
| `scripts/m3_profile.sh` | 补 `--out DIR` / `--steps N`（队列要求显式输出目录；默认行为不变） |
| `docs/experiments.md` | 使用说明、契约、失败恢复、反馈阶梯 |

关键设计：

1. **请求不可变 + 提交期校验**。完整 40 位 SHA、脚本必须在该 commit 里存在、只接受
   已注册脚本、`--out`/`result` 必须在 `{output}` 内、禁止 `--resume/--worker/--list`、
   `--only` 场景名对照该 commit 源码里的字面量场景表（AST 解析）校验。错误挡在 GPU
   run 之前。
2. **worktree 隔离**。`git worktree add --detach <commit>`；`.venv`、`models` 只做
   符号链接，不复制权重、不改公共环境。实测提交后改 worktree、再 commit 到新 HEAD，
   job 仍执行原 commit 的代码（fake 测试 + 真实 GPU smoke 都验证）。
3. **job 级进程组**。每个 job 用一个 `sleep` keeper 当组长（`process_group=0`），
   所有步骤 join 同一个 pgid；超时/取消 SIGTERM→10 s→SIGKILL，确认组内非僵尸进程
   消失后才释放 GPU 锁。步骤正常退出后清理同组残留后台进程并记录。
4. **结构化校验而非读 stdout**。gate 失败（`m3_check` 退出码 1）也会读 `report.json`
   判 `correctness=failed`；缺产物/配置不符/超时则 `not_checked`。diagnostic 单列。
5. **停队列而不是抢设备**。`serve` 取 job 前用 `nvidia-smi` 查计算进程；有外部占用
   先等 10 s（被杀进程释放 12 GB context 有秒级延迟），持续占用才 halt。
   restart 孤儿、无法确认退出都 halt，需 `serve --clear-halt`。
6. **通知可插拔**。`--notify-cmd`/`DSH_EXPERIMENT_NOTIFY_CMD` 注入
   `JOB_ID/SESSION_ID/SUMMARY_JSON/...`；未配置则 `pending`，由 Coordinator 用
   `wait`/`result` 转发。按 job 去重，通知失败不改结果。

## 2. CPU 验证

```bash
.venv/bin/python -m pytest tests/experiment -q
# 51 passed in ~35s
```

测试用临时 git 仓库提交同名 fake 脚本（`scripts/m3_check.py`、`bench/e2e/decode.py`、
`scripts/m3_memory.py`、`scripts/m3_profile.sh`），产物 schema 与真实脚本一致，
因此队列、worktree、进程组、校验器走的是同一条代码路径。覆盖 spec §7：

| 场景 | 测试 | 断言 |
|---|---|---|
| 两 session 同时提交 | `test_fifo_no_overlap_and_isolated_outputs` | FIFO；fake 脚本用 `flock` 检测重叠；输出目录互不覆盖 |
| 改动 Explorer worktree | `test_job_runs_submitted_commit_not_edited_worktree` | 产物里的 commit 与 `probe.MARKER` 都是提交的 commit |
| check 失败 / 缺产物 | `test_failed_check_stops_bench`、`test_missing_artifact_fails_without_correctness_pass` | bench 未执行；`failed`/`not_checked` 区分正确 |
| 旧配置 | `test_stale_bench_config_fails_end_to_end` + validator 单测 | 32 步产物不能冒充 4 步请求；complete=false 失败 |
| 超时 | `test_timeout_kills_process_group_and_queue_continues` | 孙进程 `sleep 120` 被杀、下一 job 继续 |
| 取消 | `test_cancel_queued_never_executes`、`test_cancel_running_waits_for_exit` | queued 不执行；running 先 `cancel_requested`，退出后才 `cancelled` |
| 执行器重启 | `test_restart_*`、`test_live_orphan_*`、`test_recovered_summary_is_adopted` | 死进程→interrupted；活孤儿→halt；cancel 杀孤儿；summary 完整→采纳 |
| 通知 | `test_notification_delivered_once_and_failure_safe` | 只投一次；失败仍是 succeeded；重复查询不重投 |
| 重复 status/result/cancel | `test_duplicate_queries_do_not_rerun` | fake 脚本启动次数恒为 1 |
| 纯 bench/profile | `test_bench_only_is_not_checked`、`test_profile_and_memory_jobs_end_to_end` | correctness 不是 passed |

## 3. 5080 真实 smoke

### 3.1 check → bench

请求见 `.tmp/experiments/smoke-request.json`（未入库），执行：

```bash
.venv/bin/python scripts/experiment.py submit --request .tmp/experiments/smoke-request.json
.venv/bin/python scripts/experiment.py serve --drain &
.venv/bin/python scripts/experiment.py wait job-000001-e6dcf12d
.venv/bin/python scripts/experiment.py result job-000001-e6dcf12d
```

结果（`results/experiments/job-000001-e6dcf12d/summary.json`）：

| 项 | 值 |
|---|---|
| state / correctness | succeeded / **passed** |
| 步骤 | check `--only graph-1024 --steps 16`；bench `--only graph-512-1024 --steps 32 --repeats 1` |
| check | **24/24 gate 通过**；1 个 diagnostic 差异（graph 与冻结 eager trace 的 free-running 分叉，已知） |
| bench | model **31.40 tok/s**（median）、system 31.37、request 22.58；32 步；peak 14.818 GB / reserved 15.005 GB / free 1.170 GB |
| queue / elapsed | 2.207 s / 128.9 s（含两次模型 load） |
| worktree | `clean=true`，成功后已删除；`.venv`、`models` 为符号链接 |
| 通知 | `pending`（未配置 notify command，Coordinator 用 result/wait 收取） |

用最终版本代码再跑一次（`job-000007-0135a6cf`，带 `--notify-cmd`）结果一致：
24/24 gate、bench 31.404 tok/s、correctness=passed、elapsed 125.6 s；通知
**delivered**（rc=0，0.20 s），并把之前 6 个 `pending` 的终态 job 各补投了一次
（`deliver_pending` 的按 job 去重，每个 job 只投一次）。

同一产物交给现有契约检查：

```bash
.venv/bin/python scripts/check_results.py \
  results/experiments/job-000001-e6dcf12d/output/check \
  results/experiments/job-000001-e6dcf12d/output/bench
# [OK] ... [OK] ... 2/2 个当前 run dir 通过
```

### 3.2 超时与队列继续

```bash
# timeout_s=6 的 graph-1024 check；后面跟一个 8 步 bench
.venv/bin/python scripts/experiment.py serve --drain
```

- `job-000002-274b8a28`：**timed_out**，elapsed 6.225 s；步骤 exit=-15，
  `confirmed_exit=true`，pgid 1088733；GPU 随即回到 0 MiB。
- `job-000003-386a2307`（同队列下一个 job）：succeeded，bench 31.45 tok/s。
- 再复跑一次（`job-000005` timeout + `job-000006` follower）结果一致，说明超时后
  队列不被卡住。

### 3.3 取消

- `job-000004-28f02342` queued 时 `cancel`：立即 `cancelled`，从未执行，summary
  落盘（correctness=not_checked）。
- running 取消由 CPU 测试覆盖（真实 GPU 上只做了 queued cancel，避免额外占用）。
- 成功后 worktree 删除；timed_out 的 job-000002/job-000005 的 worktree 保留供查，
  `.tmp/experiments/worktrees/` 里只剩这两个目录。

### 3.4 profile（真实 nsys）

工具提交后（`4891030` 已包含 `m3_profile.sh --out/--steps`）跑了一个真实 profile job：

```bash
.venv/bin/python scripts/experiment.py submit --request .tmp/experiments/profile-request.json
.venv/bin/python scripts/experiment.py serve --drain
.venv/bin/python scripts/experiment.py result job-000008-6baf748a
```

- `job-000008-6baf748a`：state=succeeded、correctness=**not_checked**（纯 profile，
  符合 spec），queue 1.9 s / elapsed 124.7 s；6 个 case × 8 步。
- 产物校验通过：`timeline_summary.json`、`summary.md`、`provenance.json`、每 case 的
  `meta.json`、`trace.nsys-rep`、`gpu_trace.csv`；`check_results.py` 对
  `output/profile` 返回 `[OK]`。
- 计时（每步 kernel 累计 / 墙钟）：eager-512 30.29 ms / 112.55 ms、eager-7680
  44.08 / 111.28；graph-512 29.74 / **32.54**、graph-7680 42.34 / **45.03**；
  graph 每步 7290 个 kernel、eager 7242，<5 µs kernel 占 93.6%——与 M3 记录一致。
- `--out` 生效：worktree 里没有留下 `results/m3/...`；成功后 worktree 删除，
  GPU 回到 0 MiB。
- 产物 295 MB（nsys trace / gpu_trace.csv 为主），按 `.gitignore` 只留在
  `results/experiments/job-000008-6baf748a/output/profile/`，不入库。

### 3.5 memory（真实）

```bash
# job-000009-1083ebdd，commit 4891030；graph-512-1024 --steps 32
.venv/bin/python scripts/experiment.py submit --request .tmp/experiments/memory-request.json
.venv/bin/python scripts/experiment.py serve --drain
.venv/bin/python scripts/experiment.py result job-000009-1083ebdd
```

- state=succeeded、correctness=not_checked、elapsed 112.9 s；10 个 stage。
- load peak 14.458 GB，max allocated **14.879 GB** / reserved 15.005 GB，min free
  **1.170 GB**；`allocated_delta` 与 `cycle2_*` 都是 0，KV data ptr 稳定，
  reset 后 prefill / step logits 一致。
- `check_results.py` 对 `output/memory` → `[OK]`；worktree 删除、GPU 回到 0 MiB。

## 4. 实现过程中踩到并修掉的问题

1. **per-step pgid 不满足“每个 job 一个进程组”**。最初的实现给每一步
   `start_new_session=True`，步骤之间换组；后来改成 job 级 keeper 当组长，步骤用
   `process_group=<keeper>` join。第一次用 `start_new_session=True` 建 keeper 时，
   步骤 `setpgid` 到另一个 session 的组会 **EPERM**，改为 `process_group=0`
   （同 session 内新建组）后通过。
2. **GPU 守卫把“正在释放的 CUDA context”当成外部占用**。首次超时 smoke 后，
   被杀的 m3_check worker 还在释放 12 GB，`nvidia-smi` 立刻仍列出 11912 MiB，
   守卫直接 halt（这是正确的保守行为，但会误伤自己的收尾）。修复：发现占用后先
   等 `GPU_DRAIN_S=10 s`、持续占用才停队列；修复后超时→下一 job 正常继续。
3. **僵尸进程让“确认退出”永远失败**。`pgid_alive` 原来用 `killpg(pgid, 0)`，僵尸
   也在组里。改成扫描 `/proc`、只把非僵尸进程算作存活；`terminate_orphan` 同样
   规则。
4. **gate 失败时 m3_check 退出码非 0**。执行器原来先判退出码，导致 correctness
   只能是 not_checked。改为退出码非 0 也先读产物：有 gate 失败 → `failed`；
   否则按退出码判 failed、correctness=not_checked。
5. **测试细节**：request.json 提交后 chmod 0444；`pgid_alive` 的僵尸语义；
   `wait` 对 serve 刚启动的 2 s 宽限（避免 Coordinator 抢跑误判 executor dead）。

## 5. 结论

- 工具在 CPU 上覆盖了 spec §7 的九类场景；在 5080 上完成了 check→bench 的真实
  小实验：correctness=passed、24/24 gate、bench 31.40 tok/s（与 M3 baseline
  31.41 一致），产物同时通过 `check_results.py` 契约。
- 四种 kind 都在 GPU 上实跑过：check/bench（`job-000001` / `job-000007`）、
  profile（`job-000008`）、memory（`job-000009`）；超时/取消能杀掉 job 进程组并
  让队列继续；执行器重启、halt、通知去重都有测试。
- `succeeded` 只表示步骤跑完；性能比较仍由 Coordinator 按可比协议判断。

## 6. 已知限制

- profile 产物约 295 MB（nsys trace 为主），只在 `results/experiments/` 保留、
  不入库；复核时需要读本地路径。
- 通知默认 `pending`，依赖 Coordinator 的 `wait`；自动投递只尝试一次，不重试。
- GPU 守卫是轮询 + 10 s 宽限，不是内核级互斥；检查与步骤启动之间仍有极小窗口。
- 所有 job 共用 `.tmp/flashinfer` JIT 缓存；首次运行被 kill 后下一次可能变慢
  （本次 follower 的 110 s 主要是模型 load + 半热 JIT），不影响数值/性能结论。
- 单 GPU、单 serve；`results/experiments/` 不入库。

## 7. 下一步

1. Coordinator 在 GPU 主机上常驻 `serve`（tmux/systemd），用后台 `wait` 收摘要；
   Explorer 沿用 `docs/optimization-workflow.md` 的流程。
2. Phase 2 的第一个 NVFP4 GEMV 候选按“check（gate）→ bench（与 baseline 同协议）”
   两步提交，用 job ID 在 journal 里引用结果。
3. 需要解释 kernel 级收益时，用同一套 profile job 在改动前后各跑一次，按
   `per_step_kernels / per_step_kernel_us / per_step_wall_us` 对比（注意 profile
   产物不入库，路径写进 journal）。
