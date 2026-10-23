# GPU 实验队列：验收 P1/P2 修复

- 日期：2026-10-23
- 审查版本：`ed52250`
- 对应：[使用说明](../experiments.md)、[实现记录](2026-10-23-experiment-tools.md)
- checkpoint revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`

## 1. 审查发现与修复

| # | 级别 | 问题 | 修复 | 回归测试 |
|---|---|---|---|---|
| 1 | P1 | 提交绝对脚本路径时实际执行主工作区代码；产物校验仍通过 | `schema.py` 只让**仓库相对路径**匹配脚本注册表；提交时拒绝指向任一 git worktree 的绝对路径（词法 + resolve）；执行器 `verify_argv_isolation()` 二次检查绝对路径与 `..` 逃逸；`request.json` 入队后按 sha256 + commit 再核对；产物 provenance 必须 `dirty=false` | `test_absolute_repo_paths_rejected_at_submit`、`test_absolute_path_to_other_worktree_rejected`、`test_tampered_request_absolute_path_fails_without_running`、`test_verify_argv_isolation`、`test_bench/check_dirty_provenance_rejected` |
| 2 | P1 | 请求两个 check、第一个通过后中断，仍报 `correctness=passed` | `compute_correctness(step_records, request)` 按**请求的全部 check 索引**判断覆盖；只执行一部分时返回 `not_checked` | `test_correctness_covers_all_requested_checks`、`test_unexecuted_check_is_not_passed`、`test_all_requested_checks_passed_and_check_before_failed_bench` |
| 3a | P1 | job 已 failed 时进程组清理失败不触发 halt | 清理失败一律设 `halt_reason`；即使 state=failed 也写 `execution.group_cleanup_error` 并在收尾后 `raise Halt` | `test_cleanup_failure_halts_failed_job`（fault injection） |
| 3b | P1 | 恢复只检查 keeper PID，漏掉仍存活的组成员 | `reconcile()` 同时检查 `pgroup_members(pgid)`；keeper 或任一成员存活都停队列 | `test_reconcile_checks_whole_process_group` |
| 3c | P1 | `nvidia-smi` 查询失败时 GPU guard 仍放行 | `gpu_guard()` 改为 fail closed：`available=false` 直接 `Halt`，不再返回放行 | `test_gpu_guard_fails_closed_when_nvidia_smi_unavailable` |
| 4 | P2 | bench 的 fixtures 路径/内容错配可通过；逐步耗时 NaN、吞吐 Infinity 可通过 | bench 记录 `provenance.fixtures`，与请求路径比对；执行期重算 manifest hash，与 `run_config.fixtures_sha256` 和每场景配置比对；`_finite()` 检查所有指标/逐步数据（nan/inf/负数） | `test_bench_fixtures_must_match_request`、`test_bench_fixtures_content_must_match_hash`、`test_bench_nan_inf_and_negative_rejected`、`test_check_fixtures_and_dirty_checked` |
| 5 | P2 | worktree 准备失败时收尾对 None 调 `.get()`，job 滞留 running | `worktree_removed=bool((execution.get("worktree") or {}).get("removed"))`；准备失败路径正常写 summary/execution 并落 failed | `test_worktree_prepare_failure_marks_job_failed` |

补充：执行器重放请求时新增 `except schema.RequestError`，篡改过的请求会以
`invalid_request` 干净失败，不再冒 traceback。

## 2. CPU 验证

```bash
.venv/bin/python -m pytest tests/experiment -q
# 68 passed in ~42s（新增 17 个回归）
```

新增覆盖：
- 绝对脚本/venv/`--fixtures` 路径、另一个 worktree 的绝对路径、篡改 request.json；
- `verify_argv_isolation` 的绝对路径与相对 `..` 逃逸；
- 请求两个 check 但只跑一个；单 check + bench 失败的对照组；
- 清理失败（failed / succeeded 两种状态）用 monkeypatch 注入；
- reconcile 的“keeper 已死、组内成员仍活”；
- nvidia-smi 退出码非 0（fail closed）；
- worktree 目录已存在；
- bench fixtures 路径/hash、manifest 内容变更、nan/inf/负数；
- check fixtures 路径/hash 与 dirty provenance。

## 3. 真实 GPU 复验

### 3.1 外部 CUDA 进程 → halt

```bash
# 外部进程持有 558 MiB CUDA context（pid 1224308）
.venv/bin/python -c "import torch,time; x=torch.zeros(64,1024,1024,device='cuda'); time.sleep(45)" &
.venv/bin/python scripts/experiment.py serve --drain   # job-000010 在队列里
```

结果：`serve` rc=1，`error=halted`，`waited_s=10.08`，detail 列出 pid/558 MiB；
job 保持 queued。杀掉外部进程后 `cancel job-000010`，`serve --clear-halt` 恢复，
GPU 回到 0 MiB。

### 3.2 新校验器下的 check→bench

`job-000011-e3f63b2a`（commit `ed52250`）：

| 项 | 值 |
|---|---|
| state / correctness | succeeded / **passed** |
| check | 24/24 gate 通过；fixtures hash 在执行期从 worktree manifest 重算并匹配 |
| bench | graph-512-1024，32 步 × 1 次，**31.401 tok/s**，逐步数据全部有限 |
| elapsed | 124.1 s（queue 2.0 s） |
| 产物契约 | `check_results.py` 对 `output/check`、`output/bench` → `[OK]` |
| 环境 | worktree clean、无残留进程、GPU 0 MiB |

此前三个真实 job（`job-000007/8/9` 的 check/bench/profile/memory 产物）也用新
校验器重新跑了一遍，全部通过；成功 job 的 worktree 已删除，fixtures 内容 hash
改为依赖记录值与跨场景一致性（执行期仍会重算）。

## 4. 已知限制

- “进程组清理失败”无法用真实进程制造（SIGKILL 后不可控），用 monkeypatch 做
  fault injection 覆盖；真实 GPU 上验证的是“外部 CUDA 进程 → halt”。
- 外部进程检查是轮询 + 10 s 宽限，不是内核级互斥；检查与启动步骤之间仍有极小窗口。
- 成功 job 的 worktree 删除后，事后复查无法重算 fixtures manifest hash。
- 修复尚未提交；待确认后提交并 push。
