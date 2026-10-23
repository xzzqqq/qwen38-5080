#!/usr/bin/env python3
"""GPU 实验助手：按 FIFO 串行执行 GPU 实验，保存结果并通知提交者。

管理单位是 **job：一次提交的实验**。命令（查询类输出 JSON）：

    submit --request PATH   持久化请求并入队，返回 job_id（不等待 GPU）
    status [--session ID]   列出 job 的 ID、提交者、状态、排队顺序与起止时间
    result JOB_ID           返回短结果与产物路径；未完成时返回当前状态
    wait JOB_ID ...         等待任意一个 job 结束；执行器失效/队列停顿时报错
    cancel JOB_ID           取消排队或运行中的 job；返回取消是否完成
    serve                   唯一的 GPU 执行进程，按 FIFO 消费队列

设计要点（详见 docs/experiments.md）：
- 请求只接受已注册脚本（scripts/m3_check.py / bench/e2e/decode.py /
  scripts/m3_memory.py / scripts/m3_profile.sh），且必须显式写到 `{output}`；
  产物由 runner/experiment/artifacts.py 结构化校验，不靠阅读 stdout。
- job 从请求 commit 的 detached worktree 执行；`.venv` / `models` 只做符号链接，
  不复制权重、不改公共环境。
- 队列串行、单 `serve`，job 期间持有 GPU 独占锁；超时/取消先杀进程组并确认
  退出，确认不了就停队列。failed / timed_out / interrupted 不自动重跑。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()


def _default_root() -> Path:
    """默认仓库根：git common dir 的父目录（从 Explorer worktree 调用时指向主仓库）。

    所有 session 共用同一个队列和 results/experiments，所以不能按 cwd 或按
    ``__file__`` 取根 —— Explorer 可能在自己的 worktree 里提交。
    """
    env = os.environ.get("EXPERIMENT_ROOT")
    if env:
        return Path(env).resolve()
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(SCRIPT.parent),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - 不在 git 仓库里时退回脚本位置
        return SCRIPT.parents[1]
    common = Path(out)
    return (common.parent if common.name == ".git" else common).resolve()


DEFAULT_ROOT = _default_root()
sys.path.insert(0, str(DEFAULT_ROOT))

from runner.experiment import executor, schema  # noqa: E402
from runner.experiment.store import (  # noqa: E402
    TERMINAL,
    SessionBusy,
    Store,
    StoreError,
    now_iso,
    parse_iso,
    read_json,
    write_json_atomic,
)


# ------------------------------------------------------------------ 输出/错误
def emit(payload: object, code: int = 0) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return code


def fail(message: str, code: int = 6, **extra: object) -> int:
    """错误也输出 JSON（stdout），stderr 只放一行人读提示。"""
    payload = {"error": message, **extra}
    print(message, file=sys.stderr)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return code


def make_store(args: argparse.Namespace) -> Store:
    return Store(
        Path(args.root),
        state_dir=Path(args.state_dir) if args.state_dir else None,
        worktree_root=Path(args.worktree_root) if args.worktree_root else None,
    )


# ------------------------------------------------------------------ payload
def job_payload(store: Store, job: dict) -> dict:
    """result / wait 的统一载荷：完成返回 summary，未完成返回当前状态。"""
    job_id = job["job_id"]
    state = store.read()
    record = state["jobs"].get(job_id) or job
    view = store.public_view(record, state)
    if record.get("state") in TERMINAL:
        path = store.summary_path(job_id)
        if path.exists():
            try:
                payload = read_json(path)
                payload["state"] = record.get("state")
                payload["notification"] = record.get("notification")
                return payload
            except Exception:  # noqa: BLE001
                pass
        view["note"] = "summary.json 缺失（中断/崩溃）；见 execution.json 与日志"
        execution_path = store.execution_path(job_id)
        if execution_path.exists():
            try:
                view["execution"] = read_json(execution_path)
            except Exception:  # noqa: BLE001
                pass
        return view
    view["note"] = "job 未完成；不触发重跑，完成后用 result 读取"
    view["job_dir"] = str(store.job_dir(job_id))
    view["logs_dir"] = str(store.logs_dir(job_id))
    view["output_dir"] = str(store.output_dir(job_id))
    return view


# ------------------------------------------------------------------ submit
def cmd_submit(args: argparse.Namespace) -> int:
    store = make_store(args)
    try:
        request = schema.load_request(Path(args.request), store.root)
    except schema.RequestError as exc:
        return fail(f"请求不合法: {exc}", 2)
    try:
        record = store.submit(request)
    except SessionBusy as exc:
        existing = exc.record
        return emit(
            {
                "error": "session_busy",
                "reason": str(exc),
                "session_id": request.session_id,
                "existing_job_id": existing.get("job_id"),
                "existing_state": existing.get("state"),
                "job_id": existing.get("job_id"),
                "note": "每个 session 最多一个未完成 job；等它结束后再提交",
            },
            3,
        )
    state = store.read()
    payload = {
        "job_id": record["job_id"],
        "state": record["state"],
        "session_id": record["session_id"],
        "commit": record["commit"],
        "timeout_s": record["timeout_s"],
        "queue_position": store.queue_position(state, record["job_id"]),
        "job_dir": record["job_dir"],
        "output_dir": record["output_dir"],
        "request_path": str(store.request_path(record["job_id"])),
        "request_sha256": record["request_sha256"],
        "steps": [
            {
                "kind": step.kind,
                "argv": step.expand_argv(Path(record["output_dir"])),
                "result": str(step.result_path(Path(record["output_dir"]))),
            }
            for step in request.steps
        ],
        "note": "已入队；执行由 GPU 主机上的 serve 消费，不会在这里等待",
    }
    return emit(payload, 0)


# ------------------------------------------------------------------ status
def cmd_status(args: argparse.Namespace) -> int:
    store = make_store(args)
    try:
        payload = store.status(session=args.session, limit=args.limit)
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)
    return emit(payload, 0)


# ------------------------------------------------------------------ result
def cmd_result(args: argparse.Namespace) -> int:
    store = make_store(args)
    try:
        job = store.get(args.job_id)
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)
    if job is None:
        return fail(f"未知 job: {args.job_id}", 5, job_id=args.job_id)
    return emit(job_payload(store, job), 0)


# ------------------------------------------------------------------ wait
def cmd_wait(args: argparse.Namespace) -> int:
    store = make_store(args)
    ids = list(dict.fromkeys(args.job_ids))
    try:
        jobs = {job_id: store.get(job_id) for job_id in ids}
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)
    unknown = [job_id for job_id, job in jobs.items() if job is None]
    if unknown:
        return fail("未知 job", 5, unknown=unknown)

    deadline = time.monotonic() + args.timeout if args.timeout else None
    while True:
        try:
            state = store.read()
        except StoreError as exc:
            return fail(f"队列不可用: {exc}", 6)
        finished = [
            job_id
            for job_id in ids
            if (state["jobs"].get(job_id) or {}).get("state") in TERMINAL
        ]
        if finished:
            return emit(
                {
                    "finished": finished,
                    "results": {
                        job_id: job_payload(store, state["jobs"][job_id])
                        for job_id in ids
                    },
                },
                0,
            )
        statuses = {
            job_id: store.public_view(state["jobs"][job_id], state) for job_id in ids
        }
        if state.get("halted"):
            return emit(
                {
                    "error": "queue_halted",
                    "reason": "队列被停用；处理 halted.reason 后用 serve --clear-halt 恢复",
                    "halted": state["halted"],
                    "jobs": statuses,
                },
                4,
            )
        health = store.serve_health(state)
        if not health["running"]:
            # 启动宽限：Coordinator 可能刚 spawn serve。连续 ~2s 都拿不到锁才报错。
            for _ in range(4):
                time.sleep(0.5)
                state = store.read()
                health = store.serve_health(state)
                if health["running"]:
                    break
            if not health["running"]:
                return emit(
                    {
                        "error": "executor_not_running",
                        "reason": "队列里有未完成 job，但没有 serve 持有 serve.lock",
                        "serve": health,
                        "jobs": {
                            job_id: store.public_view(state["jobs"][job_id], state)
                            for job_id in ids
                        },
                    },
                    4,
                )
        age = health.get("heartbeat_age_s")
        if age is not None and age > args.stale_after:
            return emit(
                {
                    "error": "executor_stalled",
                    "reason": f"serve 心跳已 {age:.0f}s 未更新（阈值 {args.stale_after:.0f}s）",
                    "serve": health,
                    "jobs": {
                        job_id: store.public_view(state["jobs"][job_id], state)
                        for job_id in ids
                    },
                },
                4,
            )
        if deadline is not None and time.monotonic() >= deadline:
            return emit(
                {
                    "error": "timeout",
                    "reason": f"等待超过 {args.timeout}s，以下 job 仍未结束",
                    "jobs": {
                        job_id: store.public_view(state["jobs"][job_id], state)
                        for job_id in ids
                    },
                },
                4,
            )
        time.sleep(args.poll)


# ------------------------------------------------------------------ cancel
def cmd_cancel(args: argparse.Namespace) -> int:
    store = make_store(args)
    try:
        job = store.get(args.job_id)
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)
    if job is None:
        return fail(f"未知 job: {args.job_id}", 5, job_id=args.job_id)
    job_id = job["job_id"]

    if job["state"] in TERMINAL:
        return emit(
            {
                "job_id": job_id,
                "state": job["state"],
                "cancelled": job["state"] == "cancelled",
                "completed": True,
                "note": "job 已结束；重复取消不产生额外操作",
            },
            0,
        )

    # queued：在队列锁内原子迁移，避免和 take_next 抢同一个 job。
    def _cancel_queued(state: dict) -> tuple[bool, dict]:
        record = state["jobs"][job_id]
        if record["state"] != "queued":
            return False, dict(record)
        created = parse_iso(record.get("created_at")) or time.time()
        record.update(
            state="cancelled",
            cancel_requested=True,
            finished_at=now_iso(),
            queue_s=max(0.0, time.time() - created),
            elapsed_s=0.0,
            correctness="not_checked",
            error={
                "step": None,
                "kind": None,
                "reason": "排队阶段被取消，未执行",
                "category": "cancelled",
            },
        )
        return True, dict(record)

    try:
        cancelled, record = store.mutate(_cancel_queued)
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)

    if cancelled:
        request = None
        try:
            request = schema.parse_request_object(
                read_json(store.request_path(job_id)), store.root
            )
        except Exception:  # noqa: BLE001 - 请求已校验过，读不出来只影响 commands 展示
            request = None
        error = record["error"]
        execution = {
            "job_id": job_id,
            "session_id": record.get("session_id"),
            "state": "cancelled",
            "commit": record.get("commit"),
            "started_at": None,
            "finished_at": record.get("finished_at"),
            "queue_s": record.get("queue_s"),
            "elapsed_s": 0.0,
            "environment": executor.light_environment(store.root),
            "steps": [],
            "error": error,
            "notification": None,
        }
        write_json_atomic(store.execution_path(job_id), execution)
        summary = executor.build_summary(
            store,
            record,
            request,
            execution,
            "cancelled",
            error,
            [],
            record.get("queue_s"),
            0.0,
        )
        write_json_atomic(store.summary_path(job_id), summary)
        executor.notify_job(store, job_id, args.notify_cmd)
        return emit(
            {
                "job_id": job_id,
                "state": "cancelled",
                "cancelled": True,
                "completed": True,
                "note": "queued job 已取消，不会再执行",
                "summary": str(store.summary_path(job_id)),
            },
            0,
        )

    # running：受理取消请求，不改状态；执行器在 1s 内终止进程组。
    record = store.request_cancel(job_id)
    health = store.serve_health()
    if health["running"]:
        return emit(
            {
                "job_id": job_id,
                "state": record["state"],
                "cancelled": True,
                "completed": False,
                "cancel_requested": True,
                "serve": health,
                "note": "执行器仍在运行；实际退出前状态保持 running，重复取消不产生额外操作",
            },
            0,
        )

    # 没有执行器：说明是执行器重启后的孤儿进程，直接终止进程组并确认退出。
    pid = record.get("pid")
    pgid = record.get("pgid")
    ticks = record.get("proc_start_ticks")
    confirmed = executor.terminate_orphan(pid, pgid, ticks)
    if not confirmed:
        store.set_halt(
            f"cancel 后无法确认 job {job_id} 的进程组 {pgid} 已退出",
            {"pid": pid, "pgid": pgid},
            job_id,
        )
        return emit(
            {
                "job_id": job_id,
                "state": record["state"],
                "cancelled": True,
                "completed": False,
                "error": "cannot_confirm_exit",
                "note": "进程仍可能存在并占用 GPU；队列已停用，需人工检查",
            },
            4,
        )
    finished_at = now_iso()
    error = {
        "step": None,
        "kind": None,
        "reason": "执行器未运行（重启后孤儿进程），cancel 已终止进程组",
        "category": "cancelled",
    }
    execution = {}
    if store.execution_path(job_id).exists():
        try:
            execution = read_json(store.execution_path(job_id))
        except Exception:  # noqa: BLE001
            execution = {}
    execution.update(
        {
            "job_id": job_id,
            "session_id": record.get("session_id"),
            "state": "cancelled",
            "commit": record.get("commit"),
            "finished_at": finished_at,
            "error": error,
            "cancel_after_restart": True,
        }
    )
    write_json_atomic(store.execution_path(job_id), execution)
    request = None
    try:
        request = schema.parse_request_object(
            read_json(store.request_path(job_id)), store.root
        )
    except Exception:  # noqa: BLE001
        request = None
    summary = executor.build_summary(
        store,
        record,
        request,
        execution,
        "cancelled",
        error,
        execution.get("steps") or [],
        record.get("queue_s"),
        None,
    )
    write_json_atomic(store.summary_path(job_id), summary)
    store.transition(
        job_id,
        "cancelled",
        finished_at=finished_at,
        cancel_requested=True,
        error={key: error.get(key) for key in ("step", "kind", "reason", "category")},
        correctness=summary["correctness"],
        summary_path=str(store.summary_path(job_id)),
    )
    executor.notify_job(store, job_id, args.notify_cmd)
    return emit(
        {
            "job_id": job_id,
            "state": "cancelled",
            "cancelled": True,
            "completed": True,
            "note": "孤儿进程已终止并确认退出",
        },
        0,
    )


# ------------------------------------------------------------------ serve
def cmd_serve(args: argparse.Namespace) -> int:
    store = make_store(args)
    try:
        code, payload = executor.serve(store, args)
    except StoreError as exc:
        return fail(f"队列不可用: {exc}", 6)
    except KeyboardInterrupt:
        return emit({"event": "serve_stop", "interrupted": True}, 0)
    return emit(payload, code)


# ------------------------------------------------------------------ 入口
GLOBAL_FLAGS = ("--root", "--state-dir", "--worktree-root", "--notify-cmd")


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """公共选项；子命令里用 SUPPRESS，让主 parser 的默认值不被覆盖。"""
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--root",
        type=Path,
        default=default if suppress else DEFAULT_ROOT,
        help=f"仓库根（默认 {DEFAULT_ROOT}；从 Explorer worktree 调用时指向主仓库）",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=default,
        help="队列/结果目录（默认 <root>/results/experiments）",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=default,
        help="job 的 detached worktree 根（默认 <root>/.tmp/experiments/worktrees）",
    )
    parser.add_argument(
        "--notify-cmd",
        default=default,
        help="可选的完成通知命令（/bin/sh -c 执行，环境变量含 JOB_ID/SESSION_ID/"
        "SUMMARY_JSON/SUMMARY_TEXT）；不配置时通知状态为 pending，由 Coordinator "
        "用 wait/result 收取",
    )


def _split_global(argv: list[str]) -> tuple[list[str], list[str]]:
    """把公共选项移到子命令之前，`experiment.py --root X status` 也能用。"""
    glob: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name in GLOBAL_FLAGS:
            glob.append(token)
            if "=" not in token and index + 1 < len(argv):
                glob.append(argv[index + 1])
                index += 2
                continue
        else:
            rest.append(token)
        index += 1
    return glob, rest


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common, suppress=True)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""常用流程：
  # GPU 主机上启动唯一执行器（full access；可放后台）
  .venv/bin/python scripts/experiment.py serve

  # Explorer：提交后结束 turn，等完成通知
  .venv/bin/python scripts/experiment.py submit --request request.json
  .venv/bin/python scripts/experiment.py result JOB_ID

  # Coordinator：后台 job 跑 wait，结束后收摘要再转发
  .venv/bin/python scripts/experiment.py wait JOB_ID ...

  # 人工处理停顿：确认 GPU 空闲、进程已退出后恢复队列
  .venv/bin/python scripts/experiment.py serve --clear-halt
文档：docs/experiments.md
""",
    )
    _add_common(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", parents=[common], help="持久化请求并入队")
    p.add_argument("--request", type=Path, required=True, help="请求 JSON 文件")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", parents=[common], help="列出 job 状态与队列")
    p.add_argument("--session", default=None, help="只看某个 session")
    p.add_argument("--limit", type=int, default=None, help="只显示最后 N 个 job")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("result", parents=[common], help="读取短结果与产物路径")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_result)

    p = sub.add_parser("wait", parents=[common], help="等待任意一个 job 结束")
    p.add_argument("job_ids", nargs="+")
    p.add_argument("--timeout", type=float, default=None, help="可选的整体等待上限（秒）")
    p.add_argument("--poll", type=float, default=0.5, help="轮询间隔（默认 0.5s）")
    p.add_argument(
        "--stale-after",
        type=float,
        default=60.0,
        help="serve 心跳超过该秒数视为停顿（默认 60s）",
    )
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("cancel", parents=[common], help="取消排队或运行中的 job")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("serve", parents=[common], help="GPU 执行器（唯一）")
    p.add_argument("--once", action="store_true", help="最多执行一个 job 后退出")
    p.add_argument("--drain", action="store_true", help="执行到队列空后退出")
    p.add_argument("--poll", type=float, default=0.5, help="空闲轮询间隔（默认 0.5s）")
    p.add_argument(
        "--no-gpu-guard",
        dest="gpu_guard",
        action="store_false",
        help="跳过外部 GPU 计算进程检查（只用于 CPU fake 命令验证）",
    )
    p.set_defaults(gpu_guard=True)
    p.add_argument(
        "--clear-halt",
        action="store_true",
        help="清除 halt 并继续执行；只在确认 GPU 空闲、孤儿进程已退出后使用",
    )
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    glob, rest = _split_global(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(glob + rest)
    try:
        return int(args.func(args))
    except (schema.RequestError, StoreError) as exc:
        return fail(f"{type(exc).__name__}: {exc}", 6)


if __name__ == "__main__":
    raise SystemExit(main())
