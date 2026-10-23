"""实验队列的磁盘状态：原子 JSON、跨进程 flock、单执行器锁。

设计：
- 状态只有一份 ``results/experiments/queue.json``；所有写操作在 ``queue.lock``
  的独占 flock 下完成，读用共享锁，写入用 tmp + ``os.replace``，避免半截 JSON。
- ``serve.lock`` 是单执行器锁：拿不到就说明已有执行器，绝不启动第二个 GPU 用户。
- ``gpu.lock`` 是 job 级的 GPU 独占锁（队列串行的显式记录），执行期间持有。
- job 目录与队列记录分开：请求/日志/产物在 job 目录，队列记录只放调度需要的字段；
  崩溃时用 pid + /proc starttime 判断“上次的进程是否还活着”。
- 不自动重跑 failed / timed_out / interrupted：状态机只允许单向终态。
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

STATE_VERSION = 1
TERMINAL = frozenset(
    {"succeeded", "failed", "timed_out", "cancelled", "interrupted"}
)
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset(
        {"succeeded", "failed", "timed_out", "cancelled", "interrupted"}
    ),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "timed_out": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}
JOB_ID_RE = re.compile(r"^job-\d{6}-[0-9a-f]{8}$")


class StoreError(RuntimeError):
    """队列状态不可用（损坏、锁超时、非法状态迁移）。"""


class SessionBusy(StoreError):
    def __init__(self, record: dict[str, Any]):
        super().__init__(
            f"session {record.get('session_id')!r} 已有未完成 job {record.get('job_id')}"
        )
        self.record = record


class ServeBusy(StoreError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "next_seq": 1,
        "order": [],
        "jobs": {},
        "serve": None,
        "halted": None,
        "updated_at": now_iso(),
    }


class Store:
    def __init__(
        self,
        root: Path,
        state_dir: Path | None = None,
        worktree_root: Path | None = None,
    ):
        self.root = Path(root).resolve()
        self.state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else self.root / "results/experiments"
        )
        self.worktree_root = (
            Path(worktree_root).resolve()
            if worktree_root is not None
            else self.root / ".tmp/experiments/worktrees"
        )
        self.queue_path = self.state_dir / "queue.json"
        self.queue_lock_path = self.state_dir / "queue.lock"
        self.serve_lock_path = self.state_dir / "serve.lock"
        self.gpu_lock_path = self.state_dir / "gpu.lock"

    # ------------------------------------------------------------- 基础路径
    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(str(job_id)):
            raise StoreError(f"非法 job id: {job_id!r}")
        return self.state_dir / job_id

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    def logs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "logs"

    def summary_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "summary.json"

    def execution_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "execution.json"

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def worktree_dir(self, job_id: str) -> Path:
        return self.worktree_root / job_id

    # ------------------------------------------------------------- 锁与读写
    @contextmanager
    def locked(self, *, exclusive: bool = True, timeout: float = 60.0) -> Iterator[None]:
        self.ensure_dirs()
        fd = os.open(self.queue_lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise StoreError(
                            f"等待队列锁超时（{timeout}s）: {self.queue_lock_path}"
                        ) from None
                    time.sleep(0.02)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

    def _read_state_unlocked(self) -> dict[str, Any]:
        if not self.queue_path.exists():
            return _new_state()
        try:
            state = read_json(self.queue_path)
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"queue.json 解析失败: {exc}") from exc
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
            raise StoreError("queue.json 结构不合法")
        if int(state.get("version", 0)) != STATE_VERSION:
            raise StoreError(f"queue.json version={state.get('version')} 不受支持")
        state.setdefault("order", [])
        state.setdefault("halted", None)
        state.setdefault("serve", None)
        state.setdefault("next_seq", len(state["jobs"]) + 1)
        return state

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_iso()
        write_json_atomic(self.queue_path, state)

    def read(self) -> dict[str, Any]:
        with self.locked(exclusive=False):
            return self._read_state_unlocked()

    def mutate(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        with self.locked():
            state = self._read_state_unlocked()
            result = fn(state)
            self._write_state_unlocked(state)
            return result

    # ------------------------------------------------------------- 提交
    def submit(self, request: Any) -> dict[str, Any]:
        """持久化请求并入队；每个 session 最多一个未完成 job。"""
        with self.locked():
            state = self._read_state_unlocked()
            for job_id in state["order"]:
                record = state["jobs"].get(job_id) or {}
                if (
                    record.get("session_id") == request.session_id
                    and record.get("state") not in TERMINAL
                ):
                    raise SessionBusy(dict(record))

            seq = int(state["next_seq"])
            job_id = f"job-{seq:06d}-{uuid.uuid4().hex[:8]}"
            state["next_seq"] = seq + 1
            job_dir = self.job_dir(job_id)
            (job_dir / "logs").mkdir(parents=True, exist_ok=True)
            (job_dir / "output").mkdir(parents=True, exist_ok=True)
            request_path = job_dir / "request.json"
            write_json_atomic(request_path, request.raw)
            # 请求入队后不可修改：只读文件 + 队列记录里的 sha256。
            try:
                os.chmod(request_path, 0o444)
            except OSError:
                pass

            record = {
                "job_id": job_id,
                "session_id": request.session_id,
                "state": "queued",
                "submit_seq": seq,
                "created_at": now_iso(),
                "started_at": None,
                "finished_at": None,
                "queue_s": None,
                "elapsed_s": None,
                "timeout_s": float(request.timeout_s),
                "commit": request.commit,
                "commit_requested": request.commit_requested,
                "request_sha256": request.sha256,
                "job_dir": str(job_dir),
                "output_dir": str(job_dir / "output"),
                "steps_total": len(request.steps),
                "step_kinds": [step.kind for step in request.steps],
                "steps_done": 0,
                "pid": None,
                "pgid": None,
                "proc_start_ticks": None,
                "cancel_requested": False,
                "error": None,
                "correctness": None,
                "notification": None,
                "worktree_removed": False,
            }
            state["jobs"][job_id] = record
            state["order"].append(job_id)
            self._write_state_unlocked(state)
            return dict(record)

    # ------------------------------------------------------------- 查询
    def get(self, job_id: str) -> dict[str, Any] | None:
        state = self.read()
        record = state["jobs"].get(job_id)
        return dict(record) if record else None

    def ordered_jobs(self) -> list[dict[str, Any]]:
        state = self.read()
        return [dict(state["jobs"][jid]) for jid in state["order"] if jid in state["jobs"]]

    def queue_position(self, state: dict[str, Any], job_id: str) -> int | None:
        record = state["jobs"].get(job_id) or {}
        if record.get("state") != "queued":
            return None
        queued = [
            jid
            for jid in state["order"]
            if (state["jobs"].get(jid) or {}).get("state") == "queued"
        ]
        try:
            return queued.index(job_id) + 1
        except ValueError:
            return None

    def public_view(self, record: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        created = parse_iso(record.get("created_at")) or now
        started = parse_iso(record.get("started_at"))
        finished = parse_iso(record.get("finished_at"))
        queue_s = record.get("queue_s")
        if queue_s is None:
            queue_s = (started or now) - created
        elapsed_s = record.get("elapsed_s")
        if elapsed_s is None and started is not None:
            elapsed_s = (finished or now) - started
        error = record.get("error") or None
        return {
            "job_id": record.get("job_id"),
            "session_id": record.get("session_id"),
            "state": record.get("state"),
            "queue_position": self.queue_position(state, str(record.get("job_id"))),
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "queue_s": round(float(queue_s), 3) if queue_s is not None else None,
            "elapsed_s": round(float(elapsed_s), 3) if elapsed_s is not None else None,
            "steps_total": record.get("steps_total"),
            "steps_done": record.get("steps_done"),
            "step_kinds": record.get("step_kinds"),
            "cancel_requested": bool(record.get("cancel_requested")),
            "commit": record.get("commit"),
            "correctness": record.get("correctness"),
            "error": error,
        }

    def status(self, session: str | None = None, limit: int | None = None) -> dict[str, Any]:
        state = self.read()
        queued_total = sum(
            1
            for job_id in state["order"]
            if (state["jobs"].get(job_id) or {}).get("state") == "queued"
        )
        jobs = []
        for job_id in state["order"]:
            record = state["jobs"].get(job_id)
            if not record:
                continue
            if session is not None and record.get("session_id") != session:
                continue
            jobs.append(self.public_view(record, state))
        if limit is not None:
            jobs = jobs[-int(limit) :]
        return {
            "serve": self.serve_health(state),
            "halted": state.get("halted"),
            "queue_depth": queued_total,
            "jobs": jobs,
        }

    # ------------------------------------------------------------- 执行器
    def take_next(self) -> dict[str, Any] | None:
        """原子地把队首 queued job 迁到 running 并返回记录。"""

        def _take(state: dict[str, Any]) -> dict[str, Any] | None:
            for job_id in state["order"]:
                record = state["jobs"].get(job_id)
                if not record or record.get("state") != "queued":
                    continue
                if record.get("cancel_requested"):
                    # 正常路径下 cancel 已经落成 cancelled；这里只做防御。
                    record["state"] = "cancelled"
                    record["finished_at"] = now_iso()
                    record["queue_s"] = (
                        (parse_iso(record.get("started_at")) or time.time())
                        - (parse_iso(record.get("created_at")) or time.time())
                    )
                    record["elapsed_s"] = 0.0
                    continue
                started = now_iso()
                created = parse_iso(record.get("created_at")) or time.time()
                record["state"] = "running"
                record["started_at"] = started
                record["queue_s"] = max(0.0, parse_iso(started) - created)
                record["elapsed_s"] = None
                record["steps_done"] = 0
                return dict(record)
            return None

        return self.mutate(_take)

    def transition(self, job_id: str, new_state: str, **fields: Any) -> dict[str, Any]:
        def _apply(state: dict[str, Any]) -> dict[str, Any]:
            record = state["jobs"].get(job_id)
            if record is None:
                raise StoreError(f"未知 job: {job_id}")
            current = str(record.get("state"))
            if new_state != current and new_state not in ALLOWED_TRANSITIONS.get(
                current, frozenset()
            ):
                raise StoreError(f"非法状态迁移 {current} -> {new_state}（{job_id}）")
            record.update(fields)
            record["state"] = new_state
            return dict(record)

        return self.mutate(_apply)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        def _apply(state: dict[str, Any]) -> dict[str, Any]:
            record = state["jobs"].get(job_id)
            if record is None:
                raise StoreError(f"未知 job: {job_id}")
            record.update(fields)
            return dict(record)

        return self.mutate(_apply)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        """只设置 cancel_requested（running 用）；不改状态。"""
        return self.update_job(job_id, cancel_requested=True, cancel_requested_at=now_iso())

    # ------------------------------------------------------------- serve 记录
    def set_serve(self, pid: int, extra: dict[str, Any] | None = None) -> None:
        def _apply(state: dict[str, Any]) -> None:
            previous = state.get("serve") or {}
            state["serve"] = {
                "pid": pid,
                "hostname": socket.gethostname(),
                "started_at": previous.get("started_at") if previous.get("pid") == pid else now_iso(),
                "heartbeat_at": now_iso(),
                "running": True,
            }
            if extra:
                state["serve"].update(extra)

        self.mutate(_apply)

    def heartbeat(self) -> None:
        def _apply(state: dict[str, Any]) -> None:
            serve = state.get("serve")
            if not serve:
                return
            serve["heartbeat_at"] = now_iso()
            serve["running"] = True

        self.mutate(_apply)

    def clear_serve(self, pid: int) -> None:
        def _apply(state: dict[str, Any]) -> None:
            serve = state.get("serve") or {}
            if serve.get("pid") != pid:
                return
            serve["running"] = False
            serve["heartbeat_at"] = now_iso()
            serve["stopped_at"] = now_iso()
            state["serve"] = serve

        self.mutate(_apply)

    def serve_health(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state if state is not None else self.read()
        serve = state.get("serve") or {}
        held = self.lock_is_held(self.serve_lock_path)
        heartbeat = parse_iso(serve.get("heartbeat_at"))
        age = time.time() - heartbeat if heartbeat is not None else None
        return {
            "running": held,
            "lock_held": held,
            "pid": serve.get("pid"),
            "started_at": serve.get("started_at"),
            "heartbeat_at": serve.get("heartbeat_at"),
            "heartbeat_age_s": round(age, 3) if age is not None else None,
        }

    # ------------------------------------------------------------- halt
    def set_halt(self, reason: str, detail: Any = None, job_id: str | None = None) -> None:
        def _apply(state: dict[str, Any]) -> None:
            state["halted"] = {
                "reason": reason,
                "detail": detail,
                "job_id": job_id,
                "at": now_iso(),
                "notified": False,
            }

        self.mutate(_apply)

    def clear_halt(self) -> dict[str, Any] | None:
        def _apply(state: dict[str, Any]) -> dict[str, Any] | None:
            previous = state.get("halted")
            state["halted"] = None
            return previous

        return self.mutate(_apply)

    # ------------------------------------------------------------- flock 工具
    @staticmethod
    def lock_is_held(path: Path) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def acquire_lock_file(self, path: Path, *, timeout: float = 0.0) -> int:
        """拿一个长期持有的 flock（serve / gpu）；返回 fd，调用方负责 unlock。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise ServeBusy(f"锁已被占用: {path}") from None
                time.sleep(0.05)

    @staticmethod
    def release_lock_file(fd: int | None) -> None:
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)
