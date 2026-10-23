"""实验执行器：worktree、进程组、超时/取消、产物校验、通知与重启核对。

设计：
- 每次 submit 对应一个 detached worktree（指定 commit）；执行目录只读源码，
  ``.venv`` 与 ``models`` 用符号链接指向 GPU 主机上的现有环境，不复制权重。
- 每个步骤独立进程组（``start_new_session``）；超时/取消时先 SIGTERM、再
  SIGKILL，并确认整个进程组消失后才释放 GPU 锁、执行下一项。确认不了就停队列。
- 步骤必须退出码 0 且产物通过 ``artifacts`` 校验才继续；失败/超时/取消都不
  自动重跑。
- 队列串行 + ``serve.lock`` 单执行器 + ``gpu.lock`` job 级独占；启动前用
  ``nvidia-smi`` 检查是否已有别的计算进程占用 GPU，有就停队列而不是抢设备。
- 执行器重启时核对上次记录的 pid + /proc starttime：存活则停队列等人工处理，
  已退出且无完整结果则标 interrupted。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.experiment import artifacts, schema
from runner.experiment.store import (
    TERMINAL,
    ServeBusy,
    Store,
    StoreError,
    now_iso,
    parse_iso,
    read_json,
    write_json_atomic,
)

POLL_S = 0.2
CANCEL_CHECK_S = 1.0
HEARTBEAT_S = 2.0
TERM_GRACE_S = 10.0
KILL_GRACE_S = 20.0
LOCK_CONFIRM_S = 5.0
NOTIFY_TIMEOUT_S = 120.0
GPU_GUARD_INTERVAL_S = 5.0
# 刚被杀掉的 CUDA 进程释放 12 GB context 有秒级延迟；先等它退干净，持续占用才停队列。
GPU_DRAIN_S = 10.0


class Halt(RuntimeError):
    """无法确认 GPU 可用/进程已退出：停队列，记录原因并通知 Coordinator。"""

    def __init__(self, reason: str, detail: Any = None, job_id: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.job_id = job_id


class JobAbort(RuntimeError):
    """job 的终态原因（failed / timed_out / cancelled / interrupted）。"""

    def __init__(
        self,
        state: str,
        reason: str,
        *,
        detail: Any = None,
        step: dict[str, Any] | None = None,
        category: str = "invalid_artifact",
    ):
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.detail = detail
        self.step = step
        self.category = category


# ------------------------------------------------------------------ /proc 工具
def proc_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    try:
        rest = stat.rsplit(") ", 1)[1].split()
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def pid_alive(pid: int | None, start_ticks: int | None = None) -> bool:
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    try:
        rest = stat.rsplit(") ", 1)[1].split()
        if rest[0] == "Z":
            return False
        if start_ticks is not None and int(rest[19]) != int(start_ticks):
            return False
    except (IndexError, ValueError):
        return False
    return True


def pgroup_members(pgid: int | None) -> list[int]:
    """进程组里仍存活的（非僵尸）pid。

    不用 ``killpg(pgid, 0)``：僵尸也留在进程组里，会让“确认退出”一直失败；
    孤儿进程被 init 回收前是僵尸，但已经不再占用 GPU。
    """
    if not pgid:
        return []
    members: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return members
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text(
                encoding="utf-8", errors="replace"
            )
            rest = stat.rsplit(") ", 1)[1].split()
            if rest[0] == "Z":
                continue
            if int(rest[2]) == int(pgid):
                members.append(int(entry))
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError, OSError):
            continue
    return members


def pgid_alive(pgid: int | None) -> bool:
    return bool(pgroup_members(pgid))


def _smi(args: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_compute_apps() -> dict[str, Any]:
    """当前 GPU 上的计算进程；不可用时 available=false（CPU 场景不阻塞）。"""
    proc = _smi(
        [
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None:
        return {"available": False, "reason": "nvidia-smi 不可用"}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": f"nvidia-smi 退出码 {proc.returncode}",
            "stderr": proc.stderr.strip()[:400],
        }
    apps = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        apps.append(
            {
                "pid": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "used_memory_mib": parts[2],
            }
        )
    return {"available": True, "apps": apps}


def gpu_guard(args: Any) -> dict[str, Any]:
    """确认没有别的计算进程占用 GPU；离开前给刚退出的 CUDA 进程一点释放时间。"""
    if not getattr(args, "gpu_guard", True):
        return {"checked": False}
    info = gpu_compute_apps()
    if not info.get("available"):
        # 无法确认 GPU 是否空闲时必须停队列：宁可等人工确认，也不能和残留进程抢卡。
        info["checked"] = True
        raise Halt(
            "无法用 nvidia-smi 确认 GPU 是否空闲，停队列", info
        )
    if not info.get("apps"):
        info["checked"] = True
        return info
    started = time.monotonic()
    while time.monotonic() - started < GPU_DRAIN_S:
        time.sleep(0.5)
        info = gpu_compute_apps()
        if not info.get("available") or not info.get("apps"):
            info["waited_s"] = round(time.monotonic() - started, 2)
            info["checked"] = True
            return info
    info["waited_s"] = round(time.monotonic() - started, 2)
    raise Halt(
        f"GPU 上已有其他计算进程（等待 {GPU_DRAIN_S:.0f}s 仍未释放），停队列（不抢设备）",
        info,
    )


def gpu_snapshot() -> dict[str, Any]:
    proc = _smi(
        [
            "--query-gpu=name,memory.total,memory.used,driver_version,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None or proc.returncode != 0:
        return {"available": False}
    parts = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
    keys = ("name", "memory_total_mib", "memory_used_mib", "driver", "clock_sm_mhz", "power_w", "temperature_c")
    return {"available": True, **dict(zip(keys, parts))}


def light_environment(root: Path) -> dict[str, Any]:
    """执行器视角的环境（不 import torch，避免 serve 自己占用 GPU 上下文）。"""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "root": str(root),
        "venv": str(root / ".venv"),
        "cuda_home": os.environ.get("CUDA_HOME")
        or str(Path.home() / "micromamba/envs/cuda13"),
        "gpu": gpu_snapshot(),
    }
    try:
        from runner import report as reporting

        info["checkpoint"] = {
            "path": str(root / reporting.CHECKPOINT),
            "revision": reporting.CHECKPOINT_REVISION,
        }
        info["sol_bandwidth_bytes"] = reporting.SOL_BANDWIDTH_BYTES
    except Exception as exc:  # noqa: BLE001 - 记录但不阻断执行
        info["checkpoint"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


# ------------------------------------------------------------------ worktree
def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )


def _venv_source(root: Path) -> Path | None:
    candidate = root / ".venv"
    if candidate.exists():
        return candidate
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    return None


def prepare_worktree(store: Store, job: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """为 job 建 detached worktree，并链接共享环境。"""
    job_id = job["job_id"]
    commit = job["commit"]
    worktree = store.worktree_dir(job_id)
    if worktree.exists():
        raise JobAbort(
            "failed",
            f"worktree 已存在，拒绝复用: {worktree}",
            detail={"worktree": str(worktree)},
        )
    store.worktree_root.mkdir(parents=True, exist_ok=True)
    proc = _git(store.root, "worktree", "add", "--detach", str(worktree), commit)
    if proc.returncode != 0:
        raise JobAbort(
            "failed",
            f"git worktree add 失败: {proc.stderr.strip()[:300]}",
            detail={"stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]},
        )
    head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise JobAbort(
            "failed",
            f"worktree HEAD={head[:12]} 与请求 commit={commit[:12]} 不一致",
            detail={"worktree": str(worktree)},
        )

    links: list[dict[str, str]] = []
    venv = _venv_source(store.root)
    if venv is not None:
        link = worktree / ".venv"
        if not link.exists():
            link.symlink_to(venv, target_is_directory=True)
            links.append({".venv": str(venv)})
    models = store.root / "models"
    if models.exists():
        link = worktree / "models"
        if not link.exists():
            link.symlink_to(models, target_is_directory=True)
            links.append({"models": str(models)})

    dirty = _git(worktree, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise JobAbort(
            "failed",
            "worktree 在 checkout 后有 tracked 改动，结果不可复现",
            detail={"status": dirty.splitlines()[:20]},
        )
    info = {
        "path": str(worktree),
        "commit": head,
        "symlinks": links,
        "clean": True,
        "removed": False,
    }
    return worktree, info


def remove_worktree(store: Store, job_id: str) -> bool:
    worktree = store.worktree_dir(job_id).resolve()
    root = store.worktree_root.resolve()
    if root not in worktree.parents:
        return False
    proc = _git(store.root, "worktree", "remove", "--force", str(worktree))
    if proc.returncode != 0:
        shutil.rmtree(worktree, ignore_errors=True)
        _git(store.root, "worktree", "prune")
    return not worktree.exists()


# ------------------------------------------------------------------ 环境
def build_step_env(
    store: Store,
    job: dict[str, Any],
    request: schema.Request,
    worktree: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    venv = store.root / ".venv"
    site = next(iter((venv / "lib").glob("python*/site-packages")), None)
    cuda_home = Path(env.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13")
    cpath: list[str] = []
    library: list[str] = []
    if site is not None:
        cpath.append(str(site / "nvidia/cu13/include"))
        library.append(str(site / "nvidia/cu13/lib"))
    cpath.append(str(cuda_home / "targets/x86_64-linux/include"))
    library += [
        str(cuda_home / "targets/x86_64-linux/lib"),
        str(cuda_home / "lib"),
    ]
    if env.get("CPATH"):
        cpath.append(env["CPATH"])
    if env.get("LIBRARY_PATH"):
        library.append(env["LIBRARY_PATH"])
    pythonpath = [str(worktree)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "VENV": str(venv),
            "CPATH": os.pathsep.join(cpath),
            "LIBRARY_PATH": os.pathsep.join(library),
            "PATH": os.pathsep.join([str(cuda_home / "bin"), env.get("PATH", "")]),
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "FLASHINFER_WORKSPACE_BASE": env.get(
                "FLASHINFER_WORKSPACE_BASE", str(store.root / ".tmp/flashinfer")
            ),
            "EXPERIMENT_JOB_ID": job["job_id"],
            "EXPERIMENT_SESSION_ID": job["session_id"],
            "EXPERIMENT_COMMIT": job["commit"],
            "EXPERIMENT_OUTPUT_DIR": job["output_dir"],
            "EXPERIMENT_STATE_DIR": str(store.state_dir),
        }
    )
    return env


def env_summary(env: dict[str, str]) -> dict[str, Any]:
    keys = (
        "CUDA_HOME",
        "VENV",
        "CPATH",
        "LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "FLASHINFER_WORKSPACE_BASE",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key: env.get(key) for key in keys if key in env}


# ------------------------------------------------------------------ 进程控制
def wait_child(
    store: Store,
    job_id: str,
    proc: subprocess.Popen,
    deadline_monotonic: float,
    stop: threading.Event,
) -> tuple[str, int | None]:
    last_cancel = 0.0
    last_beat = 0.0
    while True:
        code = proc.poll()
        if code is not None:
            return "exited", int(code)
        now = time.monotonic()
        if now >= deadline_monotonic:
            return "timeout", None
        if stop.is_set():
            return "interrupt", None
        if now - last_cancel >= CANCEL_CHECK_S:
            try:
                job = store.get(job_id)
            except StoreError:
                job = None
            if job and job.get("cancel_requested"):
                return "cancel", None
            last_cancel = now
        if now - last_beat >= HEARTBEAT_S:
            try:
                store.heartbeat()
            except StoreError:
                pass
            last_beat = now
        time.sleep(POLL_S)


def terminate_group(proc: subprocess.Popen, pgid: int) -> bool:
    """SIGTERM -> SIGKILL，返回整个进程组是否确认退出。"""
    if proc.poll() is not None and not pgid_alive(pgid):
        return True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None or pgid_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + LOCK_CONFIRM_S
    while time.monotonic() < deadline:
        if proc.poll() is not None and not pgid_alive(pgid):
            return True
        time.sleep(0.05)
    return proc.poll() is not None and not pgid_alive(pgid)


def kill_strays(pgid: int, keep_pid: int) -> list[int]:
    """步骤结束后，清理同组里残留的后台进程（避免污染下一步的测量）。"""
    strays = [pid for pid in pgroup_members(pgid) if pid != keep_pid]
    for pid in strays:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return strays


def cleanup_job_group(keeper: subprocess.Popen | None, pgid: int | None) -> bool:
    """job 结束时杀掉并确认 job 级进程组消失。

    组 leader 是一个只 sleep 的 keeper：它让所有步骤共享同一个进程组，
    这样超时/取消/残留后台进程都能被一次 killpg 覆盖。
    """
    if keeper is None or not pgid:
        return True
    members = pgroup_members(pgid)
    if members:
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        keeper.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + LOCK_CONFIRM_S
    while time.monotonic() < deadline:
        if not pgroup_members(pgid):
            return True
        time.sleep(0.05)
    return not pgroup_members(pgid)


def verify_argv_isolation(
    argv: list[str], root: Path, worktree: Path, output_dir: Path
) -> None:
    """拒绝 argv 里任何指向本仓库 worktree 的路径，保证步骤只读提交的 commit。

    绝对路径同时按词法和 resolve 后检查；相对路径先按 worktree 词法归一化，
    逃逸出 worktree 且落回仓库源码时拒绝。worktree 内的 `.venv` 是符号链接，
    词法在 worktree 内就放行。
    """
    wt = Path(worktree).resolve()
    out = Path(output_dir).resolve()
    bases = [base for base in schema.repo_worktrees(root) if base != wt]

    def under(path: Path, parents: list[Path]) -> bool:
        return any(path == parent or parent in path.parents for parent in parents)

    for token in argv:
        if token.startswith("-"):
            continue
        path = Path(token)
        if path.is_absolute():
            lexical = Path(os.path.normpath(token))
            resolved = lexical.resolve(strict=False)
            if under(lexical, [out]) or under(resolved, [out]):
                continue
            if under(lexical, [wt]) or under(resolved, [wt]):
                continue
            if under(lexical, bases) or under(resolved, bases):
                raise JobAbort(
                    "failed",
                    f"argv 含仓库内绝对路径 {token!r}，会绕过 job worktree，拒绝执行",
                    detail={"token": token, "worktree": str(wt)},
                )
            continue
        lexical = Path(os.path.normpath(str(wt / path)))
        if under(lexical, [wt]):
            continue
        resolved = lexical.resolve(strict=False)
        if under(resolved, bases):
            raise JobAbort(
                "failed",
                f"argv 相对路径 {token!r} 逃逸出 job worktree 并落回仓库源码，拒绝执行",
                detail={"token": token, "resolved": str(resolved)},
            )


# ------------------------------------------------------------------ 步骤执行
def run_step(
    store: Store,
    job: dict[str, Any],
    request: schema.Request,
    step: schema.ResolvedStep,
    worktree: Path,
    env: dict[str, str],
    output_dir: Path,
    logs_dir: Path,
    deadline_monotonic: float,
    stop: threading.Event,
    step_records: list[dict[str, Any]],
    pgid: int,
    keeper_pid: int,
) -> dict[str, Any]:
    argv = step.expand_argv(output_dir)
    result_path = step.result_path(output_dir)
    expect = schema.Expectation.build(step, request, output_dir, worktree)
    script_path = worktree / step.script
    index = step.index
    record: dict[str, Any] = {
        "index": index,
        "kind": step.kind,
        "argv": argv,
        "script": step.script,
        "result": str(result_path),
        "started_at": now_iso(),
        "finished_at": None,
        "duration_s": None,
        "exit_code": None,
        "stdout": str(logs_dir / f"{index:02d}-{step.kind}.stdout.log"),
        "stderr": str(logs_dir / f"{index:02d}-{step.kind}.stderr.log"),
        "pid": None,
        "pgid": None,
        "validation": None,
    }
    step_records.append(record)
    verify_argv_isolation(argv, store.root, worktree, output_dir)
    resolved_script = script_path.resolve()
    worktree_resolved = worktree.resolve()
    if not (
        resolved_script == worktree_resolved
        or worktree_resolved in resolved_script.parents
    ):
        raise JobAbort(
            "failed",
            f"步骤 {index}: 脚本 {step.script} 解析到 job worktree 之外",
            step=record,
            detail={"script": str(resolved_script), "worktree": str(worktree_resolved)},
        )
    if not script_path.exists():
        raise JobAbort(
            "failed",
            f"步骤 {index}: worktree 里没有脚本 {step.script}",
            step=record,
            detail={"worktree": str(worktree), "script": step.script},
        )

    started = time.monotonic()
    stdout_path = Path(record["stdout"])
    stderr_path = Path(record["stderr"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as out, open(
        stderr_path, "w", encoding="utf-8"
    ) as err:
        if not pid_alive(keeper_pid):
            raise JobAbort(
                "failed",
                f"步骤 {index}: job 进程组 leader {keeper_pid} 已退出，拒绝继续",
                step=record,
                detail={"pgid": pgid, "keeper_pid": keeper_pid},
            )
        try:
            proc = subprocess.Popen(
                argv,
                cwd=worktree,
                env=env,
                stdout=out,
                stderr=err,
                process_group=pgid,
                close_fds=True,
            )
        except OSError as exc:
            raise JobAbort(
                "failed",
                f"步骤 {index}: 无法启动 {argv[0]!r}: {exc}",
                step=record,
            ) from exc
        record.update({"pid": proc.pid, "pgid": pgid})
        reason, code = wait_child(store, job["job_id"], proc, deadline_monotonic, stop)
        if reason != "exited":
            confirmed = terminate_group(proc, pgid)
            record.update(
                {
                    "finished_at": now_iso(),
                    "duration_s": time.monotonic() - started,
                    "terminated": reason,
                    "exit_code": proc.poll(),
                }
            )
            if not confirmed:
                raise Halt(
                    f"步骤 {index} 的进程组 {pgid} 在 SIGKILL 后仍未确认退出",
                    {"job_id": job["job_id"], "pid": proc.pid, "pgid": pgid},
                    job["job_id"],
                )
            state = {"timeout": "timed_out", "cancel": "cancelled", "interrupt": "interrupted"}[
                reason
            ]
            label = {"timeout": "超出 timeout_s", "cancel": "收到取消请求", "interrupt": "执行器收到 SIGTERM"}[
                reason
            ]
            raise JobAbort(
                state,
                f"步骤 {index} ({step.kind}) {label}，已终止进程组 {pgid}",
                step=record,
                detail={"pid": proc.pid, "pgid": pgid, "confirmed_exit": confirmed},
            )
        record["exit_code"] = int(code or 0)
        record["finished_at"] = now_iso()
        record["duration_s"] = time.monotonic() - started
        # 步骤自己应该把子进程收干净；同组残留会被杀掉并记录，避免污染下一步。
        record["stray_pids_killed"] = kill_strays(pgid, keeper_pid)

    # 即使退出码非 0 也要尝试校验产物：真实 m3_check 在 gate 失败时返回 1，
    # 只有读到 report.json 才能把 correctness 判成 failed 而不是 not_checked。
    artifact_error: artifacts.ArtifactError | None = None
    validation: dict[str, Any] | None = None
    try:
        validation = artifacts.validate(step.kind, expect)
    except artifacts.ArtifactError as exc:
        artifact_error = exc
        validation = {
            "ok": False,
            "kind": step.kind,
            "category": exc.category,
            "reason": exc.reason,
            "detail": exc.detail,
            "correctness": "failed" if exc.category == "gate_failed" else "not_checked",
        }
    record["validation"] = validation
    if artifact_error is not None:
        reason = f"步骤 {index} ({step.kind}) 产物校验失败: {artifact_error.reason}"
        if record["exit_code"] != 0:
            reason += f"（退出码 {record['exit_code']}）"
        raise JobAbort(
            "failed",
            reason,
            step=record,
            detail=artifact_error.detail,
            category=artifact_error.category,
        ) from artifact_error
    if record["exit_code"] != 0:
        raise JobAbort(
            "failed",
            f"步骤 {index} ({step.kind}) 退出码 {record['exit_code']}",
            step=record,
            detail={"stderr_tail": _tail(stderr_path), "stdout_tail": _tail(stdout_path)},
        )
    record["result_sha256"] = _sha256_file(result_path)
    return record


def _tail(path: Path, limit: int = 1200) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return data[-limit:]


def _sha256_file(path: Path) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# ------------------------------------------------------------------ 结果
def compute_correctness(
    step_records: list[dict[str, Any]], request: schema.Request | None
) -> str:
    """只有**请求里的全部 check** 都执行且通过才能算 passed。

    只看已执行记录会把“第一个 check 通过、后面的 check 根本没跑”误判成 passed；
    这里必须用 request 的步骤集合做覆盖判断。
    """
    checks = [record for record in step_records if record.get("kind") == "check"]
    for record in checks:
        validation = record.get("validation") or {}
        if validation.get("correctness") == "failed":
            return "failed"
    if request is None:
        return "not_checked"
    requested = {step.index for step in request.steps if step.kind == "check"}
    passed = {
        record.get("index")
        for record in checks
        if (record.get("validation") or {}).get("ok")
    }
    if requested and requested == passed:
        return "passed"
    return "not_checked"


def merge_metrics(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    check: dict[str, Any] = {
        "scenarios": [],
        "gates_total": 0,
        "gates_passed": 0,
        "gate_failures": [],
        "diagnostics_failed": [],
    }
    has_check = False
    for record in step_records:
        validation = record.get("validation") or {}
        kind = record["kind"]
        # check 即使 gate 失败，产物里的 gate 计数/失败项也是有意义的结果；
        # bench/memory/profile 的失败产物不完整，不并入 metrics。
        if not validation.get("ok") and kind != "check":
            continue
        data = validation.get("metrics") or {}
        if kind == "check":
            has_check = True
            check["scenarios"].extend(data.get("scenarios") or [])
            check["gates_total"] += int(data.get("gates_total") or 0)
            check["gates_passed"] += int(data.get("gates_passed") or 0)
            check["gate_failures"].extend(data.get("gate_failures") or [])
            check["diagnostics_failed"].extend(data.get("diagnostics_failed") or [])
            check["fixtures_sha256"] = data.get("fixtures_sha256")
            check["steps"] = data.get("steps")
        else:
            metrics.setdefault(kind, []).extend(data.get("scenarios") or [])
    if has_check:
        metrics["check"] = check
    return metrics


def build_summary(
    store: Store,
    job: dict[str, Any],
    request: schema.Request | None,
    execution: dict[str, Any],
    state: str,
    error: dict[str, Any] | None,
    step_records: list[dict[str, Any]],
    queue_s: float | None,
    elapsed_s: float | None,
) -> dict[str, Any]:
    output_dir = Path(job["output_dir"])
    commands: list[list[str]] = []
    if request is not None:
        for step in request.steps:
            commands.append(step.expand_argv(output_dir))
    else:
        commands = [list(record.get("argv") or []) for record in step_records]
    return {
        "job_id": job["job_id"],
        "session_id": job["session_id"],
        "state": state,
        "commit": job["commit"],
        "commit_requested": job.get("commit_requested"),
        "commands": commands,
        "correctness": compute_correctness(step_records, request),
        "metrics": merge_metrics(step_records),
        "queue_s": round(queue_s, 3) if queue_s is not None else None,
        "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
        "timeout_s": job.get("timeout_s"),
        "error": error,
        "output_dir": str(store.job_dir(job["job_id"])),
        "request_path": str(store.request_path(job["job_id"])),
        "execution_path": str(store.execution_path(job["job_id"])),
        "summary_path": str(store.summary_path(job["job_id"])),
        "started_at": job.get("started_at"),
        "finished_at": now_iso() if state in TERMINAL else None,
        "steps": step_records,
        "artifacts": {
            "output": str(output_dir),
            "logs": str(store.logs_dir(job["job_id"])),
            "logs_files": sorted(str(path) for path in store.logs_dir(job["job_id"]).glob("*.log")),
        },
        "checkpoint": (execution.get("environment") or {}).get("checkpoint"),
        "environment": execution.get("environment"),
        "git": execution.get("worktree"),
        "notification": execution.get("notification"),
    }


def error_from_abort(
    abort: JobAbort, step_records: list[dict[str, Any]]
) -> dict[str, Any]:
    step = abort.step or {}
    index = step.get("index")
    if index is None and step_records:
        index = step_records[-1].get("index")
    kind = step.get("kind")
    if kind is None and index is not None and index < len(step_records):
        kind = step_records[index].get("kind")
    return {
        "step": index,
        "kind": kind,
        "reason": abort.reason,
        "category": abort.category,
        "detail": abort.detail,
        "log": {
            "stdout": step.get("stdout"),
            "stderr": step.get("stderr"),
        }
        if step
        else None,
    }


# ------------------------------------------------------------------ 通知
SUMMARY_TEXT = (
    "[experiment] job={job_id} state={state} correctness={correctness} "
    "session={session_id} commit={commit}\n"
    "result: .venv/bin/python scripts/experiment.py result {job_id}\n"
    "output: {output_dir}"
)


def _short_summary(summary: dict[str, Any]) -> str:
    return SUMMARY_TEXT.format(
        job_id=summary.get("job_id"),
        state=summary.get("state"),
        correctness=summary.get("correctness"),
        session_id=summary.get("session_id"),
        commit=str(summary.get("commit"))[:12],
        output_dir=summary.get("output_dir"),
    )


def _run_notifier(
    store: Store,
    command: str,
    payload: dict[str, Any],
    log_path: Path,
    timeout: float = NOTIFY_TIMEOUT_S,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "JOB_ID": str(payload.get("job_id") or ""),
            "SESSION_ID": str(payload.get("session_id") or ""),
            "JOB_STATE": str(payload.get("state") or ""),
            "CORRECTNESS": str(payload.get("correctness") or ""),
            "OUTPUT_DIR": str(payload.get("output_dir") or ""),
            "SUMMARY_PATH": str(payload.get("summary_path") or ""),
            "SUMMARY_TEXT": _short_summary(payload),
            "SUMMARY_JSON": json.dumps(payload, ensure_ascii=False),
        }
    )
    started = time.monotonic()
    code: int | None = None
    stdout = ""
    stderr = ""
    # 用 Popen + 轮询而不是 subprocess.run：通知可能跑几十秒，期间要维持 serve 心跳，
    # 否则 wait 会把“正在发通知”误判成执行器停顿。输出写临时文件，避免管道塞满。
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        try:
            proc = subprocess.Popen(
                ["/bin/sh", "-c", command],
                cwd=store.root,
                env=env,
                stdout=out_file,
                stderr=err_file,
            )
        except OSError as exc:
            proc = None
            stderr = f"无法启动通知命令: {exc}"
        if proc is not None:
            deadline = time.monotonic() + timeout
            last_beat = 0.0
            while proc.poll() is None and time.monotonic() < deadline:
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_S:
                    try:
                        store.heartbeat()
                    except StoreError:
                        pass
                    last_beat = now
                time.sleep(POLL_S)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
                stderr = f"通知命令超时（{timeout}s）"
            else:
                code = int(proc.returncode)
                out_file.seek(0)
                err_file.seek(0)
                stdout = out_file.read().decode("utf-8", "replace")
                stderr = err_file.read().decode("utf-8", "replace")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            f"=== {now_iso()} rc={code} {time.monotonic() - started:.2f}s\n"
            f"$ {command}\n[stdout]\n{stdout}\n[stderr]\n{stderr}\n"
        )
    return {
        "state": "delivered" if code == 0 else "failed",
        "attempts": 1,
        "at": now_iso(),
        "command": command,
        "returncode": code,
        "duration_s": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "stdout_tail": stdout[-400:] if stdout else "",
        "stderr_tail": stderr[-400:] if stderr else "",
    }


def _update_summary_notification(store: Store, job_id: str, note: dict[str, Any]) -> None:
    path = store.summary_path(job_id)
    if not path.exists():
        return
    try:
        summary = read_json(path)
    except Exception:  # noqa: BLE001
        return
    summary["notification"] = note
    write_json_atomic(path, summary)


def notify_job(store: Store, job_id: str, command: str | None) -> dict[str, Any] | None:
    """把短摘要送回原 session；失败不改结果，也不触发重跑。按 job 去重。"""
    job = store.get(job_id)
    if job is None or job.get("state") not in TERMINAL:
        return None
    note = job.get("notification") or {}
    if note.get("state") in ("delivered", "failed"):
        return note
    summary_path = store.summary_path(job_id)
    if summary_path.exists():
        try:
            payload = read_json(summary_path)
        except Exception:  # noqa: BLE001
            payload = {}
    else:
        payload = {}
    payload.setdefault("job_id", job_id)
    payload.setdefault("session_id", job.get("session_id"))
    payload.setdefault("state", job.get("state"))
    payload.setdefault("output_dir", job.get("output_dir"))
    payload.setdefault("summary_path", str(summary_path))
    if not command:
        note = {
            "state": "pending",
            "attempts": 0,
            "at": now_iso(),
            "reason": "未配置通知命令；Coordinator 用 wait/result 收取并转发",
        }
    else:
        try:
            note = _run_notifier(
                store, command, payload, store.logs_dir(job_id) / "notify.log"
            )
        except Exception as exc:  # noqa: BLE001 - 通知失败不改变实验结果
            note = {
                "state": "failed",
                "attempts": 1,
                "at": now_iso(),
                "command": command,
                "returncode": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    store.update_job(job_id, notification=note)
    _update_summary_notification(store, job_id, note)
    return note


def deliver_pending(store: Store, command: str | None) -> None:
    """执行器重启或 queued job 被取消后，补投一次尚未投递的通知。"""
    if not command:
        return
    for job in store.ordered_jobs():
        if job.get("state") not in TERMINAL:
            continue
        note = job.get("notification") or {}
        if note.get("attempts", 0) >= 1:
            continue
        try:
            notify_job(store, job["job_id"], command)
        except Exception:  # noqa: BLE001 - 通知失败不影响队列
            continue


def notify_halt(store: Store, command: str | None, halt: dict[str, Any] | None = None) -> None:
    if not command:
        return
    state = store.read()
    record = halt or state.get("halted")
    if not record or record.get("notified"):
        return
    payload = {
        "job_id": record.get("job_id") or "",
        "session_id": "",
        "state": "halted",
        "correctness": "",
        "output_dir": str(store.state_dir),
        "summary_path": str(store.state_dir / "queue.json"),
    }
    try:
        note = _run_notifier(
            store, command, payload, store.state_dir / "halt-notify.log"
        )
    except Exception as exc:  # noqa: BLE001
        note = {"state": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    store.mutate(
        lambda current: (current.get("halted") or {}).update(
            {"notified": True, "notification": note}
        )
    )


# ------------------------------------------------------------------ job 主体
def run_job(
    store: Store, job: dict[str, Any], args: Any, stop: threading.Event
) -> dict[str, Any]:
    job_id = job["job_id"]
    job_dir = store.job_dir(job_id)
    output_dir = Path(job["output_dir"])
    logs_dir = store.logs_dir(job_id)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        error = {
            "step": None,
            "reason": f"output 目录非空，拒绝覆盖: {output_dir}",
            "category": "invalid_artifact",
        }
        summary = build_summary(
            store, job, None, {}, "failed", error, [],
            job.get("queue_s"), 0.0,
        )
        write_json_atomic(store.summary_path(job_id), summary)
        store.transition(
            job_id,
            "failed",
            finished_at=now_iso(),
            elapsed_s=0.0,
            error={k: error[k] for k in ("step", "reason", "category")},
            correctness="not_checked",
        )
        notify_job(store, job_id, getattr(args, "notify_cmd", None))
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)

    request: schema.Request | None = None
    step_records: list[dict[str, Any]] = []
    execution: dict[str, Any] = {
        "job_id": job_id,
        "session_id": job.get("session_id"),
        "state": "running",
        "commit": job.get("commit"),
        "commit_requested": job.get("commit_requested"),
        "timeout_s": job.get("timeout_s"),
        "started_at": job.get("started_at"),
        "finished_at": None,
        "queue_s": job.get("queue_s"),
        "elapsed_s": None,
        "serve_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "request_path": str(store.request_path(job_id)),
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "environment": light_environment(store.root),
        "gpu_before": gpu_snapshot(),
        "env": {},
        "worktree": None,
        "steps": step_records,
        "error": None,
        "notification": None,
    }
    write_json_atomic(store.execution_path(job_id), execution)

    deadline = time.monotonic() + float(job.get("timeout_s") or 0.0)
    state = "succeeded"
    error: dict[str, Any] | None = None
    worktree: Path | None = None
    keeper: subprocess.Popen | None = None
    pgid: int | None = None
    try:
        request = schema.parse_request_object(
            read_json(store.request_path(job_id)), store.root
        )
        # 请求入队后不可修改：sha256 + commit 双重核对（文件权限只是第一道门）。
        if request.sha256 != job.get("request_sha256"):
            raise schema.RequestError(
                "request.json 与队列记录的 sha256 不一致，请求入队后被修改"
            )
        if request.commit != job.get("commit"):
            raise schema.RequestError(
                "request.json 的 commit 与队列记录不一致，请求入队后被修改"
            )
        worktree, worktree_info = prepare_worktree(store, job)
        execution["worktree"] = worktree_info
        env = build_step_env(store, job, request, worktree)
        execution["env"] = env_summary(env)
        if time.monotonic() >= deadline:
            raise JobAbort(
                "timed_out",
                "整个 job 的 timeout_s 已用尽（准备阶段）",
                detail={"timeout_s": job.get("timeout_s")},
            )
        # job 级进程组：keeper 是组 leader，所有步骤 join 同一个组。
        # process_group=0：keeper 在 serve 的 session 内新建进程组并当 leader；
        # 后续步骤用 process_group=<keeper pid> join 同一个组（同 session 才允许）。
        keeper = subprocess.Popen(
            ["sleep", str(int(float(job.get("timeout_s") or 0.0)) + 600)],
            process_group=0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pgid = keeper.pid
        keep_ticks = proc_start_ticks(keeper.pid)
        execution["job_group"] = {
            "pid": keeper.pid,
            "pgid": pgid,
            "proc_start_ticks": keep_ticks,
        }
        store.update_job(
            job_id, pid=keeper.pid, pgid=pgid, proc_start_ticks=keep_ticks
        )
        write_json_atomic(store.execution_path(job_id), execution)

        for step in request.steps:
            if stop.is_set():
                raise JobAbort(
                    "interrupted", "执行器收到 SIGTERM，未开始下一步", step=None
                )
            if time.monotonic() >= deadline:
                raise JobAbort(
                    "timed_out",
                    "整个 job 的 timeout_s 已用尽（含准备时间）",
                    step=None,
                    detail={"timeout_s": job.get("timeout_s")},
                )
            run_step(
                store,
                job,
                request,
                step,
                worktree,
                env,
                output_dir,
                logs_dir,
                deadline,
                stop,
                step_records,
                pgid,
                keeper.pid,
            )
            store.update_job(job_id, steps_done=len(step_records))
            write_json_atomic(store.execution_path(job_id), execution)
    except JobAbort as abort:
        state = abort.state
        error = error_from_abort(abort, step_records)
    except schema.RequestError as exc:
        state = "failed"
        error = {
            "step": None,
            "kind": None,
            "reason": f"请求重放校验失败: {exc}",
            "category": "invalid_request",
            "detail": None,
            "log": None,
        }
    except Halt as halt:
        state = "interrupted"
        error = {
            "step": None,
            "kind": None,
            "reason": halt.reason,
            "category": "halted",
            "detail": halt.detail,
            "log": None,
        }
        execution["halt"] = {"reason": halt.reason, "detail": halt.detail}
    except Exception:  # noqa: BLE001 - 记录 traceback，job failed
        state = "failed"
        error = {
            "step": None,
            "kind": None,
            "reason": "执行器异常",
            "category": "internal_error",
            "detail": traceback.format_exc()[-4000:],
            "log": None,
        }

    # 收尾：先清理 job 级进程组（keeper + 任何残留），再写 summary。
    group_cleaned = cleanup_job_group(keeper, pgid)
    execution["job_group_cleaned"] = group_cleaned
    halt_reason = None
    halt_detail = None
    if not group_cleaned:
        # 不管 job 本身是成功还是失败，进程组没确认退出都要停队列。
        halt_reason = f"job 进程组 {pgid} 在 SIGKILL 后仍未确认退出"
        halt_detail = {"pgid": pgid, "job_id": job_id}
        execution["group_cleanup_error"] = halt_reason
        if state == "succeeded":
            state = "interrupted"
            error = {
                "step": None,
                "kind": None,
                "reason": halt_reason,
                "category": "halted",
                "detail": {"pgid": pgid},
                "log": None,
            }
    finished_at = now_iso()
    queue_s = job.get("queue_s")
    elapsed_s = None
    started_at = parse_iso(job.get("started_at"))
    if started_at is not None:
        elapsed_s = time.time() - started_at
    if worktree is not None and state == "succeeded":
        removed = remove_worktree(store, job_id)
        execution["worktree"]["removed"] = removed
        store.update_job(job_id, worktree_removed=removed)
    execution.update(
        {
            "state": state,
            "finished_at": finished_at,
            "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "gpu_after": gpu_snapshot(),
            "error": error,
            "notification": None,
        }
    )
    write_json_atomic(store.execution_path(job_id), execution)
    summary = build_summary(
        store,
        {**job, "started_at": job.get("started_at")},
        request,
        execution,
        state,
        error,
        step_records,
        queue_s,
        elapsed_s,
    )
    write_json_atomic(store.summary_path(job_id), summary)
    record = store.transition(
        job_id,
        state,
        finished_at=finished_at,
        elapsed_s=round(elapsed_s, 3) if elapsed_s is not None else None,
        steps_done=len(step_records),
        error=(
            {key: error.get(key) for key in ("step", "kind", "reason", "category")}
            if error
            else None
        ),
        correctness=summary["correctness"],
        summary_path=str(store.summary_path(job_id)),
        worktree_removed=bool((execution.get("worktree") or {}).get("removed")),
    )
    notify_job(store, job_id, args.notify_cmd)
    execution["notification"] = (store.get(job_id) or {}).get("notification")
    write_json_atomic(store.execution_path(job_id), execution)
    # 失败/中断的 worktree 保留下来便于查日志；success 才删除（见上）。
    if halt_reason is None and error and error.get("category") == "halted":
        halt_reason = str(error.get("reason") or "job 进程组未确认退出")
        halt_detail = error.get("detail")
    if halt_reason:
        raise Halt(halt_reason, halt_detail, job_id)
    return summary


# ------------------------------------------------------------------ 重启核对
def terminate_orphan(
    pid: int | None, pgid: int | None, start_ticks: int | None
) -> bool:
    """执行器不在时取消孤儿进程：按记录的 pid/pgid 杀进程组并确认退出。"""
    if not pid_alive(pid, start_ticks) and not pgid_alive(pgid):
        return True
    if pgid:
        try:
            os.killpg(int(pgid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + TERM_GRACE_S
    while time.monotonic() < deadline:
        if not pid_alive(pid, start_ticks) and not pgid_alive(pgid):
            return True
        time.sleep(0.05)
    if pgid:
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + KILL_GRACE_S
    while time.monotonic() < deadline:
        if not pid_alive(pid, start_ticks) and not pgid_alive(pgid):
            return True
        time.sleep(0.05)
    # 再给僵尸回收一点时间
    deadline = time.monotonic() + LOCK_CONFIRM_S
    while time.monotonic() < deadline:
        if not pid_alive(pid, start_ticks) and not pgid_alive(pgid):
            return True
        time.sleep(0.05)
    return False


def reconcile(store: Store, args: Any) -> list[str]:
    """执行器启动时核对上次的 running job；返回补投通知的 job id。"""
    state = store.read()
    for job_id in list(state["order"]):
        record = state["jobs"].get(job_id) or {}
        if record.get("state") != "running":
            continue
        pid = record.get("pid")
        pgid = record.get("pgid")
        ticks = record.get("proc_start_ticks")
        members = pgroup_members(pgid)
        if pid_alive(pid, ticks) or members:
            detail = {
                "job_id": job_id,
                "pid": pid,
                "pgid": pgid,
                "members": members[:8],
            }
            store.update_job(
                job_id,
                orphan=True,
                orphan_reason=(
                    f"执行器重启时仍有进程存活（keeper={pid}, pgid={pgid}, "
                    f"members={members[:5]}）"
                ),
            )
            raise Halt(
                f"上一个 job {job_id} 的进程组仍有成员存活"
                f"（keeper={pid}, pgid={pgid}, members={members[:5]}），"
                "无法接管，停队列",
                detail,
                job_id,
            )
        summary_path = store.summary_path(job_id)
        recovered = False
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception:  # noqa: BLE001
                summary = {}
            recovered_state = summary.get("state")
            if (
                recovered_state in TERMINAL
                and summary.get("correctness") is not None
                and summary.get("finished_at")
            ):
                store.transition(
                    job_id,
                    recovered_state,
                    recovered=True,
                    finished_at=summary.get("finished_at"),
                    elapsed_s=summary.get("elapsed_s"),
                    correctness=summary.get("correctness"),
                    error=record.get("error"),
                    steps_done=len(summary.get("steps") or []),
                    summary_path=str(summary_path),
                )
                notify_job(store, job_id, getattr(args, "notify_cmd", None))
                recovered = True
        if recovered:
            continue

        finished_at = now_iso()
        error = {
            "step": None,
            "kind": None,
            "reason": f"执行器重启时进程已退出（pid={pid}），缺完整结果",
            "category": "interrupted",
            "detail": {"pid": pid, "pgid": record.get("pgid")},
            "log": None,
        }
        execution_path = store.execution_path(job_id)
        execution: dict[str, Any] = {}
        if execution_path.exists():
            try:
                execution = read_json(execution_path)
            except Exception:  # noqa: BLE001
                execution = {}
        execution.update(
            {
                "job_id": job_id,
                "session_id": record.get("session_id"),
                "state": "interrupted",
                "commit": record.get("commit"),
                "finished_at": finished_at,
                "error": error,
                "recovered": True,
            }
        )
        write_json_atomic(execution_path, execution)
        request = None
        if store.request_path(job_id).exists():
            try:
                request = schema.parse_request_object(
                    read_json(store.request_path(job_id)), store.root
                )
            except Exception:  # noqa: BLE001
                request = None
        step_records = execution.get("steps") or []
        summary = build_summary(
            store,
            record,
            request,
            execution,
            "interrupted",
            error,
            step_records,
            record.get("queue_s"),
            None,
        )
        write_json_atomic(summary_path, summary)
        store.transition(
            job_id,
            "interrupted",
            finished_at=finished_at,
            error={key: error.get(key) for key in ("step", "kind", "reason", "category")},
            correctness=summary["correctness"],
            summary_path=str(summary_path),
        )
        notify_job(store, job_id, getattr(args, "notify_cmd", None))
    return []


# ------------------------------------------------------------------ serve
@dataclass
class ServeStats:
    processed: int = 0
    halted: bool = False


def _install_signal_handlers(stop: threading.Event) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, _handler)
        except (ValueError, OSError):
            pass


def serve(store: Store, args: Any) -> tuple[int, dict[str, Any]]:
    store.ensure_dirs()
    try:
        # status/wait 的锁探测可能瞬时持有 serve.lock；给 2s 重试，避免误判。
        serve_fd = store.acquire_lock_file(store.serve_lock_path, timeout=2.0)
    except ServeBusy:
        return 3, {
            "error": "serve_already_running",
            "detail": "已有执行器持有 serve.lock；不能启动第二个 GPU 用户",
            "serve": store.serve_health(),
        }
    pid = os.getpid()
    stop = threading.Event()
    _install_signal_handlers(stop)
    stats = ServeStats()
    try:
        store.set_serve(pid, {"argv": sys.argv, "clear_halt": bool(args.clear_halt)})
        state = store.read()
        if state.get("halted"):
            if not args.clear_halt:
                return 5, {
                    "error": "queue_halted",
                    "detail": "队列被停用；确认设备空闲与进程已退出后，用 serve --clear-halt 恢复",
                    "halted": state["halted"],
                }
            previous = store.clear_halt()
            print(
                json.dumps({"cleared_halt": previous}, ensure_ascii=False),
                file=sys.stderr,
            )
        try:
            reconcile(store, args)
            gpu_guard(args)
        except Halt as halt:
            store.set_halt(halt.reason, halt.detail, halt.job_id)
            stats.halted = True
            notify_halt(store, args.notify_cmd)
            return 1, {"error": "halted", "reason": halt.reason, "detail": halt.detail}

        deliver_pending(store, args.notify_cmd)
        last_guard = 0.0
        while not stop.is_set():
            job = None
            gpu_fd: int | None = None
            try:
                if time.monotonic() - last_guard >= GPU_GUARD_INTERVAL_S:
                    gpu_guard(args)
                    last_guard = time.monotonic()
                try:
                    gpu_fd = store.acquire_lock_file(store.gpu_lock_path)
                except ServeBusy:
                    raise Halt(
                        "gpu.lock 被其他进程占用，停队列",
                        {"path": str(store.gpu_lock_path)},
                    ) from None
                job = store.take_next()
                if job is None:
                    store.heartbeat()
                    if args.once or args.drain:
                        break
                    time.sleep(args.poll)
                    continue
                print(
                    json.dumps(
                        {
                            "event": "job_start",
                            "job_id": job["job_id"],
                            "session_id": job["session_id"],
                            "commit": job["commit"],
                            "timeout_s": job["timeout_s"],
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                summary = run_job(store, job, args, stop)
                stats.processed += 1
                print(
                    json.dumps(
                        {
                            "event": "job_finish",
                            "job_id": summary["job_id"],
                            "state": summary["state"],
                            "correctness": summary["correctness"],
                            "elapsed_s": summary["elapsed_s"],
                            "output_dir": summary["output_dir"],
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                deliver_pending(store, args.notify_cmd)
                if args.once:
                    break
            except Halt as halt:
                store.set_halt(halt.reason, halt.detail, halt.job_id)
                stats.halted = True
                notify_halt(store, args.notify_cmd)
                return 1, {
                    "error": "halted",
                    "reason": halt.reason,
                    "detail": halt.detail,
                    "job_id": halt.job_id,
                }
            finally:
                store.release_lock_file(gpu_fd)
        return 0, {
            "event": "serve_stop",
            "processed": stats.processed,
            "halted": stats.halted,
            "stopped": stop.is_set(),
        }
    finally:
        try:
            store.clear_serve(pid)
        finally:
            store.release_lock_file(serve_fd)
