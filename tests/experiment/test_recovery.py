"""执行器重启、halt、孤儿进程与外部 GPU 进程守卫。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from conftest import (
    check_step,
    cli,
    head,
    make_request,
    payload,
    pid_gone,
    result,
    serve,
    start_serve,
    status,
    submit,
    wait_job,
    wait_state,
)
from runner.experiment import executor


def _queue_path(repo: Path) -> Path:
    return repo / "results/experiments/queue.json"


def _read_queue(repo: Path) -> dict:
    return json.loads(_queue_path(repo).read_text())


def _write_queue(repo: Path, state: dict) -> None:
    _queue_path(repo).write_text(json.dumps(state))


def _mark_running(repo: Path, job_id: str, pid: int, ticks: int | None) -> None:
    state = _read_queue(repo)
    record = state["jobs"][job_id]
    record.update(
        state="running",
        started_at=record.get("created_at"),
        pid=pid,
        pgid=pid,
        proc_start_ticks=ticks,
    )
    _write_queue(repo, state)


def test_restart_marks_dead_running_job_interrupted_and_continues(repo):
    commit = head(repo)
    stale = submit(repo, make_request("session-a", commit, [check_step()]))
    follower = submit(repo, make_request("session-b", commit, [check_step()]))
    _mark_running(repo, stale["job_id"], pid=999999, ticks=1)

    proc = serve(repo, "--drain")
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    data = result(repo, stale["job_id"])
    assert data["state"] == "interrupted"
    assert data["correctness"] == "not_checked"
    assert "重启" in data["error"]["reason"]
    assert result(repo, follower["job_id"])["state"] == "succeeded"


def test_recovered_summary_is_adopted(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    assert serve(repo, "--drain").returncode == 0
    # 模拟执行器写完 summary 但没来得及更新队列记录
    state = _read_queue(repo)
    state["jobs"][job["job_id"]].update(
        state="running", pid=999999, proc_start_ticks=1, finished_at=None
    )
    _write_queue(repo, state)
    assert serve(repo, "--drain").returncode == 0
    record = _read_queue(repo)["jobs"][job["job_id"]]
    assert record["state"] == "succeeded"
    assert record.get("recovered") is True
    assert result(repo, job["job_id"])["correctness"] == "passed"


def test_live_orphan_halts_cancel_kills_and_clear_halt_resumes(repo):
    commit = head(repo)
    orphan_job = submit(repo, make_request("session-a", commit, [check_step()]))
    follower = submit(repo, make_request("session-b", commit, [check_step()]))
    orphan = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        _mark_running(
            repo,
            orphan_job["job_id"],
            pid=orphan.pid,
            ticks=executor.proc_start_ticks(orphan.pid),
        )
        halted = serve(repo, "--drain")
        assert halted.returncode == 1, (halted.returncode, halted.stdout, halted.stderr)
        assert payload(halted)["error"] == "halted"
        state = _read_queue(repo)
        assert state["halted"]["job_id"] == orphan_job["job_id"]
        assert state["jobs"][follower["job_id"]]["state"] == "queued"
        assert status(repo)["halted"] is not None

        # 队列停用时 wait 直接报 queue_halted
        waited = wait_job(repo, follower["job_id"])
        assert waited.returncode == 4
        assert payload(waited)["error"] == "queue_halted"

        # 普通 serve 不会偷偷恢复
        blocked = serve(repo, "--drain")
        assert blocked.returncode == 5
        assert payload(blocked)["error"] == "queue_halted"

        # 取消孤儿进程（没有执行器，直接杀进程组）
        cancelled = payload(cli(repo, "cancel", orphan_job["job_id"]))
        assert cancelled["state"] == "cancelled"
        assert cancelled["completed"] is True
        orphan.wait(timeout=30)
        assert pid_gone(orphan.pid)

        resumed = serve(repo, "--clear-halt", "--drain")
        assert resumed.returncode == 0, (resumed.returncode, resumed.stdout, resumed.stderr)
        assert status(repo)["halted"] is None
        assert result(repo, follower["job_id"])["state"] == "succeeded"
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait(timeout=10)


def test_sigterm_marks_running_job_interrupted(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()], timeout_s=120))
    server = start_serve(repo, "--once", env={"FAKE_SLEEP": "30"})
    wait_state(repo, job["job_id"], {"running"})
    record = _read_queue(repo)["jobs"][job["job_id"]]
    pid = record["pid"]
    server.terminate()
    server.wait(timeout=60)
    data = wait_state(repo, job["job_id"], {"interrupted"}, timeout=30)
    assert "SIGTERM" in data["error"]["reason"]
    assert pid_gone(pid)


def test_gpu_guard_halts_on_foreign_compute_process(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    fake_bin = repo / "fake-bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *query-compute-apps*) echo '12345, python, 100' ;;\n"
        "  *query-gpu*) echo 'Fake GPU, 16303, 0, 999.0, 2000, 30, 40' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    smi.chmod(0o755)
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    proc = cli(repo, "serve", "--drain", env=env)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    data = payload(proc)
    assert data["error"] == "halted"
    assert "GPU" in data["reason"]
    state = _read_queue(repo)
    assert state["halted"] is not None
    assert state["jobs"][job["job_id"]]["state"] == "queued"


def test_cancel_queued_writes_cancelled_summary(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    data = payload(cli(repo, "cancel", job["job_id"]))
    assert data["state"] == "cancelled"
    summary = json.loads(
        (repo / "results/experiments" / job["job_id"] / "summary.json").read_text()
    )
    assert summary["state"] == "cancelled"
    assert summary["correctness"] == "not_checked"
    assert summary["queue_s"] is not None
    assert not list((repo / "results/experiments" / job["job_id"] / "output").glob("check"))


def test_missing_state_file_is_an_error_not_a_reset(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    _queue_path(repo).write_text("{ broken json")
    proc = cli(repo, "status")
    assert proc.returncode == 6
    assert "队列" in payload(proc)["error"]
    # 坏状态不会被启动 serve 时静默重置
    assert _queue_path(repo).read_text() == "{ broken json"
