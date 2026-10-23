"""验收 P1/P2 回归：commit 隔离、check 覆盖、halt 可靠性、worktree 失败。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import types
from pathlib import Path

import pytest

from conftest import (
    bench_step,
    check_step,
    cli,
    head,
    make_request,
    payload,
    result,
    serve,
    submit,
)
from runner.experiment import executor as exp
from runner.experiment import schema
from runner.experiment.store import Store


def _submit_error(repo: Path, request: dict, code: int = 2) -> dict:
    path = repo / ".tmp" / f"bad-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    proc = cli(repo, "submit", "--request", str(path))
    data = payload(proc)
    assert proc.returncode == code, (proc.returncode, data, proc.stderr)
    return data


def _queue(repo: Path) -> dict:
    return json.loads((repo / "results/experiments/queue.json").read_text())


def _write_queue(repo: Path, state: dict) -> None:
    (repo / "results/experiments/queue.json").write_text(json.dumps(state))


def _kill_group(pgid: int | None) -> None:
    if not pgid:
        return
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except (ProcessLookupError, TypeError):
        pass


# ------------------------------------------------------------------ P1.1
def test_absolute_repo_paths_rejected_at_submit(repo):
    commit = head(repo)
    steps = []
    step = check_step()
    step["argv"][1] = str(repo / "scripts/m3_check.py")
    steps.append(step)
    step = check_step()
    step["argv"][0] = str(repo / ".venv/bin/python")
    steps.append(step)
    step = check_step()
    step["argv"] += ["--fixtures", str(repo / "bench/workloads/m3-p0")]
    steps.append(step)
    for step in steps:
        data = _submit_error(repo, make_request("s", commit, [step]))
        assert "绝对路径" in data["error"], data


def test_absolute_path_to_other_worktree_rejected(repo, tmp_path):
    commit = head(repo)
    other = tmp_path / "other-worktree"
    add = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(other), commit],
        capture_output=True,
        text=True,
    )
    assert add.returncode == 0, add.stderr
    try:
        step = check_step()
        step["argv"][1] = str(other / "scripts/m3_check.py")
        data = _submit_error(repo, make_request("s", commit, [step]))
        assert "绝对路径" in data["error"], data
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(other)],
            capture_output=True,
            text=True,
        )


def test_tampered_request_absolute_path_fails_without_running(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    request_path = Path(job["job_dir"]) / "request.json"
    request_path.chmod(0o644)
    data = json.loads(request_path.read_text())
    data["steps"][0]["argv"][1] = str(repo / "scripts/m3_check.py")
    request_path.write_text(json.dumps(data))

    proc = serve(repo, "--drain")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    out = result(repo, job["job_id"])
    assert out["state"] == "failed"
    assert "请求重放校验失败" in out["error"]["reason"]
    assert not (Path(out["artifacts"]["output"]) / "check").exists()
    assert not (repo / "results/experiments/fake_runs.log").exists()
    assert _queue(repo)["jobs"][job["job_id"]]["state"] == "failed"


def test_verify_argv_isolation(repo):
    out = repo / "results/experiments/iso/output"
    out.mkdir(parents=True)
    wt = repo / ".tmp/wt"
    wt.mkdir(parents=True)
    exp.verify_argv_isolation(
        [
            ".venv/bin/python",
            "scripts/m3_check.py",
            "--only",
            "graph-1024",
            "--out",
            str(out / "check"),
        ],
        repo,
        wt,
        out,
    )
    with pytest.raises(exp.JobAbort, match="绝对路径"):
        exp.verify_argv_isolation(
            [str(repo / "scripts/m3_check.py")], repo, wt, out
        )
    with pytest.raises(exp.JobAbort, match="逃逸"):
        exp.verify_argv_isolation(
            ["../../scripts/m3_check.py"], repo, wt, out
        )


# ------------------------------------------------------------------ P1.2
def test_correctness_covers_all_requested_checks(repo):
    commit = head(repo)
    request = schema.parse_request_object(
        make_request("s", commit, [check_step("c1"), bench_step(), check_step("c2")]),
        repo,
    )
    ok0 = {"index": 0, "kind": "check", "validation": {"ok": True, "correctness": "passed"}}
    bench = {"index": 1, "kind": "bench", "validation": {"ok": True, "correctness": "not_checked"}}
    ok2 = {"index": 2, "kind": "check", "validation": {"ok": True, "correctness": "passed"}}
    failed0 = {
        "index": 0,
        "kind": "check",
        "validation": {"ok": False, "correctness": "failed"},
    }
    assert exp.compute_correctness([ok0, bench], request) == "not_checked"
    assert exp.compute_correctness([ok0, bench, ok2], request) == "passed"
    assert exp.compute_correctness([failed0], request) == "failed"


def test_unexecuted_check_is_not_passed(repo):
    commit = head(repo)
    job = submit(
        repo,
        make_request(
            "s",
            commit,
            [check_step("c1"), bench_step(), check_step("c2")],
            timeout_s=60,
        ),
    )
    proc = serve(repo, "--drain", env={"FAKE_BENCH_COMPLETE": "0"})
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    data = result(repo, job["job_id"])
    assert data["state"] == "failed"
    assert data["correctness"] == "not_checked"
    assert len(data["steps"]) == 2
    assert data["steps"][0]["kind"] == "check"
    assert data["steps"][0]["validation"]["ok"] is True


def test_all_requested_checks_passed_and_check_before_failed_bench(repo):
    commit = head(repo)
    both = submit(repo, make_request("s1", commit, [check_step("c1"), check_step("c2")]))
    partial = submit(repo, make_request("s2", commit, [check_step(), bench_step()]))
    proc = serve(repo, "--drain", env={"FAKE_BENCH_COMPLETE": "0"})
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    both_data = result(repo, both["job_id"])
    assert both_data["state"] == "succeeded"
    assert both_data["correctness"] == "passed"
    partial_data = result(repo, partial["job_id"])
    assert partial_data["state"] == "failed"
    assert partial_data["correctness"] == "passed"  # 唯一请求的 check 已通过
    assert len(partial_data["steps"]) == 2


# ------------------------------------------------------------------ P1.3
def _run_job_in_process(repo, monkeypatch, *, mode: str):
    monkeypatch.setenv("FAKE_CHECK_MODE", mode)
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    store = Store(repo)
    record = store.take_next()
    assert record is not None
    monkeypatch.setattr(exp, "cleanup_job_group", lambda keeper, pgid: False)
    pgid = record.get("pgid")
    try:
        with pytest.raises(exp.Halt, match="进程组"):
            exp.run_job(
                store,
                record,
                types.SimpleNamespace(notify_cmd=None),
                threading.Event(),
            )
    finally:
        # 被 patch 掉的清理没有真的杀 keeper，测试自己收尾
        final = store.get(job["job_id"]) or {}
        _kill_group(final.get("pgid") or pgid)
    return job, store


def test_cleanup_failure_halts_failed_job(repo, monkeypatch):
    job, store = _run_job_in_process(repo, monkeypatch, mode="fail_gate")
    record = store.get(job["job_id"])
    assert record["state"] == "failed"
    execution = json.loads(store.execution_path(job["job_id"]).read_text())
    assert execution["group_cleanup_error"]
    assert execution["job_group_cleaned"] is False


def test_cleanup_failure_halts_succeeded_job(repo, monkeypatch):
    job, store = _run_job_in_process(repo, monkeypatch, mode="ok")
    record = store.get(job["job_id"])
    assert record["state"] == "interrupted"
    execution = json.loads(store.execution_path(job["job_id"]).read_text())
    assert execution["group_cleanup_error"]


def test_reconcile_checks_whole_process_group(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    orphan = subprocess.Popen(["sleep", "120"], start_new_session=True)
    pgid = os.getpgid(orphan.pid)
    try:
        state = _queue(repo)
        record = state["jobs"][job["job_id"]]
        record.update(
            state="running",
            started_at=record["created_at"],
            pid=999999,
            pgid=pgid,
            proc_start_ticks=1,
        )
        _write_queue(repo, state)
        halted = serve(repo, "--drain")
        assert halted.returncode == 1, (halted.returncode, halted.stdout, halted.stderr)
        data = payload(halted)
        assert data["error"] == "halted"
        assert "进程组" in data["reason"]
        assert _queue(repo)["halted"] is not None
    finally:
        orphan.kill()
        orphan.wait(timeout=10)
    resumed = serve(repo, "--clear-halt", "--drain")
    assert resumed.returncode == 0, (resumed.returncode, resumed.stdout, resumed.stderr)
    assert result(repo, job["job_id"])["state"] == "interrupted"


def test_gpu_guard_fails_closed_when_nvidia_smi_unavailable(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    fake_bin = repo / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nexit 1\n")
    smi.chmod(0o755)
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    proc = cli(repo, "serve", "--drain", env=env)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    data = payload(proc)
    assert data["error"] == "halted"
    assert "nvidia-smi" in data["reason"]
    assert _queue(repo)["jobs"][job["job_id"]]["state"] == "queued"


# ------------------------------------------------------------------ P2.5
def test_worktree_prepare_failure_marks_job_failed(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    worktree = repo / ".tmp/worktrees" / job["job_id"]
    worktree.mkdir(parents=True)
    (worktree / "keep.txt").write_text("stale")

    proc = serve(repo, "--drain")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert b"Traceback" not in proc.stderr.encode()
    data = result(repo, job["job_id"])
    assert data["state"] == "failed"
    assert "worktree 已存在" in data["error"]["reason"]
    record = _queue(repo)["jobs"][job["job_id"]]
    assert record["state"] == "failed"
    execution = json.loads(
        (repo / "results/experiments" / job["job_id"] / "execution.json").read_text()
    )
    assert execution["worktree"] is None
    assert execution["job_group_cleaned"] is True
