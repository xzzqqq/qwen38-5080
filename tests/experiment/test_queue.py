"""端到端队列行为：FIFO、worktree 隔离、失败短路、超时、取消、通知、wait。"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from conftest import (
    bench_step,
    check_step,
    cli,
    head,
    make_request,
    memory_step,
    payload,
    pid_gone,
    profile_step,
    result,
    serve,
    start_serve,
    status,
    submit,
    wait_job,
    wait_state,
)


def _state_dir(repo: Path) -> Path:
    return repo / "results/experiments"


def _runs(repo: Path) -> list[str]:
    path = _state_dir(repo) / "fake_runs.log"
    return path.read_text().splitlines() if path.exists() else []


def test_fifo_no_overlap_and_isolated_outputs(repo):
    commit = head(repo)
    first = submit(repo, make_request("session-a", commit, [check_step()]))
    second = submit(repo, make_request("session-b", commit, [check_step()]))
    assert submit(repo, make_request("session-c", commit, [bench_step()]))["queue_position"] == 3

    proc = serve(repo, "--drain")
    assert proc.returncode == 0, proc.stderr
    for job in (first, second):
        data = result(repo, job["job_id"])
        assert data["state"] == "succeeded", data
        assert data["correctness"] == "passed"
        assert data["metrics"]["check"]["gates_total"] == 2
        assert data["output_dir"] == str(_state_dir(repo) / job["job_id"])
        assert Path(job["output_dir"], "check/check.json").exists()
        assert data["artifacts"]["output"] == job["output_dir"]

    assert first["output_dir"] != second["output_dir"]
    runs = _runs(repo)
    starts = [line.split()[1] for line in runs if line.startswith("start")]
    ends = [line.split()[1] for line in runs if line.startswith("end")]
    assert starts == [first["job_id"], second["job_id"]]
    assert ends == [first["job_id"], second["job_id"]]
    # 串行：第一个 job 结束之后第二个才开始
    assert runs[0].startswith("start") and runs[1].startswith("end")


def test_job_runs_submitted_commit_not_edited_worktree(repo):
    commit_a = head(repo)
    job = submit(repo, make_request("session-a", commit_a, [check_step()]))
    # 提交后改 Explorer worktree，并再做一个 commit 把 HEAD 移走
    (repo / "probe.py").write_text('MARKER = "commit-b"\n')
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "b"],
        check=True,
        capture_output=True,
    )
    assert head(repo) != commit_a

    proc = serve(repo, "--drain")
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "succeeded", data
    assert data["commit"] == commit_a
    artifact = json.loads((Path(data["artifacts"]["output"]) / "check/check.json").read_text())
    assert artifact["probe"] == "commit-a"
    assert artifact["results"][0]["provenance"]["git"]["commit"] == commit_a


def test_failed_check_stops_bench(repo):
    commit = head(repo)
    job = submit(
        repo,
        make_request(
            "session-a",
            commit,
            [check_step(), bench_step()],
            timeout_s=60,
        ),
    )
    proc = serve(repo, "--drain", env={"FAKE_CHECK_MODE": "fail_gate"})
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "failed"
    assert data["correctness"] == "failed"
    assert data["error"]["step"] == 0
    assert data["error"]["category"] == "gate_failed"
    assert data["steps"][0]["exit_code"] == 1
    assert len(data["steps"]) == 1
    assert not (Path(data["artifacts"]["output"]) / "bench").exists()
    # 失败 job 的 worktree 保留，便于查日志
    assert (repo / ".tmp/worktrees" / job["job_id"]).exists()


def test_missing_artifact_fails_without_correctness_pass(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    proc = serve(repo, "--drain", env={"FAKE_CHECK_MODE": "no_artifact"})
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "failed"
    assert data["correctness"] == "not_checked"
    assert data["error"]["category"] == "invalid_artifact"
    assert "产物" in data["error"]["reason"]


def test_bench_only_is_not_checked(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [bench_step()]))
    proc = serve(repo, "--drain")
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "succeeded"
    assert data["correctness"] == "not_checked"
    assert data["metrics"]["bench"][0]["model_tok_s"]["median"] == 31.0
    # 成功 job 的 worktree 删除，避免堆积
    assert not (repo / ".tmp/worktrees" / job["job_id"]).exists()


def test_timeout_kills_process_group_and_queue_continues(repo):
    commit = head(repo)
    victim = submit(
        repo, make_request("session-a", commit, [check_step()], timeout_s=2)
    )
    follower = submit(repo, make_request("session-b", commit, [check_step()]))
    proc = serve(
        repo,
        "--drain",
        env={
            "FAKE_SLEEP": "30",
            "FAKE_SLEEP_SESSION": "session-a",
            "FAKE_GRANDCHILD": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    data = result(repo, victim["job_id"])
    assert data["state"] == "timed_out"
    assert data["correctness"] == "not_checked"
    assert "超出" in data["error"]["reason"]
    grandchild_pid = int(
        (Path(data["artifacts"]["output"]) / "check" / "grandchild.pid").read_text().strip()
    )
    assert pid_gone(grandchild_pid), f"grandchild {grandchild_pid} 仍在运行"
    assert result(repo, follower["job_id"])["state"] == "succeeded"
    runs = _runs(repo)
    victim_starts = [i for i, line in enumerate(runs) if line.startswith(f"start {victim['job_id']}")]
    follower_starts = [i for i, line in enumerate(runs) if line.startswith(f"start {follower['job_id']}")]
    assert victim_starts and follower_starts
    assert follower_starts[0] > victim_starts[0]


def test_cancel_queued_never_executes(repo):
    commit = head(repo)
    first = submit(repo, make_request("session-a", commit, [check_step()]))
    second = submit(repo, make_request("session-b", commit, [check_step()]))
    server = start_serve(
        repo, "--drain", env={"FAKE_SLEEP": "2", "FAKE_SLEEP_SESSION": "session-a"}
    )
    wait_state(repo, first["job_id"], {"running"})
    proc = cli(repo, "cancel", second["job_id"])
    data = payload(proc)
    assert data["state"] == "cancelled"
    assert data["completed"] is True
    assert data["cancelled"] is True
    assert result(repo, second["job_id"])["state"] == "cancelled"
    server.wait(timeout=120)
    assert server.returncode == 0, server.stderr.read()
    assert result(repo, first["job_id"])["state"] == "succeeded"
    starts = [line for line in _runs(repo) if line.startswith("start")]
    assert all(second["job_id"] not in line for line in starts)


def test_cancel_running_waits_for_exit(repo):
    commit = head(repo)
    job = submit(
        repo, make_request("session-a", commit, [check_step()], timeout_s=120)
    )
    server = start_serve(repo, "--once", env={"FAKE_SLEEP": "30"})
    wait_state(repo, job["job_id"], {"running"})
    record = json.loads((_state_dir(repo) / "queue.json").read_text())["jobs"][job["job_id"]]
    pid = record["pid"]
    data = payload(cli(repo, "cancel", job["job_id"]))
    assert data["cancelled"] is True
    assert data["completed"] is False or data["state"] == "cancelled"
    state = wait_state(repo, job["job_id"], {"cancelled"}, timeout=30)
    assert state["correctness"] == "not_checked"
    assert pid_gone(pid)
    server.wait(timeout=60)
    # 重复 cancel 不产生额外操作
    again = payload(cli(repo, "cancel", job["job_id"]))
    assert again["completed"] is True
    assert again["state"] == "cancelled"


def test_duplicate_queries_do_not_rerun(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    assert serve(repo, "--drain").returncode == 0
    before = len([line for line in _runs(repo) if line.startswith("start")])
    for _ in range(3):
        status(repo)
        result(repo, job["job_id"])
        payload(cli(repo, "cancel", job["job_id"]))
    assert serve(repo, "--drain").returncode == 0
    after = len([line for line in _runs(repo) if line.startswith("start")])
    assert before == after == 1


def test_serve_is_singleton(repo):
    commit = head(repo)
    submit(repo, make_request("s", commit, [check_step()]))
    server = start_serve(repo, env={"FAKE_SLEEP": "2"})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if status(repo)["serve"]["running"]:
            break
        time.sleep(0.05)
    second = serve(repo, "--once")
    if second.returncode != 3:
        raise AssertionError(
            f"second serve rc={second.returncode}; first poll={server.poll()}; "
            f"stdout={second.stdout!r}; stderr={second.stderr!r}; "
            f"first_stderr={server.stderr.read() if server.poll() is not None else '<running>'!r}"
        )
    assert payload(second)["error"] == "serve_already_running"
    server.terminate()
    server.wait(timeout=30)


def test_wait_paths(repo):
    commit = head(repo)
    done = submit(repo, make_request("session-a", commit, [check_step()]))
    assert serve(repo, "--drain").returncode == 0
    quick = wait_job(repo, done["job_id"])
    assert quick.returncode == 0
    assert payload(quick)["finished"] == [done["job_id"]]

    queued = submit(repo, make_request("session-b", commit, [check_step()]))
    no_executor = wait_job(repo, queued["job_id"])
    assert no_executor.returncode == 4
    assert payload(no_executor)["error"] == "executor_not_running"

    server = start_serve(repo, "--drain")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not status(repo)["serve"]["running"]:
        time.sleep(0.05)
    waited = wait_job(repo, queued["job_id"])
    assert waited.returncode == 0
    assert payload(waited)["results"][queued["job_id"]]["state"] == "succeeded"
    server.wait(timeout=60)
    unknown = wait_job(repo, "job-999999-deadbeef")
    assert unknown.returncode == 5


def test_notification_delivered_once_and_failure_safe(repo):
    commit = head(repo)
    log = repo / "notify.log"
    command = f"printf '%s %s\\n' \"$JOB_ID\" \"$JOB_STATE\" >> {log}"
    first = submit(repo, make_request("session-a", commit, [check_step()]))
    second = submit(repo, make_request("session-b", commit, [bench_step()]))
    proc = serve(repo, "--drain", "--notify-cmd", command)
    assert proc.returncode == 0, proc.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert {line.split()[0] for line in lines} == {first["job_id"], second["job_id"]}
    assert all(line.split()[1] in ("succeeded",) for line in lines)
    for job in (first, second):
        note = result(repo, job["job_id"])["notification"]
        assert note["state"] == "delivered"
        assert note["attempts"] == 1
    # 重复查询和再次 serve 不会重复通知
    status(repo)
    result(repo, first["job_id"])
    serve(repo, "--drain", "--notify-cmd", command)
    assert len(log.read_text().splitlines()) == 2

    failed_note_job = submit(repo, make_request("session-c", commit, [bench_step()]))
    serve(repo, "--drain", "--notify-cmd", "exit 9")
    data = result(repo, failed_note_job["job_id"])
    assert data["state"] == "succeeded"
    assert data["notification"]["state"] == "failed"
    assert data["notification"]["returncode"] == 9


def test_notification_pending_without_command(repo):
    commit = head(repo)
    job = submit(repo, make_request("s", commit, [check_step()]))
    assert serve(repo, "--drain").returncode == 0
    note = result(repo, job["job_id"])["notification"]
    assert note["state"] == "pending"
    assert "wait/result" in note["reason"]


def test_stale_bench_config_fails_end_to_end(repo):
    commit = head(repo)
    job = submit(
        repo,
        make_request(
            "s",
            commit,
            [bench_step(steps=4, repeats=1)],
            timeout_s=60,
        ),
    )
    proc = serve(repo, "--drain", env={"FAKE_BENCH_STEPS": "32"})
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "failed"
    assert data["correctness"] == "not_checked"
    assert data["error"]["category"] == "invalid_artifact"
    assert "steps" in data["error"]["reason"]
    assert data["steps"][0]["exit_code"] == 0  # 旧结果本身退出码是 0，靠产物校验拦下


def test_profile_and_memory_jobs_end_to_end(repo):
    commit = head(repo)
    profile_job = submit(
        repo,
        make_request("session-p", commit, [profile_step(steps=2)], timeout_s=60),
    )
    memory_job = submit(
        repo,
        make_request("session-m", commit, [memory_step(steps=4)], timeout_s=60),
    )
    proc = serve(repo, "--drain")
    assert proc.returncode == 0, proc.stderr
    profile = result(repo, profile_job["job_id"])
    assert profile["state"] == "succeeded"
    assert profile["correctness"] == "not_checked"
    assert profile["metrics"]["profile"][0]["per_step_kernels"] > 0
    assert (Path(profile["artifacts"]["output"]) / "profile/timeline_summary.json").exists()
    memory = result(repo, memory_job["job_id"])
    assert memory["state"] == "succeeded"
    assert memory["correctness"] == "not_checked"
    assert memory["metrics"]["memory"][0]["stages"] == 6


def test_stray_background_process_is_killed_between_steps(repo):
    commit = head(repo)
    job = submit(
        repo,
        make_request("s", commit, [check_step(), bench_step()], timeout_s=60),
    )
    proc = serve(repo, "--drain", env={"FAKE_STRAY": "1"})
    assert proc.returncode == 0, proc.stderr
    data = result(repo, job["job_id"])
    assert data["state"] == "succeeded", data
    strays = data["steps"][0]["stray_pids_killed"]
    assert strays, "残留后台进程没有被记录"
    assert pid_gone(strays[0])
    # 第二步仍然执行（残留进程在释放 GPU 锁之前被清掉）
    assert len(data["steps"]) == 2
    assert data["steps"][1]["kind"] == "bench"


def test_job_env_is_provided_to_steps(repo):
    commit = head(repo)
    job = submit(repo, make_request("env-session", commit, [check_step()]))
    assert serve(repo, "--drain").returncode == 0
    data = result(repo, job["job_id"])
    artifact = json.loads(
        (Path(data["artifacts"]["output"]) / "check/check.json").read_text()
    )
    assert artifact["job_id"] == job["job_id"]
