"""提交期校验：字段、commit、脚本注册、--out/result、session 互斥。"""

from __future__ import annotations

import json
import time

from conftest import (
    bench_step,
    check_step,
    cli,
    head,
    make_request,
    payload,
    submit,
)


def _submit_error(repo, request: dict, code: int = 2) -> dict:
    path = repo / ".tmp" / f"bad-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    proc = cli(repo, "submit", "--request", str(path))
    data = payload(proc)
    assert proc.returncode == code, (proc.returncode, data, proc.stderr)
    return data


def test_valid_request_enqueues(repo):
    commit = head(repo)
    data = submit(repo, make_request("session-a", commit, [check_step()]))
    assert data["state"] == "queued"
    assert data["job_id"].startswith("job-")
    assert data["commit"] == commit
    assert data["queue_position"] == 1
    assert data["steps"][0]["kind"] == "check"
    assert data["steps"][0]["argv"][0].endswith(".venv/bin/python")
    assert data["steps"][0]["result"].startswith(data["output_dir"] + "/")
    assert data["request_sha256"]


def test_unknown_fields_rejected(repo):
    commit = head(repo)
    request = make_request("s", commit, [check_step()])
    request["extra"] = 1
    _submit_error(repo, request)

    request = make_request("s", commit, [check_step()])
    request["steps"][0]["env"] = {"A": "1"}
    _submit_error(repo, request)

    request = make_request("s", commit, [check_step()])
    del request["steps"]
    _submit_error(repo, request)


def test_timeout_must_be_positive_number(repo):
    commit = head(repo)
    for bad in (0, -1, "300", True, 10**9):
        request = make_request("s", commit, [check_step()], timeout_s=bad)
        _submit_error(repo, request)


def test_commit_must_be_full_existing_sha(repo):
    commit = head(repo)
    for bad in (commit[:12], "0" * 40, "main", ""):
        request = make_request("s", bad, [check_step()])
        _submit_error(repo, request)


def test_script_must_be_registered_and_kind_match(repo):
    commit = head(repo)
    request = make_request(
        "s",
        commit,
        [
            {
                "kind": "check",
                "argv": [".venv/bin/python", "tools/fake_profile.py", "--out", "{output}/x"],
                "result": "{output}/x/check.json",
            }
        ],
    )
    assert "已注册" in _submit_error(repo, request)["error"]

    request = make_request("s", commit, [check_step()])
    request["steps"][0]["kind"] = "bench"  # m3_check 只允许 check
    _submit_error(repo, request)

    request = make_request("s", commit, [bench_step()])
    request["steps"][0]["kind"] = "memory"
    _submit_error(repo, request)


def test_out_and_result_must_stay_in_output(repo):
    commit = head(repo)
    step = check_step()
    step["argv"] = [token for token in step["argv"] if token != "--out"]
    # 去掉 --out 后，紧跟其后的路径也成了普通参数；这里再明确删一次
    step["argv"] = [token for token in step["argv"] if token != "{output}/check"]
    _submit_error(repo, make_request("s", commit, [step]))

    step = check_step()
    step["result"] = "{output}/other/check.json"  # --out 是 {output}/check
    _submit_error(repo, make_request("s", commit, [step]))

    step = check_step()
    step["result"] = "/tmp/check.json"
    _submit_error(repo, make_request("s", commit, [step]))

    step = check_step()
    step["argv"][-1] = "{output}/../escape"
    _submit_error(repo, make_request("s", commit, [step]))


def test_resume_worker_and_list_rejected(repo):
    commit = head(repo)
    for flag in ("--resume", "--worker", "--list"):
        step = bench_step()
        step["argv"].append(flag)
        _submit_error(repo, make_request("s", commit, [step]))


def test_unknown_only_scenario_rejected(repo):
    commit = head(repo)
    step = check_step(only="graph-1024,does-not-exist")
    assert "未知场景" in _submit_error(repo, make_request("s", commit, [step]))["error"]


def test_numeric_flags_validated(repo):
    commit = head(repo)
    step = check_step()
    step["argv"][step["argv"].index("--steps") + 1] = "0"
    _submit_error(repo, make_request("s", commit, [step]))

    step = bench_step(repeats=0)
    _submit_error(repo, make_request("s", commit, [step]))


def test_duplicate_result_rejected(repo):
    commit = head(repo)
    _submit_error(
        repo,
        make_request("s", commit, [check_step("check"), check_step("check")]),
    )


def test_one_unfinished_job_per_session(repo):
    commit = head(repo)
    first = submit(repo, make_request("session-a", commit, [check_step()]))
    second = _submit_error(
        repo, make_request("session-a", commit, [bench_step()]), code=3
    )
    assert second["error"] == "session_busy"
    assert second["existing_job_id"] == first["job_id"]
    assert second["job_id"] == first["job_id"]

    # 另一个 session 可以提交
    other = submit(repo, make_request("session-b", commit, [check_step()]))
    assert other["job_id"] != first["job_id"]

    # 取消 queued 后同一 session 可以再提交
    proc = cli(repo, "cancel", first["job_id"])
    assert payload(proc)["state"] == "cancelled"
    third = submit(repo, make_request("session-a", commit, [bench_step()]))
    assert third["job_id"] not in (first["job_id"], other["job_id"])


def test_session_id_validated(repo):
    commit = head(repo)
    for bad in ("", "   ", "a" * 201, "with\nnewline"):
        _submit_error(repo, make_request(bad, commit, [check_step()]))
