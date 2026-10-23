"""产物校验器：check / bench / memory / profile 的正例与旧配置/缺产物反例。"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace

import pytest

from conftest import PYTHON, head
from runner.experiment import artifacts, schema


def _run(repo, argv, out, env=None, expect_rc=None):
    env2 = {
        **os.environ,
        "EXPERIMENT_STATE_DIR": str(repo / "results/experiments"),
        "EXPERIMENT_JOB_ID": "unit-test",
        **(env or {}),
    }
    proc = subprocess.run(
        [*argv, "--out", str(out)],
        cwd=repo,
        env=env2,
        capture_output=True,
        text=True,
    )
    if expect_rc is not None:
        assert proc.returncode == expect_rc, (proc.returncode, proc.stderr)
    return proc


def _expectation(repo, kind, out, **overrides):
    base = dict(
        kind=kind,
        argv=("unit",),
        commit=head(repo),
        output_dir=out,
        result_path=out / {"check": "check.json", "bench": "bench.json",
                           "memory": "memory.json", "profile": "timeline_summary.json"}[kind],
        worktree=repo,
        scenarios=None,
        steps=4,
        repeats=None,
        repeats_non_graph=None,
        chunk_size=128,
        graph_warmup=3,
        fixtures=None,
        smoke_max_new_tokens=None,
    )
    base.update(overrides)
    return schema.Expectation(**base)


def _make_check(repo, out, *, only="graph-1024", steps=16, env=None):
    return _run(
        repo,
        [PYTHON, "scripts/m3_check.py", "--only", only, "--steps", str(steps)],
        out,
        env=env,
        expect_rc=None,
    )


def _make_bench(
    repo,
    out,
    *,
    only="graph-512-1024",
    steps=4,
    repeats=1,
    repeats_non_graph=None,
    env=None,
):
    argv = [
        PYTHON,
        "bench/e2e/decode.py",
        "--only",
        only,
        "--steps",
        str(steps),
        "--repeats",
        str(repeats),
    ]
    if repeats_non_graph is not None:
        argv += ["--repeats-non-graph", str(repeats_non_graph)]
    return _run(repo, argv, out, env=env)


def _make_memory(repo, out, *, steps=4):
    return _run(
        repo,
        [PYTHON, "scripts/m3_memory.py", "--only", "graph-512-1024", "--steps", str(steps)],
        out,
    )


def _make_profile(repo, out, *, steps=2):
    return _run(repo, ["bash", "scripts/m3_profile.sh", "--steps", str(steps)], out)


# ------------------------------------------------------------------ check
def test_check_ok(repo, tmp_path):
    out = tmp_path / "check"
    proc = _make_check(repo, out)
    assert proc.returncode == 0, proc.stderr
    result = artifacts.validate(
        "check",
        _expectation(
            repo,
            "check",
            out,
            scenarios=("graph-1024",),
            steps=16,
            fixtures="bench/workloads/m3-p0",
        ),
    )
    assert result["correctness"] == "passed"
    assert result["metrics"]["gates_total"] == 2
    assert result["metrics"]["diagnostics_failed"] == []


def test_check_gate_failure_is_correctness_failure(repo, tmp_path):
    out = tmp_path / "check"
    proc = _make_check(repo, out, env={"FAKE_CHECK_MODE": "fail_gate"})
    assert proc.returncode == 1
    with pytest.raises(artifacts.ArtifactError) as exc:
        artifacts.validate(
            "check",
            _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16),
        )
    assert exc.value.category == "gate_failed"
    assert "gate" in exc.value.reason


def test_check_diagnostic_failure_does_not_fail_gate(repo, tmp_path):
    out = tmp_path / "check"
    proc = _make_check(repo, out, env={"FAKE_CHECK_MODE": "diag_fail"})
    assert proc.returncode == 0
    result = artifacts.validate(
        "check",
        _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16),
    )
    assert result["correctness"] == "passed"
    assert result["metrics"]["diagnostics_failed"]


def test_check_coverage_gate_required(repo, tmp_path):
    out = tmp_path / "check"
    _make_check(repo, out)
    report_path = out / "graph-1024" / "report.json"
    report = json.loads(report_path.read_text())
    report["items"] = [
        item for item in report["items"] if not item["name"].startswith("coverage:")
    ]
    report_path.write_text(json.dumps(report))
    with pytest.raises(artifacts.ArtifactError, match="coverage"):
        artifacts.validate(
            "check",
            _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16),
        )


def test_check_gates_cannot_be_only_diagnostics(repo, tmp_path):
    out = tmp_path / "check"
    _make_check(repo, out)
    report_path = out / "graph-1024" / "report.json"
    report = json.loads(report_path.read_text())
    report["items"] = [item for item in report["items"] if item["level"] != "gate"]
    report_path.write_text(json.dumps(report))
    with pytest.raises(artifacts.ArtifactError, match="没有 gate"):
        artifacts.validate(
            "check",
            _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16),
        )


def test_check_empty_and_missing_artifact(repo, tmp_path):
    out = tmp_path / "check"
    _make_check(repo, out)
    (out / "check.json").write_text("{}")
    with pytest.raises(artifacts.ArtifactError, match="为空"):
        artifacts.validate(
            "check",
            _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16),
        )
    out2 = tmp_path / "check2"
    _make_check(repo, out2)
    (out2 / "graph-1024" / "report.json").unlink()
    with pytest.raises(artifacts.ArtifactError, match="缺产物"):
        artifacts.validate(
            "check",
            _expectation(repo, "check", out2, scenarios=("graph-1024",), steps=16),
        )


def test_check_request_mismatch(repo, tmp_path):
    out = tmp_path / "check"
    _make_check(repo, out)
    base = _expectation(repo, "check", out, scenarios=("graph-1024",), steps=16)
    with pytest.raises(artifacts.ArtifactError, match="steps"):
        artifacts.validate("check", replace(base, steps=511))
    with pytest.raises(artifacts.ArtifactError, match="场景"):
        artifacts.validate("check", replace(base, scenarios=("numeric-512",)))
    with pytest.raises(artifacts.ArtifactError, match="commit"):
        artifacts.validate("check", replace(base, commit="0" * 40))


# ------------------------------------------------------------------ bench
def test_bench_ok_is_not_checked(repo, tmp_path):
    out = tmp_path / "bench"
    proc = _make_bench(repo, out)
    assert proc.returncode == 0, proc.stderr
    result = artifacts.validate(
        "bench",
        _expectation(
            repo,
            "bench",
            out,
            scenarios=("graph-512-1024",),
            steps=4,
            repeats=1,
        ),
    )
    assert result["correctness"] == "not_checked"
    assert result["metrics"]["scenarios"][0]["model_tok_s"]["median"] == 31.0


def test_bench_repeats_non_graph(repo, tmp_path):
    out = tmp_path / "bench"
    proc = _make_bench(
        repo,
        out,
        only="graph-512-1024,cap-512-1024",
        repeats=1,
        repeats_non_graph=2,
    )
    assert proc.returncode == 0, proc.stderr
    expect = _expectation(
        repo,
        "bench",
        out,
        scenarios=("graph-512-1024", "cap-512-1024"),
        steps=4,
        repeats=1,
        repeats_non_graph=2,
    )
    result = artifacts.validate("bench", expect)
    repeats = {
        item["scenario"]: item["repeats"] for item in result["metrics"]["scenarios"]
    }
    assert repeats == {"graph-512-1024": 1, "cap-512-1024": 2}
    with pytest.raises(artifacts.ArtifactError, match="repeats"):
        artifacts.validate("bench", replace(expect, repeats_non_graph=None, repeats=2))


def test_bench_incomplete_or_old_config_rejected(repo, tmp_path):
    out = tmp_path / "bench"
    _make_bench(repo, out, env={"FAKE_BENCH_COMPLETE": "0"})
    with pytest.raises(artifacts.ArtifactError, match="complete"):
        artifacts.validate(
            "bench",
            _expectation(repo, "bench", out, scenarios=("graph-512-1024",), steps=4, repeats=1),
        )

    out2 = tmp_path / "bench2"
    _make_bench(repo, out2, env={"FAKE_BENCH_STEPS": "32"})
    with pytest.raises(artifacts.ArtifactError, match="steps"):
        artifacts.validate(
            "bench",
            _expectation(repo, "bench", out2, scenarios=("graph-512-1024",), steps=4, repeats=1),
        )

    out3 = tmp_path / "bench3"
    _make_bench(repo, out3, env={"FAKE_BENCH_REPEATS": "2"})
    with pytest.raises(artifacts.ArtifactError, match="repeats"):
        artifacts.validate(
            "bench",
            _expectation(repo, "bench", out3, scenarios=("graph-512-1024",), steps=4, repeats=1),
        )


def test_bench_sample_shape_checked(repo, tmp_path):
    out = tmp_path / "bench"
    _make_bench(repo, out)
    bench_path = out / "bench.json"
    data = json.loads(bench_path.read_text())
    data["results"][0]["samples"][0]["model"]["per_step_ms"] = [1.0, 2.0]
    bench_path.write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactError, match="per_step_ms"):
        artifacts.validate(
            "bench",
            _expectation(repo, "bench", out, scenarios=("graph-512-1024",), steps=4, repeats=1),
        )


def test_bench_provenance_matrix_checked(repo, tmp_path):
    out = tmp_path / "bench"
    _make_bench(repo, out)
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["run_config"]["steps"] = 32
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(artifacts.ArtifactError, match="run_config.steps"):
        artifacts.validate(
            "bench",
            _expectation(repo, "bench", out, scenarios=("graph-512-1024",), steps=4, repeats=1),
        )


# ------------------------------------------------------------------ memory
def test_memory_ok_and_missing_stage(repo, tmp_path):
    out = tmp_path / "memory"
    proc = _make_memory(repo, out)
    assert proc.returncode == 0, proc.stderr
    result = artifacts.validate(
        "memory",
        _expectation(repo, "memory", out, scenarios=("graph-512-1024",), steps=4),
    )
    assert result["correctness"] == "not_checked"
    assert result["metrics"]["scenarios"][0]["stages"] == 6

    data = json.loads((out / "memory.json").read_text())
    data["results"][0]["stages"] = [
        row for row in data["results"][0]["stages"] if row["stage"] != "decode"
    ]
    (out / "memory.json").write_text(json.dumps(data))
    with pytest.raises(artifacts.ArtifactError, match="stage"):
        artifacts.validate(
            "memory",
            _expectation(repo, "memory", out, scenarios=("graph-512-1024",), steps=4),
        )


# ------------------------------------------------------------------ profile
def test_profile_ok_and_trace_required(repo, tmp_path):
    out = tmp_path / "profile"
    proc = _make_profile(repo, out)
    assert proc.returncode == 0, proc.stderr
    result = artifacts.validate(
        "profile", _expectation(repo, "profile", out, steps=2)
    )
    assert result["correctness"] == "not_checked"
    assert len(result["metrics"]["scenarios"]) == 2

    (out / "graph-512" / "gpu_trace.csv").unlink()
    with pytest.raises(artifacts.ArtifactError, match="gpu_trace"):
        artifacts.validate("profile", _expectation(repo, "profile", out, steps=2))


def test_profile_steps_mismatch(repo, tmp_path):
    out = tmp_path / "profile"
    _make_profile(repo, out, steps=2)
    with pytest.raises(artifacts.ArtifactError, match="steps"):
        artifacts.validate("profile", _expectation(repo, "profile", out, steps=8))
