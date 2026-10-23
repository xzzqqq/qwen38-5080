"""op kind 的产物契约：OpReport 写入器 + validate_op 的正反例。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import conftest
from runner.experiment import artifacts, schema
from runner.experiment.op_artifact import OpReport

COMMIT = "a" * 40


def _expectation(out: Path, **overrides) -> schema.Expectation:
    base = dict(
        kind="op",
        argv=("test",),
        commit=COMMIT,
        output_dir=out,
        result_path=out / "op.json",
        worktree=None,
        scenarios=None,
        steps=1,
        repeats=None,
        repeats_non_graph=None,
        chunk_size=128,
        graph_warmup=3,
        fixtures=None,
        smoke_max_new_tokens=None,
    )
    base.update(overrides)
    return schema.Expectation(**base)


def _write(path: Path, *, gates_ok: bool = True, coverage: bool = True) -> dict:
    """在非 git 目录写一份 op.json，并把 provenance 的 commit 固定成测试值。"""
    report = OpReport(
        path.parent,
        "cand-x",
        "bf16-kv",
        candidate_extra={"backend": "triton"},
        root=path.parent,
        steps=4,
    )
    case = report.case("decode-513", length=513, heads=24)
    if coverage:
        case.gate("coverage: L=1/513/8191 + mask", True)
    case.gate("reference nrmse", gates_ok, {"nrmse": 1e-3})
    case.diag("vs baseline", False, {"note": "different reduction order"})
    case.metric("attn_ms", 0.31)
    payload = report.finish()
    payload["provenance"]["git"] = {"commit": COMMIT, "dirty": False}
    payload["candidate"]["commit"] = COMMIT
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_valid_op_passes(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    _write(out / "op.json")
    result = artifacts.validate_op(_expectation(out))
    assert result["correctness"] == "passed"
    assert result["metrics"]["gates_total"] == 2
    assert result["metrics"]["diagnostics_failed"] == [{"case": "decode-513", "name": "vs baseline"}]


def test_op_gate_failure_is_gate_failed(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    _write(out / "op.json", gates_ok=False)
    with pytest.raises(artifacts.ArtifactError) as excinfo:
        artifacts.validate_op(_expectation(out))
    assert excinfo.value.category == "gate_failed"
    assert excinfo.value.detail["total"] == 1


def test_op_dirty_tree_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    payload = _write(out / "op.json")
    payload["provenance"]["git"]["dirty"] = True
    (out / "op.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.validate_op(_expectation(out))


def test_op_missing_candidate_format_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    payload = _write(out / "op.json")
    payload["candidate"]["format"] = ""
    (out / "op.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.validate_op(_expectation(out))


def test_op_without_coverage_gate_cannot_be_written(tmp_path: Path) -> None:
    report = OpReport(tmp_path, "cand-x", "bf16-kv", root=tmp_path)
    case = report.case("only-one")
    case.gate("some gate", True)
    case.metric("ms", 1.0)
    with pytest.raises(ValueError, match="coverage"):
        report.finish()


def test_op_case_without_metric_cannot_be_written(tmp_path: Path) -> None:
    report = OpReport(tmp_path, "cand-x", "bf16-kv", root=tmp_path)
    case = report.case("only-one")
    case.gate("coverage: 必测项", True)
    with pytest.raises(ValueError, match="metric"):
        report.finish()


def test_op_scenario_set_must_match_request(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    _write(out / "op.json")
    with pytest.raises(artifacts.ArtifactError, match="场景"):
        artifacts.validate_op(_expectation(out, scenarios=("decode-513", "decode-8191")))


def test_op_steps_must_match_request(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    _write(out / "op.json")
    argv = ("scripts/phase2_attn_check.py", "--out", "{output}", "--steps", "8")
    with pytest.raises(artifacts.ArtifactError, match="steps"):
        artifacts.validate_op(_expectation(out, argv=argv, steps=8))


def test_submit_accepts_registered_op_script(repo: Path) -> None:
    """op 脚本已登记：submit 只看脚本在 commit 里存在 + 场景名合法。"""
    script = repo / "scripts/phase2_attn_check.py"
    script.write_text(
        "SCENARIOS = ('decode-513', 'decode-8191')\n\ndef main():\n    return 0\n",
        encoding="utf-8",
    )
    commit = conftest.git_commit(repo, "add op script")
    request = conftest.make_request(
        "explorer-attn",
        commit,
        [
            {
                "kind": "op",
                "argv": [
                    ".venv/bin/python",
                    "scripts/phase2_attn_check.py",
                    "--only",
                    "decode-513",
                    "--out",
                    "{output}/op",
                ],
                "result": "{output}/op/op.json",
            }
        ],
    )
    data = conftest.submit(repo, request)
    assert data["job_id"].startswith("job-")
    assert data["steps"][0]["kind"] == "op"


def test_submit_rejects_kind_mismatch_for_op_script(repo: Path) -> None:
    script = repo / "scripts/phase2_attn_check.py"
    script.write_text("SCENARIOS = ('decode-513',)\n", encoding="utf-8")
    commit = conftest.git_commit(repo, "add op script")
    request = conftest.make_request(
        "explorer-attn",
        commit,
        [
            {
                "kind": "bench",
                "argv": [".venv/bin/python", "scripts/phase2_attn_check.py", "--out", "{output}/op"],
                "result": "{output}/op/op.json",
            }
        ],
    )
    proc = conftest.cli(
        repo,
        "submit",
        "--request",
        str(_request_file(repo, request)),
    )
    assert proc.returncode != 0


def _request_file(repo: Path, request: dict) -> Path:
    path = repo / ".tmp" / "request-op.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_op_artifact_shape_is_stable() -> None:
    """契约里的字段名是 Explorer 脚本与校验器的接口，改名必须同步。"""
    assert artifacts.OP_SCHEMA == "phase2-op/1"
    assert "op" in schema.KINDS
    assert schema.SCRIPTS["scripts/phase2_attn_check.py"].kinds == ("op",)
    assert schema.SCRIPTS["bench/micro/nvfp4_gemv.py"].kinds == ("op",)
    assert "op" in artifacts.VALIDATORS


def test_op_harness_symlinks_do_not_count_as_dirty(tmp_path: Path) -> None:
    """job worktree 的 .venv/models 符号链接会以 untracked 出现，不算源码改动。"""
    out = tmp_path / "op"
    out.mkdir()
    payload = _write(out / "op.json")
    payload["provenance"]["git"] = {
        "commit": COMMIT,
        "dirty": True,
        "status": ["?? .venv", "?? models"],
    }
    (out / "op.json").write_text(json.dumps(payload), encoding="utf-8")
    result = artifacts.validate_op(_expectation(out))
    assert result["correctness"] == "passed"


def test_op_untracked_source_is_still_rejected(tmp_path: Path) -> None:
    out = tmp_path / "op"
    out.mkdir()
    payload = _write(out / "op.json")
    payload["provenance"]["git"] = {
        "commit": COMMIT,
        "dirty": True,
        "status": ["?? .venv", "?? runner/ops/gdn/uncommitted.py"],
    }
    (out / "op.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(artifacts.ArtifactError, match="untracked 源码"):
        artifacts.validate_op(_expectation(out))


# ---------------------------------------------------------- executor 聚合
def test_op_step_counts_as_correctness_gate() -> None:
    """只有 op 步骤的 job 也要能判成 passed，且 op 指标要进 summary.metrics。"""
    from runner.experiment import executor as exp

    request = schema.Request(
        raw={},
        session_id="s",
        commit_requested=COMMIT,
        commit=COMMIT,
        timeout_s=60.0,
        steps=(
            schema.ResolvedStep(
                index=0,
                kind="op",
                argv=("scripts/phase2_gdn_check.py",),
                result="{output}/op/op.json",
                script="scripts/phase2_gdn_check.py",
                flags={},
                scenarios=None,
            ),
        ),
        sha256="x",
    )
    record = {
        "index": 0,
        "kind": "op",
        "validation": {
            "ok": True,
            "correctness": "passed",
            "metrics": {
                "candidate": {"name": "cand", "format": "bf16-kv"},
                "cases": [{"case": "c1", "gates": 2, "gates_passed": 2}],
                "gates_total": 2,
                "gates_passed": 2,
                "gate_failures": [],
                "diagnostics_failed": [],
            },
        },
    }
    assert exp.compute_correctness([record], request) == "passed"
    metrics = exp.merge_metrics([record])
    assert metrics["op"]["gates_total"] == 2
    assert metrics["op"]["candidate"]["name"] == "cand"
    assert [case["case"] for case in metrics["op"]["cases"]] == ["c1"]

    # op gate 失败要判 failed，而不是 not_checked
    failed = {
        "index": 0,
        "kind": "op",
        "validation": {"ok": False, "correctness": "failed", "metrics": {}},
    }
    assert exp.compute_correctness([failed], request) == "failed"
