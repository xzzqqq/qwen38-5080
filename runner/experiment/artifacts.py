"""实验产物校验：check / bench / memory / profile 的结构化结果。

设计：
- 校验只读 JSON 产物，不看 stdout，也不依赖 GPU；执行器和 CPU 测试共用同一套。
- 每个校验器同时回答两个问题：
  1. 产物是否**属于本次请求**（commit、场景、steps、repeats、输入 hash、
     执行配置），所以旧的 32 步 / 1 次结果不可能被当成 511 步 / 5 次完成；
  2. 产物本身是否完整（样本数、逐步数据长度、gate 覆盖、必需 stage、trace 等）。
- gate 失败标 ``gate_failed``（correctness=failed）；缺产物、配置不一致等标
  ``invalid_artifact``（没有完成一次有效检查，correctness=not_checked）。
- diagnostic 单列，不参与 gate 判定，也不因为 diagnostic 差异改判 gate。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from runner.experiment.schema import Expectation


class ArtifactError(Exception):
    def __init__(
        self,
        reason: str,
        detail: Any = None,
        *,
        category: str = "invalid_artifact",
    ):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.category = category


def _require(
    condition: bool,
    reason: str,
    detail: Any = None,
    *,
    category: str = "invalid_artifact",
) -> None:
    if not condition:
        raise ArtifactError(reason, detail, category=category)


def _load_json(path: Path, what: str) -> Any:
    _require(path.exists(), f"缺产物 {what}", {"path": str(path)})
    _require(path.is_file(), f"{what} 不是文件", {"path": str(path)})
    _require(path.stat().st_size > 0, f"{what} 是空文件", {"path": str(path)})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            f"{what} 解析失败: {exc}", {"path": str(path)}
        ) from exc
    _require(bool(data), f"{what} 为空（异常结果）", {"path": str(path)})
    return data


def _load_obj(path: Path, what: str) -> dict:
    data = _load_json(path, what)
    _require(isinstance(data, dict), f"{what} 必须是对象", {"type": type(data).__name__})
    return data


def _git_commit(prov: Any, expected: str, where: str) -> None:
    git = (prov or {}).get("git") if isinstance(prov, dict) else None
    got = (git or {}).get("commit")
    _require(bool(got), f"{where} 缺 provenance.git.commit", {"where": where})
    _require(
        got == expected,
        f"{where} 的 commit 与请求不一致",
        {"expected": expected, "got": got},
    )
    _require(
        git.get("dirty") is False,
        f"{where} 的源码不是 clean worktree（dirty=true），结果不可复现",
        {"dirty": git.get("dirty")},
    )


def _check_scenarios(got: list[str], expected: tuple[str, ...] | None, where: str) -> None:
    _require(bool(got), f"{where} 没有场景")
    if expected is None:
        return
    missing = [name for name in expected if name not in got]
    extra = [name for name in got if name not in expected]
    _require(
        not missing and not extra,
        f"{where} 的场景与请求不一致",
        {"missing": missing, "extra": extra, "expected": list(expected), "got": got},
    )
    _require(len(got) == len(set(got)), f"{where} 场景重复", got)


def _flag_value(argv: Any, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        if isinstance(token, str) and token.startswith(flag + "="):
            return token[len(flag) + 1 :]
    return None


def _as_int(value: Any, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{what} 不是整数: {value!r}") from exc


def _finite(
    value: Any,
    what: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{what} 不是数字: {value!r}") from exc
    _require(math.isfinite(number), f"{what} 不是有限数", {"got": value})
    if strictly_positive:
        _require(number > 0, f"{what} 必须为正数", {"got": number})
    elif minimum is not None:
        _require(number >= minimum, f"{what} 不能小于 {minimum}", {"got": number})
    return number


def _positive(value: Any, what: str) -> float:
    return _finite(value, what, strictly_positive=True)


def _resolve_fixtures(value: str, expect: Expectation) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if expect.worktree is None:
        return None
    return (expect.worktree / path).resolve()


def _fixtures_manifest_sha(dir_path: Path) -> str | None:
    """按 bench/m3_check 的口径 hash fixtures manifest 对象。"""
    try:
        payload = json.loads((dir_path / "manifest.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _fixtures_sha(reports: list[dict], where: str) -> str | None:
    hashes = set()
    for report in reports:
        fixtures = report.get("fixtures") or {}
        sha = fixtures.get("manifest_sha256")
        if sha:
            hashes.add(str(sha))
    _require(len(hashes) <= 1, f"{where} 各场景的 fixture hash 不一致", sorted(hashes))
    return next(iter(hashes)) if hashes else None


def _check_fixtures_dir(report: dict, expect: Expectation, where: str) -> None:
    if expect.fixtures is None or expect.worktree is None:
        return
    got = (report.get("fixtures") or {}).get("dir")
    if not got:
        return
    requested = Path(expect.fixtures)
    wanted = requested if requested.is_absolute() else expect.worktree / requested
    try:
        got_path = Path(got)
        if not got_path.is_absolute():
            got_path = expect.worktree / got_path
        same = got_path.resolve() == wanted.resolve()
    except OSError:
        same = False
    _require(
        same,
        f"{where} 的 fixtures 目录与请求不一致",
        {"expected": str(wanted), "got": str(got)},
    )


# ------------------------------------------------------------------- check
def validate_check(expect: Expectation) -> dict:
    result_path = expect.result_path
    data = _load_obj(result_path, "check.json")
    out_dir = result_path.parent
    top_prov = _load_obj(out_dir / "provenance.json", "check provenance.json")
    _git_commit(top_prov, expect.commit, "check provenance")

    summary = data.get("summary") or {}
    _require(isinstance(summary, dict), "check.json.summary 不是对象")
    summary_ok = summary.get("ok") is True
    scenario_entries = summary.get("scenarios") or []
    _require(bool(scenario_entries), "check.json 没有场景")
    names = [str(entry.get("scenario")) for entry in scenario_entries]
    _check_scenarios(names, expect.scenarios, "check")

    results = data.get("results") or []
    _require(isinstance(results, list) and bool(results), "check.json 没有 results")
    by_name = {str(entry.get("scenario")): entry for entry in results}
    _require(
        set(by_name) == set(names),
        "check.json 的 summary/results 场景集合不一致",
        {"summary": names, "results": list(by_name)},
    )

    scenarios_metrics: list[dict] = []
    gate_total = 0
    gate_passed = 0
    gate_failures: list[dict] = []
    diagnostics_failed: list[dict] = []
    diagnostic_total = 0
    reports: list[dict] = []

    non_gate_failures: list[str] = []
    for name in names:
        entry = by_name[name]
        if entry.get("ok") is not True:
            non_gate_failures.append(f"{name}: 场景 ok=false")
        if _as_int(entry.get("returncode", -1), f"check {name} returncode") != 0:
            non_gate_failures.append(
                f"{name}: 子进程退出码 {entry.get('returncode')}"
            )
        report_path = out_dir / name / "report.json"
        report = _load_obj(report_path, f"check {name}/report.json")
        reports.append(report)
        _git_commit(report.get("provenance"), expect.commit, f"check {name} report")
        _require(
            report.get("scenario") == name,
            f"check {name}: report.scenario={report.get('scenario')!r} 不匹配",
        )
        if report.get("ok") is not True:
            non_gate_failures.append(f"{name}: report.ok=false")
        items = report.get("items") or []
        _require(isinstance(items, list) and bool(items), f"check {name}: items 为空")
        gates = [item for item in items if item.get("level") == "gate"]
        diagnostics = [item for item in items if item.get("level") == "diagnostic"]
        _require(bool(gates), f"check {name}: 没有 gate（不能只靠 diagnostic 通过）")
        _require(
            any(str(item.get("name", "")).startswith("coverage:") for item in gates),
            f"check {name}: 缺 coverage gate（无法证明必测项齐全）",
        )
        bad = [item for item in gates if not item.get("ok")]
        for item in bad:
            gate_failures.append(
                {"scenario": name, "name": item.get("name"), "detail": item.get("detail")}
            )
        gate_total += len(gates)
        gate_passed += len(gates) - len(bad)
        diagnostic_total += len(diagnostics)
        for item in diagnostics:
            if not item.get("ok"):
                diagnostics_failed.append(
                    {"scenario": name, "name": item.get("name")}
                )

        worker_argv = (report.get("provenance") or {}).get("argv") or []
        handled_steps = _flag_value(worker_argv, "--steps")
        if handled_steps is not None:
            _require(
                _as_int(handled_steps, f"check {name} --steps") == expect.steps,
                f"check {name}: 实际 steps={handled_steps} != 请求 {expect.steps}",
            )
        _check_fixtures_dir(report, expect, f"check {name}")
        fixtures_meta = report.get("fixtures") or {}
        fixtures_value = fixtures_meta.get("dir")
        _require(bool(fixtures_value), f"check {name}: 缺 fixtures.dir")
        fixtures_dir = _resolve_fixtures(str(fixtures_value), expect)
        _require(fixtures_dir is not None, f"check {name}: 无法解析 fixtures 目录")
        # 执行期 worktree 还在，必须重算 manifest hash；成功 job 的 worktree 会被删，
        # 事后复查时只能依赖记录值与跨场景一致性。
        if fixtures_dir.exists():
            expected_fixtures_sha = _fixtures_manifest_sha(fixtures_dir)
            _require(
                expected_fixtures_sha is not None,
                f"check {name}: 无法读取 fixtures manifest，不能确认输入身份",
                {"dir": str(fixtures_dir)},
            )
            _require(
                fixtures_meta.get("manifest_sha256") == expected_fixtures_sha,
                f"check {name}: fixtures hash 与 manifest 内容不一致",
                {"artifact": fixtures_meta.get("manifest_sha256"), "manifest": expected_fixtures_sha},
            )
        else:
            _require(
                bool(fixtures_meta.get("manifest_sha256")),
                f"check {name}: 缺 fixtures manifest hash",
            )

        scenarios_metrics.append(
            {
                "scenario": name,
                "gates": len(gates),
                "gates_passed": len(gates) - len(bad),
                "gate_failures": [item.get("name") for item in bad],
                "diagnostics": len(diagnostics),
                "diagnostics_failed": [item.get("name") for item in diagnostics if not item.get("ok")],
                "steps": expect.steps,
            }
        )

    fixtures_sha = _fixtures_sha(reports, "check")
    # gate 失败优先判成 correctness=failed；其余失败（退出码/ok=false/空结果）
    # 说明没有完成一次有效检查，correctness 保持 not_checked。
    _require(
        not gate_failures,
        "check gate 未全部通过",
        {"failures": gate_failures[:8], "total": len(gate_failures)},
        category="gate_failed",
    )
    _require(
        not non_gate_failures,
        "check 没有完整跑通，不能算通过",
        {"failures": non_gate_failures[:8]},
    )
    _require(summary_ok, "check.json summary.ok != true", summary)

    metrics = {
        "scenarios": scenarios_metrics,
        "gates_total": gate_total,
        "gates_passed": gate_passed,
        "gate_failures": [],
        "diagnostics_total": diagnostic_total,
        "diagnostics_failed": diagnostics_failed,
        "fixtures_sha256": fixtures_sha,
        "steps": expect.steps,
    }
    return {
        "kind": "check",
        "correctness": "passed",
        "metrics": metrics,
        "checks": [
            f"{len(names)} 场景 gate 全部通过（{gate_total} 项）",
            "coverage gate 覆盖必测项",
            "commit/场景/steps/fixtures 与请求一致",
        ],
    }


# ------------------------------------------------------------------- bench
def validate_bench(expect: Expectation) -> dict:
    result_path = expect.result_path
    data = _load_obj(result_path, "bench.json")
    out_dir = result_path.parent
    top_prov = _load_obj(out_dir / "provenance.json", "bench provenance.json")
    _git_commit(top_prov, expect.commit, "bench provenance")

    _require(data.get("complete") is True, "bench.json complete!=true（可能被中断）")
    scenario_names = data.get("scenarios") or []
    _require(isinstance(scenario_names, list) and bool(scenario_names), "bench.json 没有 scenarios")
    _check_scenarios([str(name) for name in scenario_names], expect.scenarios, "bench")

    _require(
        _as_int(data.get("steps", -1), "bench steps") == expect.steps,
        f"bench.json steps={data.get('steps')} != 请求 {expect.steps}",
    )
    _require(
        _as_int(data.get("repeats", -1), "bench repeats") == expect.repeats,
        f"bench.json repeats={data.get('repeats')} != 请求 {expect.repeats}",
    )
    matrix = top_prov.get("run_config") or {}
    _require(bool(matrix), "bench provenance 缺 run_config（无法确认本次配置）")
    for key, wanted in (
        ("steps", expect.steps),
        ("repeats", expect.repeats),
        ("chunk_size", expect.chunk_size),
        ("graph_warmup", expect.graph_warmup),
    ):
        _require(
            _as_int(matrix.get(key, -1), f"provenance.run_config.{key}") == wanted,
            f"bench provenance.run_config.{key}={matrix.get(key)} != 请求 {wanted}",
        )
    _require(
        matrix.get("repeats_non_graph") == expect.repeats_non_graph,
        "bench provenance.run_config.repeats_non_graph 与请求不一致",
        {"expected": expect.repeats_non_graph, "got": matrix.get("repeats_non_graph")},
    )
    matrix_scenarios = matrix.get("scenarios") or []
    _check_scenarios([str(name) for name in matrix_scenarios], expect.scenarios, "bench provenance")
    _require(
        matrix.get("fixtures_sha256"), "bench provenance.run_config 缺 fixtures_sha256"
    )
    _require(
        bool(top_prov.get("environment_fingerprint")),
        "bench provenance 缺 environment_fingerprint",
    )

    recorded_fixtures = top_prov.get("fixtures")
    _require(
        isinstance(recorded_fixtures, str) and bool(recorded_fixtures),
        "bench provenance 缺 fixtures 路径，不能确认输入身份",
    )
    recorded_fixtures_dir = _resolve_fixtures(recorded_fixtures, expect)
    _require(recorded_fixtures_dir is not None, "无法解析 bench fixtures 目录（缺 worktree）")
    if expect.fixtures is not None:
        requested_fixtures_dir = _resolve_fixtures(expect.fixtures, expect)
        _require(
            requested_fixtures_dir == recorded_fixtures_dir,
            "bench 实际使用的 fixtures 与请求不一致",
            {"requested": str(requested_fixtures_dir), "actual": str(recorded_fixtures_dir)},
        )
    if recorded_fixtures_dir.exists():
        expected_fixtures_sha = _fixtures_manifest_sha(recorded_fixtures_dir)
        _require(
            expected_fixtures_sha is not None,
            "无法读取 bench fixtures manifest，不能确认输入身份",
            {"dir": str(recorded_fixtures_dir)},
        )
        _require(
            matrix.get("fixtures_sha256") == expected_fixtures_sha,
            "bench provenance fixtures_sha256 与 manifest 内容不一致",
            {"recorded": matrix.get("fixtures_sha256"), "manifest": expected_fixtures_sha},
        )
    else:
        expected_fixtures_sha = str(matrix.get("fixtures_sha256") or "")
        _require(
            bool(expected_fixtures_sha),
            "bench provenance 缺 fixtures_sha256，不能确认输入身份",
        )

    results = data.get("results") or []
    _require(isinstance(results, list) and bool(results), "bench.json 没有 results")
    got_names = [str(entry.get("scenario")) for entry in results]
    _require(
        got_names == [str(name) for name in scenario_names],
        "bench.json 的 results 顺序/集合与 scenarios 不一致",
        {"scenarios": scenario_names, "results": got_names},
    )

    scenarios_metrics: list[dict] = []
    fixtures_hashes: set[str] = set()
    for entry in results:
        name = str(entry.get("scenario"))
        _require(entry.get("ok") is True, f"bench {name}: ok=false", entry.get("error"))
        spec = entry.get("spec") or {}
        _require(bool(spec), f"bench {name}: 缺 spec")
        execution = str(spec.get("execution"))
        expected_repeats = expect.expected_repeats(execution)
        _require(
            _as_int(entry.get("repeats_requested", -1), f"bench {name} repeats") == expected_repeats,
            f"bench {name}: 实际 repeats={entry.get('repeats_requested')} "
            f"!= 请求 {expected_repeats}（execution={execution}）",
        )
        steps = _as_int(entry.get("steps", -1), f"bench {name} steps")
        _require(
            steps == expect.steps,
            f"bench {name}: 实际 steps={steps} != 请求 {expect.steps}",
        )
        config = entry.get("run_config") or {}
        _require(bool(config), f"bench {name}: 缺 run_config")
        for key, wanted in (
            ("steps", steps),
            ("repeats", expected_repeats),
            ("chunk_size", expect.chunk_size),
            ("graph_warmup", expect.graph_warmup),
        ):
            _require(
                _as_int(config.get(key, -1), f"bench {name} run_config.{key}") == wanted,
                f"bench {name}: run_config.{key}={config.get(key)} != {wanted}",
            )
        _require(
            config.get("scenario") == name, f"bench {name}: run_config.scenario 不匹配"
        )
        _require(
            bool(entry.get("environment_fingerprint")),
            f"bench {name}: 缺 environment_fingerprint",
        )
        _git_commit(entry.get("provenance"), expect.commit, f"bench {name} result")
        fixtures_sha = config.get("fixtures_sha256")
        _require(bool(fixtures_sha), f"bench {name}: 缺 fixtures_sha256")
        _require(
            fixtures_sha == expected_fixtures_sha,
            f"bench {name}: fixtures_sha256 与 manifest 内容不一致",
            {"recorded": fixtures_sha, "manifest": expected_fixtures_sha},
        )
        fixtures_hashes.add(str(fixtures_sha))

        samples = entry.get("samples") or []
        _require(isinstance(samples, list), f"bench {name}: samples 不是数组")
        _require(
            len(samples) == expected_repeats,
            f"bench {name}: 样本数 {len(samples)} != repeats {expected_repeats}",
        )
        for index, sample in enumerate(samples):
            model = sample.get("model") or {}
            system = sample.get("system") or {}
            tokens = sample.get("tokens") or {}
            per_step = model.get("per_step_ms") or []
            _require(
                len(per_step) == steps,
                f"bench {name} sample {index}: per_step_ms {len(per_step)} != steps {steps}",
            )
            for step_index, value in enumerate(per_step):
                _finite(
                    value,
                    f"bench {name} sample {index} per_step_ms[{step_index}]",
                    strictly_positive=True,
                )
            _require(
                _as_int(tokens.get("timed_steps", -1), f"bench {name} timed_steps") == steps,
                f"bench {name} sample {index}: timed_steps != steps",
            )
            _require(
                _as_int(tokens.get("generated", -1), f"bench {name} generated") == steps + 1,
                f"bench {name} sample {index}: generated != steps+1",
            )
            _require(
                isinstance(tokens.get("matches_frozen_trace"), bool),
                f"bench {name} sample {index}: 缺 matches_frozen_trace",
            )
            _positive(model.get("model_tok_s"), f"bench {name} sample {index} model_tok_s")
            _positive(system.get("system_tok_s"), f"bench {name} sample {index} system_tok_s")
            _positive(
                (sample.get("request") or {}).get("request_tok_s"),
                f"bench {name} sample {index} request_tok_s",
            )
            _positive(
                model.get("peak_allocated_gb"),
                f"bench {name} sample {index} peak_allocated_gb",
            )
            _positive(
                model.get("peak_reserved_gb"),
                f"bench {name} sample {index} peak_reserved_gb",
            )
            _finite(
                model.get("free_gb"),
                f"bench {name} sample {index} free_gb",
                minimum=0.0,
            )
            _finite(
                (sample.get("prefill") or {}).get("prefill_s"),
                f"bench {name} sample {index} prefill_s",
                minimum=0.0,
            )
            if "ttft_s" in (sample.get("request") or {}):
                _positive(
                    (sample.get("request") or {}).get("ttft_s"),
                    f"bench {name} sample {index} ttft_s",
                )

        summary = entry.get("summary") or {}
        _require(isinstance(summary, dict) and bool(summary), f"bench {name}: summary 为空")
        for key in (
            "model_tok_s",
            "system_tok_s",
            "request_tok_s",
            "step_ms",
            "prefill_s",
            "peak_allocated_gb",
            "peak_reserved_gb",
            "free_gb_min",
            "trace_match_all",
        ):
            _require(key in summary, f"bench {name}: summary 缺 {key}")
        for key in ("model_tok_s", "system_tok_s", "request_tok_s"):
            stats = summary[key] or {}
            _require(
                _as_int(stats.get("n", -1), f"bench {name} summary.{key}.n")
                == expected_repeats,
                f"bench {name}: summary.{key}.n != repeats",
            )
            for field in ("median", "min", "max"):
                _positive(
                    stats.get(field), f"bench {name} summary.{key}.{field}"
                )
            if "std" in stats:
                _finite(stats["std"], f"bench {name} summary.{key}.std", minimum=0.0)
        step_ms = summary["step_ms"] or {}
        _require(
            _as_int(step_ms.get("n", -1), f"bench {name} summary.step_ms.n")
            == steps * expected_repeats,
            f"bench {name}: summary.step_ms.n != steps*repeats",
        )
        for field in ("p50_ms", "p95_ms"):
            _positive(step_ms.get(field), f"bench {name} summary.step_ms.{field}")
        for field in ("mean_ms", "min_ms", "max_ms"):
            if field in step_ms:
                _positive(step_ms[field], f"bench {name} summary.step_ms.{field}")
        if "std_ms" in step_ms:
            _finite(step_ms["std_ms"], f"bench {name} summary.step_ms.std_ms", minimum=0.0)
        prefill_stats = summary["prefill_s"] or {}
        _positive(prefill_stats.get("median"), f"bench {name} summary.prefill_s.median")
        if summary.get("ttft_s"):
            _positive(
                (summary["ttft_s"] or {}).get("median"),
                f"bench {name} summary.ttft_s.median",
            )
        _positive(summary["peak_allocated_gb"], f"bench {name} summary.peak_allocated_gb")
        _positive(summary["peak_reserved_gb"], f"bench {name} summary.peak_reserved_gb")
        _finite(summary["free_gb_min"], f"bench {name} summary.free_gb_min", minimum=0.0)
        _require(
            isinstance(summary["trace_match_all"], bool),
            f"bench {name}: summary.trace_match_all 必须是 bool",
        )

        scenarios_metrics.append(
            {
                "scenario": name,
                "execution": execution,
                "depth": spec.get("depth"),
                "capacity": spec.get("capacity"),
                "steps": steps,
                "repeats": expected_repeats,
                "model_tok_s": summary["model_tok_s"],
                "system_tok_s": summary["system_tok_s"],
                "request_tok_s": summary["request_tok_s"],
                "step_ms": {
                    "p50_ms": step_ms.get("p50_ms"),
                    "p95_ms": step_ms.get("p95_ms"),
                },
                "prefill_s": summary["prefill_s"],
                "ttft_s": (summary.get("ttft_s") or {}),
                "peak_allocated_gb": summary.get("peak_allocated_gb"),
                "peak_reserved_gb": summary.get("peak_reserved_gb"),
                "free_gb_min": summary.get("free_gb_min"),
                "trace_match_all": summary.get("trace_match_all"),
                "fixtures_sha256": fixtures_sha,
                "timing": "model=逐 step CUDA events（D2D copy 在 start event 之前）；"
                "system=真实贪心闭环墙钟；request=prefill+decode",
            }
        )

    _require(
        len(fixtures_hashes) <= 1, "bench 各场景 fixture hash 不一致", sorted(fixtures_hashes)
    )
    if fixtures_hashes:
        _require(
            fixtures_hashes == {str(matrix.get("fixtures_sha256"))},
            "bench 场景与 provenance 的 fixture hash 不一致",
            {"scenarios": sorted(fixtures_hashes), "matrix": matrix.get("fixtures_sha256")},
        )

    metrics = {
        "scenarios": scenarios_metrics,
        "steps": expect.steps,
        "repeats": expect.repeats,
        "repeats_non_graph": expect.repeats_non_graph,
        "fixtures_sha256": next(iter(fixtures_hashes), None),
    }
    return {
        "kind": "bench",
        "correctness": "not_checked",
        "metrics": metrics,
        "checks": [
            "complete=true，样本数 = repeats",
            "逐步数据长度 = steps，generated = steps+1",
            "场景/steps/repeats/chunk/graph_warmup/fixtures 与请求一致",
        ],
    }


# ------------------------------------------------------------------- memory
MEMORY_REQUIRED_STAGES = ("load", "cache_init", "prefill_cold", "decode")


def validate_memory(expect: Expectation) -> dict:
    result_path = expect.result_path
    data = _load_obj(result_path, "memory.json")
    out_dir = result_path.parent
    top_prov = _load_obj(out_dir / "provenance.json", "memory provenance.json")
    _git_commit(top_prov, expect.commit, "memory provenance")
    _require(data.get("complete") is True, "memory.json complete!=true")
    matrix = top_prov.get("run_config") or {}
    if matrix:
        _require(
            _as_int(matrix.get("steps", -1), "memory provenance.steps") == expect.steps,
            "memory provenance.steps 与请求不一致",
        )

    results = data.get("results") or []
    _require(isinstance(results, list) and bool(results), "memory.json 没有 results")
    names = [str(entry.get("scenario")) for entry in results]
    _check_scenarios(names, expect.scenarios, "memory")

    scenarios_metrics: list[dict] = []
    for entry in results:
        name = str(entry.get("scenario"))
        _require(entry.get("ok") is True, f"memory {name}: ok=false", entry.get("error"))
        steps = _as_int(entry.get("steps", -1), f"memory {name} steps")
        _require(
            steps == expect.steps,
            f"memory {name}: 实际 steps={steps} != 请求 {expect.steps}",
        )
        stages = entry.get("stages") or []
        _require(isinstance(stages, list), f"memory {name}: stages 不是数组")
        _require(len(stages) >= 6, f"memory {name}: stage 数 {len(stages)} < 6")
        stage_names = {str(row.get("stage")) for row in stages}
        missing = [stage for stage in MEMORY_REQUIRED_STAGES if stage not in stage_names]
        _require(not missing, f"memory {name}: 缺必需 stage", {"missing": missing})
        determinism = entry.get("determinism") or {}
        _require(
            determinism.get("prefill_logits_equal") is True,
            f"memory {name}: reset 后 prefill logits 不一致",
        )
        _require(
            determinism.get("step_logits_equal") is True,
            f"memory {name}: reset 后 decode logits 不一致",
        )
        growth = entry.get("growth") or {}
        for key in (
            "allocated_delta_gb",
            "reserved_delta_gb",
            "cycle2_allocated_delta_gb",
            "cycle2_reserved_delta_gb",
        ):
            _require(key in growth, f"memory {name}: growth 缺 {key}")
            _finite(growth[key], f"memory {name}: growth.{key}")
        _require(
            isinstance(growth.get("kv_data_ptr_stable"), bool),
            f"memory {name}: growth.kv_data_ptr_stable 必须是 bool",
        )
        for row in stages:
            for key in (
                "peak_allocated_gb",
                "peak_reserved_gb",
                "end_allocated_gb",
                "end_reserved_gb",
                "free_gb",
            ):
                if key in row:
                    _finite(row[key], f"memory {name}: stage {row.get('stage')}.{key}", minimum=0.0)
        peaks = entry.get("peaks") or {}
        for key in (
            "load_peak_allocated_gb",
            "max_peak_allocated_gb",
            "max_peak_reserved_gb",
            "min_free_gb",
        ):
            _require(key in peaks, f"memory {name}: peaks 缺 {key}")
        for key in ("load_peak_allocated_gb", "max_peak_allocated_gb", "max_peak_reserved_gb"):
            _positive(peaks[key], f"memory {name} {key}")
        _positive(peaks["min_free_gb"], f"memory {name} min_free_gb")
        spec = entry.get("spec") or {}
        if str(spec.get("execution")) == "graph":
            _require(
                "max_phase_peak_allocated_gb" in (entry.get("graph") or {}),
                f"memory {name}: 缺 graph 子阶段峰值",
            )
        scenarios_metrics.append(
            {
                "scenario": name,
                "execution": spec.get("execution"),
                "depth": spec.get("depth"),
                "capacity": spec.get("capacity"),
                "steps": steps,
                "stages": len(stages),
                "peaks": {
                    "load_peak_allocated_gb": peaks["load_peak_allocated_gb"],
                    "max_peak_allocated_gb": peaks["max_peak_allocated_gb"],
                    "max_peak_reserved_gb": peaks["max_peak_reserved_gb"],
                    "min_free_gb": peaks["min_free_gb"],
                },
                "growth": {
                    "allocated_delta_gb": growth["allocated_delta_gb"],
                    "cycle2_allocated_delta_gb": growth["cycle2_allocated_delta_gb"],
                    "cycle2_reserved_delta_gb": growth["cycle2_reserved_delta_gb"],
                    "kv_data_ptr_stable": growth["kv_data_ptr_stable"],
                },
                "determinism": {
                    "prefill_logits_equal": True,
                    "step_logits_equal": True,
                },
            }
        )

    metrics = {"scenarios": scenarios_metrics, "steps": expect.steps}
    return {
        "kind": "memory",
        "correctness": "not_checked",
        "metrics": metrics,
        "checks": [
            "complete=true，steps 与请求一致",
            "必需 stage 齐全，reset 后数值一致",
            "无跨生命周期增长，峰值/free 有效",
        ],
    }


# ------------------------------------------------------------------- profile
def _resolve_csv(path: Path) -> Path | None:
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    return gz if gz.exists() else None


def validate_profile(expect: Expectation) -> dict:
    result_path = expect.result_path
    timeline = _load_obj(result_path, "timeline_summary.json")
    out_dir = result_path.parent
    top_prov = _load_obj(out_dir / "provenance.json", "profile provenance.json")
    _git_commit(top_prov, expect.commit, "profile provenance")
    _require((out_dir / "summary.md").exists(), "profile 缺 summary.md")
    _require((out_dir / "summary.md").stat().st_size > 0, "profile summary.md 为空")

    case_dirs = sorted(path for path in out_dir.iterdir() if path.is_dir())
    _require(bool(case_dirs), "profile 没有 case 目录")
    case_names = [path.name for path in case_dirs]
    missing = [name for name in case_names if name not in timeline]
    _require(not missing, "profile timeline_summary 缺 case", {"missing": missing})

    scenarios_metrics: list[dict] = []
    for case_dir in case_dirs:
        name = case_dir.name
        stats = timeline.get(name)
        _require(isinstance(stats, dict) and bool(stats), f"profile {name}: 时间线为空")
        meta = _load_obj(case_dir / "meta.json", f"profile {name}/meta.json")
        _git_commit(meta.get("provenance"), expect.commit, f"profile {name} meta")
        _require((case_dir / "trace.nsys-rep").exists(), f"profile {name}: 缺 trace.nsys-rep")
        _require(
            (case_dir / "trace.nsys-rep").stat().st_size > 0,
            f"profile {name}: trace.nsys-rep 为空",
        )
        _require(_resolve_csv(case_dir / "gpu_trace.csv") is not None, f"profile {name}: 缺 gpu_trace.csv(.gz)")
        _require(
            _as_int(meta.get("steps", -1), f"profile {name} meta.steps") == expect.steps,
            f"profile {name}: meta.steps={meta.get('steps')} != 请求 {expect.steps}",
        )
        steps = _as_int(stats.get("steps", -1), f"profile {name} steps")
        _require(
            steps == expect.steps,
            f"profile {name}: 时间线 steps={steps} != 请求 {expect.steps}",
        )
        for key in ("kernels", "per_step_kernels", "per_step_kernel_us", "per_step_wall_us"):
            _positive(stats.get(key), f"profile {name} {key}")
        if "tiny_kernel_count_pct" in stats:
            pct = _finite(
                stats["tiny_kernel_count_pct"],
                f"profile {name}: tiny_kernel_count_pct",
                minimum=0.0,
            )
            _require(pct <= 100.0, f"profile {name}: tiny_kernel_count_pct 越界", {"got": pct})
        top = stats.get("top") or []
        scenarios_metrics.append(
            {
                "case": name,
                "execution": meta.get("execution"),
                "depth": meta.get("depth"),
                "capacity": meta.get("capacity"),
                "steps": steps,
                "kernels": stats.get("kernels"),
                "per_step_kernels": stats.get("per_step_kernels"),
                "per_step_kernel_us": stats.get("per_step_kernel_us"),
                "per_step_wall_us": stats.get("per_step_wall_us"),
                "tiny_kernel_count_pct": stats.get("tiny_kernel_count_pct"),
                "top_kernel": (top[0].get("name") if top and isinstance(top[0], dict) else None),
            }
        )

    metrics = {"scenarios": scenarios_metrics, "steps": expect.steps}
    return {
        "kind": "profile",
        "correctness": "not_checked",
        "metrics": metrics,
        "checks": [
            "timeline_summary / summary.md / provenance 齐全",
            "每个 case 有 nsys trace 与 gpu_trace.csv",
            "case steps 与请求一致，每步 kernel 数有效",
        ],
    }


VALIDATORS = {
    "check": validate_check,
    "bench": validate_bench,
    "memory": validate_memory,
    "profile": validate_profile,
}


def validate(kind: str, expect: Expectation) -> dict:
    """按 kind 校验产物；返回 metrics/checks，失败抛 ArtifactError。"""
    if kind not in VALIDATORS:
        raise ArtifactError(f"没有 {kind!r} 的产物校验函数（不允许提交）")
    _require(
        expect.result_path.resolve().is_relative_to(expect.output_dir.resolve()),
        "result 不在本 job 的 output 目录内",
        {"result": str(expect.result_path), "output": str(expect.output_dir)},
    )
    result = VALIDATORS[kind](expect)
    result.setdefault("ok", True)
    return result


__all__ = [
    "ArtifactError",
    "VALIDATORS",
    "PLACEHOLDER",
    "validate",
]
