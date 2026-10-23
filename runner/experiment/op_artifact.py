"""Phase 2 新增算子的实验产物（``op`` kind）写入器。

设计：
- Phase 2 的每个候选 kernel 都要有自己的 ≤5 分钟反馈入口（op gate + 局部
  microbench）。这些入口不加载整份模型，产物也和 ``m3_check.py`` 的
  fixtures/report 结构不同，所以单独定义一种 kind，不去改 m3 的检查器。
- 契约由 ``runner/experiment/artifacts.validate_op`` 校验；本模块负责把同一个
  契约写正确，避免每个脚本各写一份 JSON 各错各的。
- **每个 case 必须有 gate 和 metric**：只有 diagnostic 的产物不能算通过
  （``not_checked`` 不能当证据），没有数值的 microbench 也不能入选。
- gate 名以 ``coverage:`` 开头的项用来证明“必测 shape / 边界 / state 齐全”，
  和 m3_check 的 coverage gate 同一口径；缺它整份产物判失败。

产物（``{output}/op.json``）：

    {
      "schema": "phase2-op/1",
      "candidate": {"name": ..., "format": ..., "commit": ..., ...},
      "provenance": {"git": {"commit", "dirty", ...}, "argv": [...], "environment": {...}},
      "summary": {"ok": bool, "cases": [{"case", "gates", "gates_passed", "gates_failed"}]},
      "cases": [{"case", "config", "gates": [...], "diagnostics": [...], "metrics": {...}}]
    }

数据流：脚本算完一个 case -> ``report.case(name, **config)`` -> ``gate()`` /
``diag()`` / ``metric()`` -> 全部 case 完成 -> ``report.finish()`` 原子写出 JSON。

预期瓶颈：与候选 kernel 无关；写文件是毫秒级，不影响计时区间。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "phase2-op/1"


def _repo_root(start: Path | None = None) -> Path:
    """当前 worktree 的根：job 里就是 job worktree，dev 时是 Explorer worktree。"""
    cwd = Path(start or Path.cwd())
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - 不在 git 里时退回 cwd
        return cwd
    return Path(out) if out else cwd


def _git_state(root: Path) -> dict[str, Any]:
    """commit + dirty（只关心 tracked 改动和 untracked 源码，与 report.git_state 同口径）。"""
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines()[:20] if status else [],
    }


def _numbers_only(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = float(value)
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                out[key] = number
            continue
        raise TypeError(f"metric {key!r} 必须是有限数值，得到 {type(value).__name__}")
    return out


class OpCase:
    """一个 case 的 gate / diagnostic / metric 容器。"""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = dict(config)
        self.gates: list[dict[str, Any]] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.metrics: dict[str, float] = {}

    def gate(self, name: str, ok: bool, detail: Any = None) -> bool:
        item: dict[str, Any] = {"name": name, "ok": bool(ok)}
        if detail is not None:
            item["detail"] = detail
        self.gates.append(item)
        return bool(ok)

    def diag(self, name: str, ok: bool, detail: Any = None) -> None:
        """diagnostic：记录差异和解释，不参与判定。"""
        item: dict[str, Any] = {"name": name, "ok": bool(ok)}
        if detail is not None:
            item["detail"] = detail
        self.diagnostics.append(item)

    def metric(self, name: str, value: float) -> None:
        self.metrics[name] = _numbers_only({name: value})[name]

    @property
    def ok(self) -> bool:
        return bool(self.gates) and all(item["ok"] for item in self.gates)

    def has_coverage(self) -> bool:
        return any(str(item["name"]).startswith("coverage:") for item in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.name,
            "config": self.config,
            "gates": self.gates,
            "diagnostics": self.diagnostics,
            "metrics": self.metrics,
        }


class OpReport:
    """``op.json`` 的写入器：按 case 收集，最后一次性写出。

    Args:
        out_dir: ``--out`` 目录；``op.json`` 直接写在这里。
        candidate_name: 候选标识，例如 ``attn-gqa-bf16``。
        candidate_format: 数值格式，例如 ``bf16-kv`` / ``nvfp4-w4a4``；
            SOL 和可比性都依赖它，不能留空。
        candidate_extra: 候选的其它身份信息（backend、参数、patch 说明）。
    """

    def __init__(
        self,
        out_dir: Path | str,
        candidate_name: str,
        candidate_format: str,
        *,
        candidate_extra: dict[str, Any] | None = None,
        argv: list[str] | None = None,
        root: Path | None = None,
        environment: dict[str, Any] | None = None,
        steps: int | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.root = _repo_root(root)
        self.candidate_name = str(candidate_name)
        self.candidate_format = str(candidate_format)
        self.candidate_extra = dict(candidate_extra or {})
        self.argv = list(argv if argv is not None else _argv())
        self.environment = dict(environment or {})
        self.steps = steps
        self.cases: list[OpCase] = []
        self._finished = False

    def case(self, name: str, **config: Any) -> OpCase:
        case = OpCase(str(name), config)
        self.cases.append(case)
        return case

    def finish(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("OpReport.finish() 只能调用一次")
        if not self.cases:
            raise ValueError("OpReport 没有任何 case")
        if not self.candidate_name or not self.candidate_format:
            raise ValueError("candidate_name / candidate_format 不能为空")
        for case in self.cases:
            if not case.gates:
                raise ValueError(f"case {case.name!r} 没有 gate；只有 diagnostic 不能算通过")
            if not case.has_coverage():
                raise ValueError(
                    f"case {case.name!r} 缺 coverage: gate（无法证明必测项齐全）"
                )
            if not case.metrics:
                raise ValueError(f"case {case.name!r} 没有 metric")
        state = _git_state(self.root)
        summary_cases = [
            {
                "case": case.name,
                "gates": len(case.gates),
                "gates_passed": sum(1 for item in case.gates if item["ok"]),
                "gates_failed": [item["name"] for item in case.gates if not item["ok"]],
            }
            for case in self.cases
        ]
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "candidate": {
                "name": self.candidate_name,
                "format": self.candidate_format,
                "commit": state["commit"],
                **self.candidate_extra,
            },
            "provenance": {
                "git": state,
                "argv": self.argv,
                "cwd": str(Path.cwd()),
                "environment": self.environment,
            },
            "steps": self.steps,
            "summary": {
                "ok": all(case.ok for case in self.cases),
                "cases": summary_cases,
            },
            "cases": [case.to_dict() for case in self.cases],
        }
        if extra:
            payload.update(extra)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "op.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        self._finished = True
        return payload


def _argv() -> list[str]:
    import sys

    return list(sys.argv)


__all__ = ["SCHEMA", "OpCase", "OpReport"]
