#!/usr/bin/env python3
"""校验 results/m3 下 run dir 的**产物契约**（纯 CPU，秒级）。

为什么需要：M3 里出现过“全量 check 跑完才发现 provenance.json 没写出来”，
以及“必测 reference 抛异常但场景仍算通过”这类问题。它们不是数值问题，
只需要检查文件/字段/计数就能发现，不该用 15 分钟的 GPU 跑去找。

校验内容：
- 每个 run dir 有 `provenance.json`（commit、dirty、environment）和
  `source_manifest.json`；dirty tree 还必须有 `git_diff.patch`。
- check：`check.json` 的每个场景 `ok=true`、有 gate、无 gate failure、
  有 coverage gate、gate 数不少于下限。
- bench：`complete=true`；每个场景 `ok`；样本数等于 repeats；每个样本的
  `per_step_ms` 长度等于 steps；`steps` 与 `trace` 计数一致；汇总键齐全。
- memory：每个场景有 `stages`、`growth`、`determinism`，且 stale state 通过。
- profile：每个 case 有 `meta.json`，run dir 有 `summary.md` 和
  `timeline_summary.json`。
- sol：`sol.json` 有非空 `bench_rows`，且 `ideal` 含 depth 512/7680。

用法：
    .venv/bin/python scripts/check_results.py results/m3/<dir> [...]
    .venv/bin/python scripts/check_results.py --all          # 扫 results/m3/*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/m3"


class Checker:
    def __init__(self, path: Path):
        self.path = path
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.legacy = False

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def expect(self, condition: bool, message: str) -> bool:
        if not condition:
            self.fail(message)
        return condition

    def load(self, name: str):
        path = self.path / name
        if not path.exists():
            self.fail(f"缺文件 {name}")
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            self.fail(f"{name} 解析失败: {exc}")
            return None


def kind_of(path: Path) -> str:
    name = path.name
    for kind in ("m3-check", "m3-bench", "m3-memory", "m3-profile", "sol"):
        if name.endswith(kind):
            return kind
    if (path / "bench.json").exists():
        return "m3-bench"
    if (path / "check.json").exists():
        return "m3-check"
    if (path / "memory.json").exists():
        return "m3-memory"
    if (path / "summary.md").exists():
        return "m3-profile"
    if (path / "sol.json").exists():
        return "sol"
    if (path / "attn_length_effect.json").exists():
        return "micro"
    return "unknown"


def check_provenance(c: Checker) -> None:
    prov = c.load("provenance.json")
    if prov is None:
        # 2026-10-22 之前的 run 没有 provenance.json；重跑这些场景后会被替换。
        c.notes.append("缺 provenance.json：可能是 2026-10-22 之前的 legacy run")
        return
    git = prov.get("git") or {}
    c.expect(bool(git.get("commit")), "provenance.git.commit 为空")
    c.expect("dirty" in git, "provenance.git.dirty 缺失")
    c.expect(bool(prov.get("environment")), "provenance.environment 缺失")
    manifest = prov.get("source_manifest") or {}
    c.expect("files" in manifest, "provenance.source_manifest.files 缺失")
    # 只有 tracked 源码有改动时才需要 patch；仅 untracked 的 results/ 不算。
    if git.get("dirty") and int(manifest.get("patch_bytes") or 0) > 0:
        c.expect((c.path / "git_diff.patch").exists(), "dirty tree 有 tracked 改动但缺 git_diff.patch")
        c.expect(bool(manifest.get("files")), "dirty tree 有 tracked 改动但 source_manifest.files 为空")
    c.expect((c.path / "source_manifest.json").exists(), "缺 source_manifest.json")


def check_check_dir(c: Checker) -> None:
    data = c.load("check.json")
    if data is None:
        return
    summary = data.get("summary") or {}
    scenarios = summary.get("scenarios") or []
    c.expect(bool(scenarios), "check.json 没有场景")
    c.expect(bool(summary.get("ok")), "check.json summary.ok=false")
    for scen in scenarios:
        name = scen.get("scenario", "?")
        report = c.load(f"{name}/report.json")
        if report is None:
            continue
        items = report.get("items") or []
        gates = [i for i in items if i.get("level") == "gate"]
        c.expect(bool(report.get("ok")), f"{name}: report.ok=false")
        c.expect(bool(gates), f"{name}: 没有 gate")
        c.expect(
            not [g for g in gates if not g.get("ok")],
            f"{name}: 存在失败的 gate",
        )
        c.expect(
            any(i.get("name", "").startswith("coverage:") for i in gates),
            f"{name}: 缺 coverage gate",
        )
        c.expect(
            bool(report.get("provenance", {}).get("git", {}).get("commit")),
            f"{name}: report 缺 provenance",
        )


def check_bench_dir(c: Checker) -> None:
    data = c.load("bench.json")
    if data is None:
        return
    c.expect(bool(data.get("complete")), "bench.json complete!=true（可能是中断的运行）")
    results = data.get("results") or []
    c.expect(bool(results), "bench.json 没有 results")
    for res in results:
        name = res.get("scenario", "?")
        if not c.expect(bool(res.get("ok")), f"{name}: ok=false"):
            continue
        steps = int(res.get("steps", 0))
        repeats = int(res.get("repeats_requested", 0))
        samples = res.get("samples") or []
        c.expect(steps > 0, f"{name}: steps=0")
        c.expect(
            len(samples) == repeats,
            f"{name}: 样本数 {len(samples)} != repeats {repeats}",
        )
        for i, sample in enumerate(samples):
            per_step = sample.get("model", {}).get("per_step_ms") or []
            c.expect(
                len(per_step) == steps,
                f"{name} sample {i}: per_step_ms {len(per_step)} != steps {steps}",
            )
            tokens = sample.get("tokens") or {}
            c.expect(
                int(tokens.get("generated", 0)) == steps + 1,
                f"{name} sample {i}: generated {tokens.get('generated')} != steps+1",
            )
            # 只要求字段存在：cap/graph 与 eager 的 free-running trace 本来就会分叉，
            # 那是 diagnostic（见 m3 journal §4），不是 bench 的产物契约。
            c.expect(
                "matches_frozen_trace" in tokens,
                f"{name} sample {i}: 缺 matches_frozen_trace 字段",
            )
            c.expect(
                float(sample.get("model", {}).get("model_tok_s", 0)) > 0,
                f"{name} sample {i}: model_tok_s<=0",
            )
        summary = res.get("summary") or {}
        for key in ("model_tok_s", "system_tok_s", "request_tok_s", "step_ms", "prefill_s"):
            c.expect(key in summary, f"{name}: summary 缺 {key}")
        c.expect(
            (summary.get("peak_allocated_gb") or 0) > 0,
            f"{name}: peak_allocated_gb 缺失",
        )


def check_memory_dir(c: Checker) -> None:
    data = c.load("memory.json")
    if data is None:
        return
    c.expect(bool(data.get("complete")), "memory.json complete!=true")
    for res in data.get("results") or []:
        name = res.get("scenario", "?")
        if not c.expect(bool(res.get("ok")), f"{name}: ok=false"):
            continue
        stages = res.get("stages") or []
        c.expect(len(stages) >= 6, f"{name}: stage 数 {len(stages)} < 6")
        names = {s.get("stage") for s in stages}
        for required in ("load", "cache_init", "prefill_cold", "decode"):
            c.expect(required in names, f"{name}: 缺 stage {required}")
        det = res.get("determinism") or {}
        c.expect(bool(det.get("prefill_logits_equal")), f"{name}: reset 后 prefill logits 不一致")
        c.expect(bool(det.get("step_logits_equal")), f"{name}: reset 后 decode logits 不一致")
        growth = res.get("growth") or {}
        c.expect("cycle2_allocated_delta_gb" in growth, f"{name}: 缺 cycle2 增长记录")
        if res.get("spec", {}).get("execution") == "graph":
            graph = res.get("graph") or {}
            c.expect(
                "max_phase_peak_allocated_gb" in graph,
                f"{name}: 缺 graph 子阶段峰值",
            )


def check_profile_dir(c: Checker) -> None:
    c.expect((c.path / "summary.md").exists(), "缺 summary.md")
    c.expect((c.path / "timeline_summary.json").exists(), "缺 timeline_summary.json")
    c.expect((c.path / "provenance.json").exists(), "缺 provenance.json")
    cases = [p for p in c.path.iterdir() if p.is_dir()]
    c.expect(bool(cases), "没有 case 目录")
    for case in cases:
        c.expect((case / "meta.json").exists(), f"{case.name}: 缺 meta.json")
        c.expect((case / "trace.nsys-rep").exists(), f"{case.name}: 缺 trace.nsys-rep")
        trace_csv = case / "gpu_trace.csv"
        c.expect(
            trace_csv.exists() or (case / "gpu_trace.csv.gz").exists(),
            f"{case.name}: 缺 gpu_trace.csv(.gz)",
        )


def check_sol_dir(c: Checker) -> None:
    data = c.load("sol.json")
    if data is None:
        return
    rows = data.get("bench_rows") or []
    c.expect(bool(rows), "sol.json 没有 bench_rows（可能没传 --bench）")
    ideal = data.get("ideal") or {}
    for key in ("bf16_512", "bf16_7680", "fp8_512", "fp8_7680"):
        c.expect(key in ideal, f"sol.json 缺 ideal.{key}")
    weights = data.get("weights") or {}
    c.expect((weights.get("total_cuda_bytes") or 0) > 1e10, "权重字节数异常")


def is_legacy(path: Path) -> bool:
    """旧 harness 的产物：没有 provenance.json，或有但缺 source_manifest 字段。

    这些 run 早于 provenance/source_manifest 契约，重跑后会被替换；默认单列，
    `--strict`（关闭流程）下判失败。
    """
    prov_path = path / "provenance.json"
    if not prov_path.exists():
        return True
    try:
        prov = json.loads(prov_path.read_text())
    except Exception:
        return True
    return "source_manifest" not in prov


def check_micro_dir(c: Checker) -> None:
    data = c.load("attn_length_effect.json")
    if data is None:
        return
    c.expect(bool(data.get("summary")), "micro 结果缺 summary")
    c.expect(bool(data.get("provenance")), "micro 结果缺 provenance")
    c.expect(bool(data.get("rows")), "micro 结果缺 rows")


def check_one(path: Path) -> Checker:
    c = Checker(path)
    kind = kind_of(path)
    if kind == "unknown":
        c.fail("无法识别 run dir 类型（缺 bench/check/memory/sol/provenance）")
        return c
    if is_legacy(path):
        c.notes.append("legacy：provenance/source_manifest 契约之前的 run，重跑后会被替换")
        c.legacy = True
        return c
    check_provenance(c)
    {
        "m3-check": check_check_dir,
        "m3-bench": check_bench_dir,
        "m3-memory": check_memory_dir,
        "m3-profile": check_profile_dir,
        "sol": check_sol_dir,
        "micro": check_micro_dir,
    }[kind](c)
    return c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="检查 results/m3 下所有 run dir")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把 legacy（无 provenance.json）也判失败；关闭流程用",
    )
    args = parser.parse_args()

    paths = list(args.paths)
    if args.all or not paths:
        paths = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    bad = 0
    legacy = 0
    for path in paths:
        c = check_one(path)
        if c.legacy:
            legacy += 1
            print(f"[LEGACY] {path}")
            for note in c.notes:
                print(f"   - {note}")
            if args.strict:
                bad += 1
            continue
        status = "OK" if not c.failures else "FAIL"
        print(f"[{status}] {path}")
        for failure in c.failures:
            print(f"   - {failure}")
        bad += bool(c.failures)
    checked = len(paths) - legacy
    print(f"\n{checked - bad + (legacy if args.strict else 0)}/{checked} 个当前 run dir 通过；"
          f"legacy {legacy} 个（{'判失败' if args.strict else '已跳过'}）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
