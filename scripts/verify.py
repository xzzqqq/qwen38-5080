#!/usr/bin/env python3
"""按“改了什么”推荐最小验证档位；`--run` 直接执行。

原则（AGENTS.md）：改一点先跑最小闭环，全量只在里程碑关闭前跑。这个脚本把
"哪些文件 → 哪几档验证"固化成表，避免每次凭记忆决定要不要跑 95 分钟的 bench。

档位：
    S0 静态      语法 / diff 检查 / 产物契约            < 10 s
    S1 CPU 单测   pytest tests/ops tests/model            2 s
    S2 单场景     m3-check --only graph-1024              2 min
    S2n reference m3-check --only numeric-512            3 min
    S3 快档       make m3-quick                          5 min
    S4 定向全量   m3-memory / m3-sol / m3-profile        1–12 min
    S5 关闭全量   m3-check（5 场景）/ m3-bench（11 场景） 15 / 95 min

用法：
    .venv/bin/python scripts/verify.py                # 只打印建议
    .venv/bin/python scripts/verify.py --run          # 顺序执行，失败即停
    .venv/bin/python scripts/verify.py --run --keep-going
    .venv/bin/python scripts/verify.py --files runner/model/decode_runner.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Tier:
    key: str
    title: str
    command: str
    seconds: int
    proves: str


TIERS: dict[str, Tier] = {
    "s0": Tier("s0", "S0 静态+产物契约", "", 10, "语法、diff 干净、results 契约"),
    "s1": Tier("s1", "S1 CPU 单测", ".venv/bin/python -m pytest tests/ops tests/model -q --disable-warnings", 5, "op 数值与 shape-stable 回归"),
    "s2": Tier("s2", "S2 单场景 graph", ".venv/bin/python scripts/m3_check.py --only graph-1024 --steps 16 --out .tmp/verify/check-graph", 120, "graph capture / capacity guard / static cache"),
    "s2n": Tier("s2n", "S2n 单场景 reference", ".venv/bin/python scripts/m3_check.py --only numeric-512 --steps 16 --out .tmp/verify/check-numeric", 180, "padded reference / reset / boundary / coverage"),
    "s3": Tier("s3", "S3 快档", "make m3-quick", 300, "graph + 显存 + 两套计时端到端"),
    "s4-mem": Tier("s4-mem", "S4 定向显存", "make m3-memory", 720, "分阶段峰值 / 生命周期 / stale state"),
    "s4-sol": Tier("s4-sol", "S4 SOL", "make m3-sol", 60, "实际字节账与 SOL 口径"),
    "s4-prof": Tier("s4-prof", "S4 时间线", "make m3-profile", 720, "kernel/node 数与时长分布"),
    "s5-check": Tier("s5-check", "S5 全量 check", "make m3-check", 900, "5 场景全部 gate"),
    "s5-bench": Tier("s5-bench", "S5 全量 bench", "make m3-bench", 5700, "11 场景 × 5 次性能矩阵"),
}

# 前缀 → 档位。顺序无关，取并集。
RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("runner/ops/", "tests/"), ("s0", "s1")),
    (("runner/loader/", "runner/model/host_embedding.py"), ("s2", "s3")),
    (("runner/model/static_cache.py", "runner/model/padded_reference.py", "runner/model/snapshots.py", "scripts/m3_check.py"), ("s2", "s2n")),
    (("runner/model/graph_runner.py", "scripts/m3_memory.py"), ("s2", "s4-mem")),
    (("runner/model/decode_runner.py", "bench/e2e/decode.py"), ("s3",)),
    (("runner/ops/nvfp4/bytes.py", "scripts/m3_sol.py"), ("s4-sol",)),
    (("scripts/m3_profile.py", "scripts/m3_profile.sh", "scripts/m3_profile_stats.py"), ("s4-prof",)),
    (("scripts/check_results.py", "runner/report.py"), ("s0",)),
    (("docs/", "AGENTS.md", "Makefile", "README.md"), ("s0",)),
]

CLOSE_HINT = (
    "关闭里程碑前必须跑全量：make m3-check && make m3-memory && make m3-bench "
    "&& make m3-profile && make m3-sol，然后用 scripts/check_results.py 校验产物。"
)


def changed_files(explicit: list[str] | None) -> list[str]:
    if explicit:
        return [Path(p).as_posix() for p in explicit]
    out: list[str] = []

    def run(cmd: list[str]) -> list[str]:
        try:
            text = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return []
        return [line for line in text.splitlines() if line.strip()]

    out += run(["git", "diff", "--name-only", "HEAD"])
    out += run(["git", "ls-files", "--others", "--exclude-standard"])
    # 只关心源码/文档；results/、.tmp/ 不触发档位。
    return sorted({p for p in out if not p.startswith(("results/", ".tmp/"))})


def tiers_for(files: list[str]) -> tuple[list[Tier], dict[str, list[str]]]:
    triggered: dict[str, list[str]] = {}
    for path in files:
        matched = False
        for prefixes, keys in RULES:
            if any(path == p or path.startswith(p) for p in prefixes):
                for key in keys:
                    triggered.setdefault(key, []).append(path)
                matched = True
        if not matched:
            triggered.setdefault("s0", []).append(path)
    order = list(TIERS)
    tiers = [TIERS[k] for k in order if k in triggered]
    return tiers, triggered


def run_command(command: str) -> bool:
    print(f"\n$ {command}", flush=True)
    started = time.perf_counter()
    code = subprocess.call(command, shell=True, cwd=ROOT)
    print(f"  -> exit={code} in {time.perf_counter() - started:.1f}s", flush=True)
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=None, help="指定改动文件（默认读 git）")
    parser.add_argument("--run", action="store_true", help="执行推荐档位")
    parser.add_argument("--keep-going", action="store_true", help="失败后继续")
    parser.add_argument("--include-s5", action="store_true", help="连全量一起跑（关闭时用）")
    args = parser.parse_args()

    files = changed_files(args.files)
    if not files:
        print("没有改动（或只改了 results/.tmp/）；最少跑 S0：")
        print("  .venv/bin/python scripts/check_results.py --all")
        print(CLOSE_HINT)
        return 0

    tiers, triggered = tiers_for(files)
    print("改动文件：")
    for path in files:
        print(f"  - {path}")
    print("\n推荐档位：")
    total = 0
    for tier in tiers:
        total += tier.seconds
        print(f"  [{tier.key:6s}] {tier.title:18s} ~{tier.seconds:4d}s  {tier.proves}")
        print(f"           {tier.command or '.venv/bin/python scripts/check_results.py --all'}")
        print(f"           触发：{', '.join(sorted(set(triggered[tier.key])))[:120]}")
    if args.include_s5:
        for key in ("s5-check", "s5-bench"):
            tier = TIERS[key]
            total += tier.seconds
            print(f"  [{tier.key:8s}] {tier.title:18s} ~{tier.seconds:4d}s  {tier.proves}")
    print(f"\n合计约 {total / 60:.1f} 分钟（不含全量时为最小闭环）")
    print(CLOSE_HINT)

    if not args.run:
        return 0

    commands = [".venv/bin/python scripts/check_results.py --all"]
    commands += [t.command for t in tiers if t.command and t.key != "s0"]
    if args.include_s5:
        commands += [TIERS["s5-check"].command, TIERS["s5-bench"].command]
    failed = 0
    for command in commands:
        ok = run_command(command)
        if not ok:
            failed += 1
            print(f"FAILED: {command}", flush=True)
            if not args.keep_going:
                break
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
