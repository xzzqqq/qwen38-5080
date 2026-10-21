#!/usr/bin/env python3
"""解析 Nsight Systems 导出的 CSV，给 M3 的时间线报告用。

输出（对应 docs/plans/m3.md §7.4）：
- 每步 kernel 数（graph 模式下等于展开的 node 数）；
- kernel 时长直方图；
- <5 µs kernel 的数量占比与累计时间占比；
- top-5 kernel（次数、总时长、占比）；
- 每步 kernel 时间和 vs 墙钟的差值（可见空隙的粗略上界）。

用法：
    .venv/bin/python scripts/m3_profile_stats.py --dir results/m3/<run>-m3-profile
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def resolve_csv(path: Path) -> Path | None:
    """接受 .csv 或 .csv.gz（大 CSV 入库时压缩，避免几十 MB 的明文）。"""
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    return None


def parse_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    resolved = resolve_csv(path)
    if resolved is None:
        return [], []
    opener = gzip.open if resolved.suffix == ".gz" else open
    if resolved.stat().st_size == 0:
        return [], []
    rows: list[list[str]] = []
    with opener(resolved, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if not row or all(not cell.strip() for cell in row):
                continue
            rows.append([cell.strip() for cell in row])
    if not rows:
        return [], []
    header_idx = None
    for idx, row in enumerate(rows):
        if any(cell.lower() == "name" for cell in row):
            header_idx = idx
            break
    if header_idx is None:
        return [], rows
    header = rows[header_idx]
    data = [row for row in rows[header_idx + 1 :] if len(row) == len(header)]
    return header, data


def col(header: list[str], *candidates: str) -> int | None:
    lowered = [h.lower() for h in header]
    for candidate in candidates:
        for idx, name in enumerate(lowered):
            if name == candidate.lower():
                return idx
    for candidate in candidates:
        for idx, name in enumerate(lowered):
            if candidate.lower() in name:
                return idx
    return None


def load_durations(path: Path) -> tuple[list[float], list[str]]:
    header, data = parse_csv(path)
    if not header:
        return [], []
    dur_idx = col(header, "Duration (ns)", "Duration")
    name_idx = col(header, "Name", "Kernel Name")
    if dur_idx is None or name_idx is None:
        return [], []
    durations, names = [], []
    for row in data:
        name = row[name_idx]
        lowered = name.lower()
        # cuda_gpu_trace 会把 memcpy/memset 也列出来；kernel 统计要排除它们。
        if "memcpy" in lowered or "memset" in lowered or lowered.startswith("[cuda"):
            continue
        try:
            durations.append(float(row[dur_idx]) / 1000.0)  # ns -> us
        except ValueError:
            continue
        names.append(name)
    return durations, names


def histogram(durations: list[float]) -> dict[str, dict[str, float]]:
    buckets = [
        ("<1us", 0.0, 1.0),
        ("1-2us", 1.0, 2.0),
        ("2-5us", 2.0, 5.0),
        ("5-10us", 5.0, 10.0),
        ("10-50us", 10.0, 50.0),
        ("50-100us", 50.0, 100.0),
        (">=100us", 100.0, float("inf")),
    ]
    total = len(durations) or 1
    total_time = sum(durations) or 1.0
    out = {}
    for label, low, high in buckets:
        count = sum(1 for d in durations if low <= d < high)
        duration = sum(d for d in durations if low <= d < high)
        out[label] = {
            "count": count,
            "count_pct": 100.0 * count / total,
            "time_us": duration,
            "time_pct": 100.0 * duration / total_time,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    root = args.dir
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    lines = ["# M3 Nsight Systems 时间线摘要", f"run_dir: {root}", ""]
    summary = {}
    for case in cases:
        meta_path = case / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        steps = int(meta.get("steps", 0))
        durations, names = load_durations(case / "gpu_trace.csv")
        if not durations:
            lines.append(f"## {case.name}")
            lines.append("")
            lines.append("没有解析到 kernel 明细（检查 `profile.log` / `stats.err`）。")
            lines.append("")
            continue
        hist = histogram(durations)
        total_us = sum(durations)
        top = defaultdict(lambda: {"count": 0, "us": 0.0})
        for duration, name in zip(durations, names):
            top[name]["count"] += 1
            top[name]["us"] += duration
        top_sorted = sorted(top.items(), key=lambda kv: -kv[1]["us"])[: args.top]
        wall_ms = [r["wall_ms"] for r in meta.get("rows", [])]
        per_step_kernels = len(durations) / steps if steps else len(durations)
        per_step_kernel_us = total_us / steps if steps else total_us
        per_step_wall_us = (sum(wall_ms) / len(wall_ms) * 1000) if wall_ms else None
        tiny = hist["<1us"]["count"] + hist["1-2us"]["count"] + hist["2-5us"]["count"]
        tiny_time = hist["<1us"]["time_us"] + hist["1-2us"]["time_us"] + hist["2-5us"]["time_us"]
        summary[case.name] = {
            "steps": steps,
            "kernels": len(durations),
            "per_step_kernels": per_step_kernels,
            "per_step_kernel_us": per_step_kernel_us,
            "per_step_wall_us": per_step_wall_us,
            "histogram": hist,
            "tiny_kernel_count_pct": 100.0 * tiny / len(durations),
            "tiny_kernel_time_pct": 100.0 * tiny_time / total_us,
            "top": [
                {"name": name, "count": value["count"], "total_us": value["us"],
                 "time_pct": 100.0 * value["us"] / total_us}
                for name, value in top_sorted
            ],
        }
        lines += [
            f"## {case.name}（{meta.get('execution')} depth={meta.get('depth')} "
            f"capacity={meta.get('capacity')} steps={steps}）",
            "",
            f"- kernel 实例：{len(durations)}，每步 {per_step_kernels:.0f} 个"
            + ("（graph 展开后 node 数）" if meta.get("execution") == "graph" else ""),
            f"- 每步 kernel 累计时间：{per_step_kernel_us:.1f} µs"
            + (f"，每步墙钟 {per_step_wall_us:.1f} µs（含 host 提交/同步/空隙）" if per_step_wall_us else ""),
            f"- <5 µs kernel：数量 {summary[case.name]['tiny_kernel_count_pct']:.1f}%，"
            f"累计时间 {summary[case.name]['tiny_kernel_time_pct']:.1f}%",
            "",
            "| 时长桶 | 数量 | 数量占比 | 累计时间 µs | 时间占比 |",
            "|---|---|---|---|---|",
        ]
        for label, value in hist.items():
            lines.append(
                f"| {label} | {value['count']} | {value['count_pct']:.1f}% | "
                f"{value['time_us']:.1f} | {value['time_pct']:.1f}% |"
            )
        lines += [
            "",
            "| top kernel | 次数 | 总时长 µs | 时间占比 |",
            "|---|---|---|---|",
        ]
        for item in summary[case.name]["top"]:
            lines.append(
                f"| `{item['name'][:70]}` | {item['count']} | {item['total_us']:.1f} | "
                f"{item['time_pct']:.1f}% |"
            )
        lines.append("")

    lines += [
        "",
        "## 复现",
        "",
        "- 采集：`STEPS=8 make m3-profile`（nsys 2026.2.1，`--capture-range=cudaProfilerApi`，"
        "`--cuda-graph-trace=node`）。",
        "- 原始 per-kernel CSV 以 `gpu_trace.csv.gz` 保存（36 MB → 1.2 MB）；"
        "`trace.sqlite` 可由 `nsys stats` 从 `.nsys-rep` 重新生成。",
        "- 统计：`.venv/bin/python scripts/m3_profile_stats.py --dir <run_dir>`。",
        "",
    ]
    (root / "timeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
